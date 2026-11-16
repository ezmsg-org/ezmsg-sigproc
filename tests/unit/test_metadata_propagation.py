"""Every processor carries the input message's ``attrs`` and ``segment`` through to its output.

``segment`` marks where a stream stops continuing itself (a time gap, a source
reconfiguration), and stream-level metadata rides in ``attrs``; both have to
survive every stage of a pipeline. Most stages get this for free from
``replace(message, ...)``; the ones that build their output from a template
made at reset time have to take them from the *current* message.

A change of ``segment`` must also reset every stateful stage: after a gap, a
processor's output must be what a fresh one makes of the post-gap messages.

``config_id`` is passed on too, or derived: a declared input gives declared
outputs, and the output's id changes whenever the input's does, or whenever a
settings change alters the output layout.

The inventory is pinned: every concrete processor class in the package must be
either covered by a case below or listed, with a reason, in ``EXEMPT``.
"""

import asyncio
import copy
import importlib
import importlib.util
import inspect
import pkgutil
import re
import sys
import warnings
from dataclasses import replace

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

ATTRS = {"run": 3, "unit": "uV"}
SEGMENT = 3
FS = 500.0
N_CHUNK = 250  # 0.5 s per message
N_MSGS = 6  # 3 s total: enough for every buffering stage to emit


def msg(i=0, n_time=N_CHUNK, n_ch=3, attrs=ATTRS, segment=SEGMENT):
    return AxisArray(
        np.random.default_rng(i).standard_normal((n_time, n_ch)) + 2.0,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=FS, offset=i * n_time / FS),
            "ch": AxisArray.CoordinateAxis(np.array([f"ch{c}" for c in range(n_ch)]), dims=["ch"]),
        },
        key="dev",
        attrs=attrs,
        stream_dim="time",
        segment=segment,
    )


def stream(start=0, attrs=ATTRS, segment=SEGMENT, n_ch=3):
    return [msg(i, n_ch=n_ch, attrs=attrs, segment=segment) for i in range(start, start + N_MSGS)]


def windowed(start=0, attrs=ATTRS, segment=SEGMENT):
    """(win, time, ch), as FBCCA sees it downstream of a window stage."""
    return [
        AxisArray(
            np.random.default_rng(i).standard_normal((2, N_CHUNK, 3)),
            dims=["win", "time", "ch"],
            axes={"win": AxisArray.TimeAxis(fs=2.0, offset=i), "time": AxisArray.TimeAxis(fs=FS)},
            key="dev",
            attrs=attrs,
            stream_dim="win",
            segment=segment,
        )
        for i in range(start, start + N_MSGS)
    ]


def spectral(start=0, attrs=ATTRS, segment=SEGMENT):
    return [
        AxisArray(
            np.abs(np.random.default_rng(i).standard_normal((N_CHUNK, 40))),
            dims=["time", "freq"],
            axes={"time": AxisArray.TimeAxis(fs=FS, offset=i * N_CHUNK / FS), "freq": AxisArray.LinearAxis(gain=1.0)},
            key="dev",
            attrs=attrs,
            stream_dim="time",
            segment=segment,
        )
        for i in range(start, start + N_MSGS)
    ]


def int_stream(start=0, attrs=ATTRS, segment=SEGMENT):
    """int16 samples, whose dtype a stage may or may not keep."""
    return [replace(m, data=(100 * m.data).astype(np.int16)) for m in stream(start, attrs, segment)]


def run(proc, msgs):
    """Feed every message; return a flat list of every AxisArray output."""
    outs = []
    for m in msgs:
        out = proc(m)
        outs.extend(out if isinstance(out, list) else [out])
    return outs


def _resample(proc, msgs):
    outs = []
    for m in msgs:
        proc(m)
        outs.append(next(proc))
    return outs


def _sampler(proc, msgs):
    from ezmsg.sigproc.sampler import SampleTriggerMessage

    outs = run(proc, msgs[:1])
    t0 = msgs[0].axes["time"].offset
    proc.push_trigger(SampleTriggerMessage(timestamp=t0 + 0.5, period=(0.0, 0.5)))
    outs += run(proc, msgs[1:])
    # Left pending: a stale trigger must not fire on data from after a gap.
    proc.push_trigger(SampleTriggerMessage(timestamp=msgs[-1].axes["time"].offset + 1.0, period=(0.0, 0.5)))
    return outs


# ---------------------------------------------------------------------------
# Cases: class name -> (factory, messages, driver)
# ---------------------------------------------------------------------------


def _cases():
    from ezmsg.sigproc import (
        activation,
        adaptive_lattice_notch,
        adaptive_lnc,
        affinetransform,
        aggregate,
        asarray,
        bandpower,
        binned_aggregate,
        butterworthfilter,
        butterworthzerophase,
        cheby,
        combfilter,
        coordinatespaces,
        decimate,
        denormalize,
        detrend,
        diff,
        digitize,
        downsample,
        ewma,
        extract_axis,
        fbcca,
        filter,
        filterbank,
        filterbankdesign,
        fir_hilbert,
        fir_pmc,
        firfilter,
        flatten,
        gaussiansmoothing,
        kaiser,
        linear,
        materialize,
        multibandpow,
        quantize,
        resample,
        rollingscaler,
        sampler,
        scaler,
        signalinjector,
        singlebandpow,
        slicer,
        spectrogram,
        spectrum,
        transpose,
        wavelets,
        window,
    )
    from ezmsg.sigproc.math import abs, add, anscombe, clip, difference, invert, log, pow, scale

    bw = butterworthfilter.ButterworthFilterSettings
    cheb = cheby.ChebyshevFilterSettings(order=2, ripple_tol=1.0, Wn=40.0)

    c = {
        "ActivationTransformer": lambda: activation.ActivationTransformer(),
        "AdaptiveLatticeNotchFilterTransformer": lambda: adaptive_lattice_notch.AdaptiveLatticeNotchFilterTransformer(),
        "AdaptiveLNCTransformer": lambda: adaptive_lnc.AdaptiveLNCTransformer(),
        "AffineTransformTransformer": lambda: affinetransform.AffineTransformTransformer(weights=np.eye(3)),
        "CommonRereferenceTransformer": lambda: affinetransform.CommonRereferenceTransformer(),
        "AggregateTransformer": lambda: aggregate.AggregateTransformer(axis="ch"),
        "AsArrayTransformer": lambda: asarray.AsArrayTransformer(),
        "BandPowerTransformer": lambda: bandpower.BandPowerTransformer(
            spectrogram_settings=spectrogram.SpectrogramSettings(window_dur=0.5, window_shift=0.25),
            bands=[(10.0, 20.0)],
        ),
        "BinnedAggregateTransformer": lambda: binned_aggregate.BinnedAggregateTransformer(),
        "ButterworthFilterTransformer": lambda: butterworthfilter.ButterworthFilterTransformer(order=2, cutoff=40.0),
        "ButterworthBackwardFilterTransformer": lambda: butterworthzerophase.ButterworthBackwardFilterTransformer(
            butterworthzerophase.ButterworthZeroPhaseSettings(order=2, cutoff=40.0)
        ),
        "ButterworthZeroPhaseTransformer": lambda: butterworthzerophase.ButterworthZeroPhaseTransformer(
            order=2, cutoff=40.0
        ),
        "ChebyshevFilterTransformer": lambda: cheby.ChebyshevFilterTransformer(cheb),
        "ChebyForDecimateTransformer": lambda: decimate.ChebyForDecimateTransformer(cheb),
        "CombFilterTransformer": lambda: combfilter.CombFilterTransformer(),
        # Draws random gains at every reset; seeded so a fresh one draws the same.
        "DenormalizeTransformer": lambda: denormalize.DenormalizeTransformer(seed=0),
        "DetrendTransformer": lambda: detrend.DetrendTransformer(),
        "DiffTransformer": lambda: diff.DiffTransformer(),
        "DigitizeTransformer": lambda: digitize.DigitizeTransformer(min_val=-5.0, max_val=5.0),
        "DownsampleTransformer": lambda: downsample.DownsampleTransformer(factor=2),
        "EWMATransformer": lambda: ewma.EWMATransformer(),
        "ExtractAxisData": lambda: extract_axis.ExtractAxisData(axis="time", reference="time"),
        "StreamingFBCCATransformer": lambda: fbcca.StreamingFBCCATransformer(
            time_dim="time", ch_dim="ch", freqs=[10.0, 12.0], window_dur=1.0, window_shift=0.5, subbands=2
        ),
        "FilterTransformer": lambda: filter.FilterTransformer(coefs=(np.array([0.5, 0.5]), np.array([1.0]))),
        "FilterbankTransformer": lambda: filterbank.FilterbankTransformer(kernels=[np.ones(5) / 5, np.ones(9) / 9]),
        "FilterbankDesignTransformer": lambda: filterbankdesign.FilterbankDesignTransformer(
            filters=[kaiser.KaiserFilterSettings(cutoff=[10.0, 20.0], ripple=40.0, width=5.0, pass_zero=False)]
        ),
        "FIRHilbertFilterTransformer": lambda: fir_hilbert.FIRHilbertFilterTransformer(),
        "FIRHilbertEnvelopeTransformer": lambda: fir_hilbert.FIRHilbertEnvelopeTransformer(),
        "ParksMcClellanFIRTransformer": lambda: fir_pmc.ParksMcClellanFIRTransformer(order=31, cuton=10.0, cutoff=40.0),
        "FIRFilterTransformer": lambda: firfilter.FIRFilterTransformer(order=31, cutoff=40.0),
        "FlattenTransformer": lambda: flatten.FlattenTransformer(),
        "GaussianSmoothingFilterTransformer": lambda: gaussiansmoothing.GaussianSmoothingFilterTransformer(),
        "KaiserFilterTransformer": lambda: kaiser.KaiserFilterTransformer(cutoff=40.0, ripple=40.0, width=10.0),
        "LinearTransformTransformer": lambda: linear.LinearTransformTransformer(scale=2.0),
        "MaterializeTransformer": lambda: materialize.MaterializeTransformer(),
        "AnalyticFilterbankTransformer": lambda: multibandpow.AnalyticFilterbankTransformer(bands=[(10.0, 20.0)]),
        "MultiBandEnvelopePowerTransformer": lambda: multibandpow.MultiBandEnvelopePowerTransformer(
            bands=[(10.0, 20.0)]
        ),
        "QuantizeTransformer": lambda: quantize.QuantizeTransformer(max_val=5.0),
        "RollingScalerProcessor": lambda: rollingscaler.RollingScalerProcessor(),
        "AdaptiveStandardScalerTransformer": lambda: scaler.AdaptiveStandardScalerTransformer(),
        # Draws a random mixing at every reset; seeded so a fresh one draws the same.
        "SignalInjectorTransformer": lambda: signalinjector.SignalInjectorTransformer(frequency=10.0, mixing_seed=0),
        "RMSBandPowerTransformer": lambda: singlebandpow.RMSBandPowerTransformer(
            bandpass=bw(order=2, cuton=10.0, cutoff=20.0)
        ),
        "SquareLawBandPowerTransformer": lambda: singlebandpow.SquareLawBandPowerTransformer(
            bandpass=bw(order=2, cuton=10.0, cutoff=20.0),
            lowpass=bw(order=2, cutoff=5.0),
            downsample=downsample.DownsampleSettings(factor=10),
        ),
        "SlicerTransformer": lambda: slicer.SlicerTransformer(selection="0:2", axis="ch"),
        "SpectrogramTransformer": lambda: spectrogram.SpectrogramTransformer(window_dur=0.5, window_shift=0.25),
        "SpectrumTransformer": lambda: spectrum.SpectrumTransformer(),
        "TransposeTransformer": lambda: transpose.TransposeTransformer(axes=("ch", "time")),
        "CWTTransformer": lambda: wavelets.CWTTransformer(frequencies=[10.0, 20.0], wavelet="morl"),
        "WindowTransformer": lambda: window.WindowTransformer(window_dur=0.5, window_shift=0.25),
        "ResampleProcessor": lambda: resample.ResampleProcessor(resample_rate=250.0),
        "SamplerTransformer": lambda: sampler.SamplerTransformer(buffer_dur=1.0),
        # math
        "AbsTransformer": lambda: abs.AbsTransformer(),
        "ConstAddTransformer": lambda: add.ConstAddTransformer(value=1.0),
        "AnscombeTransformer": lambda: anscombe.AnscombeTransformer(),
        "InverseAnscombeTransformer": lambda: anscombe.InverseAnscombeTransformer(),
        "ClipTransformer": lambda: clip.ClipTransformer(min=-1.0, max=1.0),
        "ConstDifferenceTransformer": lambda: difference.ConstDifferenceTransformer(value=1.0),
        "InvertTransformer": lambda: invert.InvertTransformer(),
        "LogTransformer": lambda: log.LogTransformer(clip_zero=True),
        "PowTransformer": lambda: pow.PowTransformer(),
        "ScaleTransformer": lambda: scale.ScaleTransformer(scale=2.0),
    }
    cases = {k: (v, stream, run) for k, v in c.items()}
    cases["CoordinateSpacesTransformer"] = (
        lambda: coordinatespaces.CoordinateSpacesTransformer(axis="ch"),
        lambda start=0, attrs=ATTRS, segment=SEGMENT: stream(start, attrs, segment, n_ch=2),
        run,
    )
    cases["FBCCATransformer"] = (
        lambda: fbcca.FBCCATransformer(time_dim="time", ch_dim="ch", freqs=[10.0, 12.0]),
        windowed,
        run,
    )
    cases["RangedAggregateTransformer"] = (
        lambda: aggregate.RangedAggregateTransformer(axis="freq", bands=[(10.0, 20.0)]),
        spectral,
        run,
    )
    cases["ResampleProcessor"] = (c["ResampleProcessor"], stream, _resample)
    cases["SamplerTransformer"] = (c["SamplerTransformer"], stream, _sampler)
    if importlib.util.find_spec("river") is not None:
        cases["RiverAdaptiveStandardScalerTransformer"] = (
            lambda: scaler.RiverAdaptiveStandardScalerTransformer(),
            stream,
            run,
        )
    return cases


CASES = _cases()

# Classes deliberately not driven by the single-stream cases above.
EXEMPT = {
    # Abstract base for the filter-by-design family; every concrete one is covered.
    "FilterByDesignTransformer",
    # Producer: has no input message to take attrs from.
    "TriggerProducer",
    # Numbers segments and stamps config_id itself rather than passing them through: covered by test_streamstamp.py.
    "StreamStampTransformer",
    # Optional dependency (river) absent: covered above when it is installed.
    "RiverAdaptiveStandardScalerTransformer",
    # Two-input processors: covered by test_two_input_processors_number_their_own_segments
    # and test_two_input_processors_join_config_ids.
    "AlignAlongAxisProcessor",
    "ConcatProcessor",
    "MergeProcessor",
    "ResampleConcatProcessor",
    "AddProcessor",
    "DifferenceProcessor",
}


def _discover():
    """Every processor class defined in ezmsg.sigproc, abstract ones included."""
    from ezmsg.baseproc import BaseConsumer, BaseProcessor, BaseProducer, CompositeProcessor

    import ezmsg.sigproc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for info in pkgutil.walk_packages(ezmsg.sigproc.__path__, "ezmsg.sigproc."):
            try:
                importlib.import_module(info.name)
            except Exception:  # optional deps (mlx, ...) may be absent
                pass

    found = set()
    bases = (BaseProcessor, BaseProducer, BaseConsumer, CompositeProcessor)
    for name, mod in list(sys.modules.items()):
        if not name.startswith("ezmsg.sigproc"):
            continue
        for attr, obj in vars(mod).items():
            if not inspect.isclass(obj) or obj.__module__ != name:
                continue
            try:
                is_proc = issubclass(obj, bases)
            except TypeError:  # 3.10 generic aliases pass isclass (see test_axis_deprecation)
                continue
            # The two-input processors are plain classes, so catch them by name.
            if is_proc or re.search(r"(Transformer|Processor|Producer)$", attr):
                found.add(attr)
    return found


class TestTheInventoryIsPinned:
    def test_every_processor_is_covered_or_exempt(self):
        found = _discover()
        assert found - set(CASES) - EXEMPT == set(), "new processor: add a case or an exemption"

    def test_no_stale_entries(self):
        found = _discover()
        assert (set(CASES) | EXEMPT) - found == set()


def _assert_carries(outs, attrs=ATTRS, segment=SEGMENT):
    assert outs, "the processor produced nothing to check"
    n_nonempty = 0
    for out in outs:
        assert isinstance(out, AxisArray), type(out)
        assert attrs.items() <= out.attrs.items(), out.attrs
        assert out.segment == segment
        n_nonempty += out.data.size > 0
    assert n_nonempty, "feed more data: no output had samples"


@pytest.mark.parametrize("name", sorted(CASES))
def test_metadata_propagates(name):
    factory, messages, drive = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = factory()
        outs = drive(proc, messages())
    _assert_carries(outs)


@pytest.mark.parametrize("name", sorted(CASES))
def test_metadata_follows_the_current_message(name):
    """``attrs`` are not in the message hash, so a change must not wait for a reset."""
    factory, messages, drive = CASES[name]
    second = {**ATTRS, "run": 4}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = factory()
        outs_a = drive(proc, messages())
        outs_b = drive(proc, messages(start=N_MSGS, attrs=second, segment=SEGMENT + 1))
    _assert_carries(outs_a)
    _assert_carries(outs_b, second, SEGMENT + 1)


# ---------------------------------------------------------------------------
# A new segment resets: after a gap, output is what a fresh processor makes of the post-gap messages.
# ---------------------------------------------------------------------------

GAP = 100.0  # s
DC_STEP = 50.0


def _after_gap(messages):
    """The next messages, but 100 s later, with a large DC step and a new segment."""
    out = []
    for m in messages(start=N_MSGS, segment=SEGMENT + 1):
        ax = replace(m.axes[m.stream_dim], offset=m.axes[m.stream_dim].offset + GAP)
        out.append(replace(m, data=m.data + DC_STEP, axes={**m.axes, m.stream_dim: ax}))
    return out


def _head(m, n):
    """The first ``n`` samples of ``m`` along its stream axis."""
    return replace(m, data=m.data[(slice(None),) * m.get_axis_idx(m.stream_dim) + (slice(0, n),)])


def _before_gap(messages):
    """A stream whose last message stops part way, leaving partial windows, bins and phases behind."""
    pre = messages()
    pre[-1] = _head(pre[-1], pre[-1].data.shape[pre[-1].get_axis_idx(pre[-1].stream_dim)] * 11 // 20)
    return pre


def _assert_same(outs, expected):
    assert len(outs) == len(expected)
    for out, exp in zip(outs, expected):
        assert out.dims == exp.dims and out.segment == exp.segment and out.attrs == exp.attrs
        assert out.axes == exp.axes
        np.testing.assert_allclose(np.asarray(out.data), np.asarray(exp.data), rtol=1e-9, atol=1e-9)


def _fresh(at_gap, factory, post, drive):
    return drive(factory(), post)


def _keeps_everything(at_gap, factory, post, drive):
    """As if the segment had not changed: the whole state survives the gap."""
    outs = drive(at_gap, [replace(m, segment=SEGMENT) for m in post])
    return [replace(o, segment=SEGMENT + 1) for o in outs]


def _fresh_keeping(attr):
    """A fresh processor started on the post-gap stream, but with ``state.<attr>`` carried over from before it."""

    def reference(at_gap, factory, post, drive):
        fresh = factory()
        fresh._prepare_state(post[0])
        setattr(fresh.state, attr, getattr(at_gap.state, attr))
        return drive(fresh, post)

    return reference


# Processors that deliberately keep slow state across a gap (their `_reset_segment`), and what to expect instead.
GAP_REFERENCE = {
    # The tracked line frequency survives a gap; phase, filter memory and warm start do not.
    "AdaptiveLNCTransformer": _fresh_keeping("omega"),
    # The tracked notch coefficient survives a gap; the filter memory does not.
    "AdaptiveLatticeNotchFilterTransformer": _fresh_keeping("k1"),
    # Exponentially-weighted statistics survive a gap unless `reset_on_resume`; Detrend inherits it from EWMA.
    "EWMATransformer": _keeps_everything,
    "DetrendTransformer": _keeps_everything,
    # Rolling statistics are counted in samples, so a gap in time does not date them.
    "RollingScalerProcessor": _keeps_everything,
    "AdaptiveStandardScalerTransformer": _keeps_everything,
    "RiverAdaptiveStandardScalerTransformer": _keeps_everything,
}


@pytest.mark.parametrize("empty_first", [False, True], ids=["", "empty-first"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_new_segment_matches_a_fresh_processor(name, empty_first):
    factory, messages, drive = CASES[name]
    post = _after_gap(messages)
    if empty_first:
        # The new segment arrives on a message with no samples; it must still take effect.
        post = [_head(post[0], 0)] + post
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = factory()
        drive(proc, _before_gap(messages))
        at_gap = copy.deepcopy(proc) if name in GAP_REFERENCE else None
        outs = drive(proc, post)
        expected = GAP_REFERENCE.get(name, _fresh)(at_gap, factory, post, drive)
    assert all(o.segment == SEGMENT + 1 for o in outs)
    _assert_same(outs, expected)


# State derived only from the stream configuration and settings: a gap has nothing in it to invalidate.
CONFIG_ONLY = [
    "AffineTransformTransformer",
    "CommonRereferenceTransformer",
    "DenormalizeTransformer",
    "FlattenTransformer",
    "LinearTransformTransformer",
    "RangedAggregateTransformer",
    "SignalInjectorTransformer",
    "SlicerTransformer",
    "SpectrumTransformer",
    "TransposeTransformer",
]


@pytest.mark.parametrize("name", CONFIG_ONLY)
def test_a_new_segment_leaves_config_only_state_alone(name):
    factory, messages, drive = CASES[name]
    msgs = messages()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = drive(factory(), msgs)
        proc = factory()
        outs = drive(proc, msgs[:1])
        before = dict(vars(proc.state))
        outs += drive(proc, [replace(m, segment=SEGMENT + 1) for m in msgs[1:]])
    assert all(getattr(proc.state, k) is v for k, v in before.items())
    _assert_same(outs, expected[:1] + [replace(o, segment=SEGMENT + 1) for o in expected[1:]])


def _filter_coefs(proc):
    return proc.state.filter.settings.coefs


def _filterbank_kernels(proc):
    return proc.state.filterbank.settings.kernels


# Stages whose design depends only on the config, and where to find it. The filter memory still resets at a gap.
DESIGNED = {
    **dict.fromkeys(
        [
            "ButterworthFilterTransformer",
            "ChebyshevFilterTransformer",
            "ChebyForDecimateTransformer",
            "CombFilterTransformer",
            "FIRFilterTransformer",
            "FIRHilbertFilterTransformer",
            "GaussianSmoothingFilterTransformer",
            "KaiserFilterTransformer",
            "ParksMcClellanFIRTransformer",
        ],
        _filter_coefs,
    ),
    "ButterworthBackwardFilterTransformer": lambda proc: proc._coefs_cache,
    "FilterbankDesignTransformer": _filterbank_kernels,
    "AnalyticFilterbankTransformer": _filterbank_kernels,
}


def short_chunks(start=0, attrs=ATTRS, segment=SEGMENT):
    """20-sample messages, shorter than a 50-tap kernel."""
    return [msg(i, n_time=20, attrs=attrs, segment=segment) for i in range(5 * start, 5 * (start + N_MSGS))]


def _designed_variants():
    from ezmsg.sigproc import filterbank, fir_hilbert, wavelets
    from ezmsg.sigproc.filterbank import FilterbankMode as FM
    from ezmsg.sigproc.util.minphase import MinPhaseMode
    from ezmsg.sigproc.util.startup import StartupMode as SM

    variants = {}
    for messages, kernels, tag in (
        (stream, [np.ones(5) / 5, np.ones(9) / 9], ""),
        (short_chunks, [np.ones(50) / 50, np.ones(30) / 30], "-short"),
    ):
        for mode in FM:
            for startup in SM:
                if mode == FM.FFT and startup in (SM.FIRST_CHUNK, SM.BUFFERED):
                    continue  # unsupported: raises
                variants[f"FilterbankTransformer{tag}-{mode.name}-{startup.name}"] = (
                    lambda k=kernels, m=mode, s=startup: filterbank.FilterbankTransformer(kernels=k, mode=m, startup=s),
                    messages,
                    run,
                    lambda proc: proc.state.prep_kerns,
                )
    for startup in (SM.CONSTANT, SM.FIRST_CHUNK, SM.BUFFERED):
        variants[f"FIRHilbertEnvelopeTransformer-{startup.name}"] = (
            lambda s=startup: fir_hilbert.FIRHilbertEnvelopeTransformer(startup=s),
            stream,
            run,
            lambda proc: _filter_coefs(proc.state.filter),
        )
    # A new CWT design would bring a new child filterbank, and with it new kernels.
    for wavelet, min_phase, messages, tag in (
        *(("morl", mp, stream, mp.name) for mp in MinPhaseMode),
        ("cmor1.5-1.0", MinPhaseMode.NONE, stream, "complex"),
        ("morl", MinPhaseMode.NONE, short_chunks, "short"),
    ):
        variants[f"CWTTransformer-{tag}"] = (
            lambda w=wavelet, mp=min_phase: wavelets.CWTTransformer(frequencies=[10.0, 20.0], wavelet=w, min_phase=mp),
            messages,
            run,
            lambda proc: proc.state.fbgen.state.prep_kerns,
        )
    return variants


DESIGNED_CASES = {**{k: (*CASES[k], v) for k, v in DESIGNED.items()}, **_designed_variants()}


def _at_double_rate(m):
    ax = m.axes[m.stream_dim]
    return replace(m, axes={**m.axes, m.stream_dim: replace(ax, gain=ax.gain / 2)})


@pytest.mark.parametrize("name", sorted(DESIGNED_CASES))
def test_a_new_segment_keeps_the_design_but_not_the_filter_memory(name):
    factory, messages, drive, design = DESIGNED_CASES[name]
    post = _after_gap(messages)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = factory()
        drive(proc, _before_gap(messages))
        coefs = design(proc)
        # A twin rather than a deepcopy: FFT-mode filterbank state holds scipy.fft callables, which do not copy.
        twin = factory()
        drive(twin, _before_gap(messages))
        uninterrupted = _keeps_everything(twin, factory, post, drive)
        outs = drive(proc, post)
        assert design(proc) is coefs
        _assert_same(outs, drive(factory(), post))
        with pytest.raises(AssertionError):
            _assert_same(outs, uninterrupted)
        # A new sample rate is a new config, so the filter is designed again.
        drive(proc, [_at_double_rate(m) for m in post])
    assert design(proc) is not coefs


def test_buffered_startup_releases_with_the_latest_attrs():
    from ezmsg.sigproc.butterworthfilter import ButterworthFilterTransformer
    from ezmsg.sigproc.util.startup import StartupMode

    proc = ButterworthFilterTransformer(order=4, cutoff=5.0, startup=StartupMode.BUFFERED)
    outs = [proc(msg(i, n_time=10, attrs={**ATTRS, "segment": i})) for i in range(N_MSGS * 10)]
    released = next(i for i, o in enumerate(outs) if o.data.size)
    assert released > 0, "nothing was held: lower the cutoff"
    assert all(o.attrs["segment"] == i for i, o in enumerate(outs))


def _two_input_cases():
    from ezmsg.sigproc.align import AlignAlongAxisProcessor, AlignAlongAxisSettings
    from ezmsg.sigproc.concat import ConcatProcessor, ConcatSettings
    from ezmsg.sigproc.math.add import AddProcessor
    from ezmsg.sigproc.math.difference import DifferenceProcessor
    from ezmsg.sigproc.merge import MergeProcessor, MergeSettings
    from ezmsg.sigproc.resampleconcat import ResampleConcatProcessor, ResampleConcatSettings

    def queued(proc, a, b):
        proc.push_a(a)
        proc.push_b(b)
        return [asyncio.run(proc.__acall__())]

    def merge(proc, a, b):
        return [o for o in (proc(a), proc.push_b(b)) if o is not None]

    def align(proc, a, b):
        return [o for pair in (proc(a), proc.push_b(b)) if pair is not None for o in pair]

    def resampleconcat(proc, a, b):
        proc.push_reference(a)
        proc.push_signal(b)
        out = next(proc)
        return [] if out is None else [out]

    return {
        "ConcatProcessor": (lambda: ConcatProcessor(ConcatSettings(axis="ch")), queued),
        "MergeProcessor": (lambda: MergeProcessor(MergeSettings(axis="ch")), merge),
        "AlignAlongAxisProcessor": (lambda: AlignAlongAxisProcessor(AlignAlongAxisSettings()), align),
        "ResampleConcatProcessor": (lambda: ResampleConcatProcessor(ResampleConcatSettings()), resampleconcat),
        "AddProcessor": (AddProcessor, queued),
        "DifferenceProcessor": (DifferenceProcessor, queued),
    }


@pytest.mark.parametrize("name", sorted(_two_input_cases()))
def test_two_input_processors_number_their_own_segments(name):
    """Both inputs carry the same attrs, so whatever merge rule applies keeps them.

    The output starts a new segment whenever *either* input's segment changes, numbered by the
    processor itself from 0, and nothing buffered before a change is paired with anything after it.
    """
    factory, feed = _two_input_cases()[name]
    n_ch_b = 3 if name in ("AddProcessor", "DifferenceProcessor") else 2
    proc = factory()
    seg_a, seg_b = 3, 7
    phases = {0: [], 1: [], 2: []}
    for i in range(3 * N_MSGS):
        if i == N_MSGS:
            seg_a += 1  # A breaks
        if i == 2 * N_MSGS:
            seg_b += 1  # then B does
        phase = i // N_MSGS
        outs = feed(proc, msg(i, segment=seg_a), msg(i, n_ch=n_ch_b, segment=seg_b))
        phases[phase] += outs
        for out in outs:
            if phase and out.data.size:
                assert out.axes["time"].value(0) >= msg(phase * N_MSGS).axes["time"].offset - 1e-9
    for phase, outs in phases.items():
        _assert_carries(outs, ATTRS, segment=phase)


# ---------------------------------------------------------------------------
# config_id: declared in, declared out; a change of the input's or of the layout reaches the output.
# ---------------------------------------------------------------------------

ID_A, ID_B = 101, 202


def _declared(messages, config_id, start=0):
    return [replace(m, config_id=config_id) for m in messages(start=start)]


def _forget_config_warnings(monkeypatch):
    """Forget which classes baseproc has already warned about, so each test sees its own warnings.

    baseproc warns once per class about a missing config_id, or about an output layout that changed
    under an unchanged config_id, so a silent run shows every stage on the way declares its ids.
    """
    from ezmsg.baseproc import stateful

    monkeypatch.setattr(stateful, "_warned_undeclared", set())
    monkeypatch.setattr(stateful, "_warned_unannounced", set())


def _config_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "ezmsg.baseproc.stateful"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_config_id_follows_the_input(name, monkeypatch, caplog):
    """A declared input gives declared outputs, and a new input config_id gives new output ones.

    The layout is the same in both runs, so a stateful stage does not reset: the new id has to
    reach the output anyway. Nothing on the way may warn about a missing or stale config_id.
    """
    factory, messages, drive = CASES[name]
    _forget_config_warnings(monkeypatch)
    with warnings.catch_warnings(), caplog.at_level("WARNING", logger="ezmsg.baseproc.stateful"):
        warnings.simplefilter("ignore")
        proc = factory()
        ids_a = {o.config_id for o in drive(proc, _declared(messages, ID_A))}
        ids_b = {o.config_id for o in drive(proc, _declared(messages, ID_B, start=N_MSGS))}
    assert ids_a and ids_b
    assert 0 not in ids_a | ids_b
    assert not ids_a & ids_b
    assert _config_warnings(caplog) == []


@pytest.mark.parametrize("name", sorted(_two_input_cases()))
def test_two_input_processors_join_config_ids(name):
    """The output's config_id changes whenever either input's does; Align keeps each input's own."""
    factory, feed = _two_input_cases()[name]
    n_ch_b = 3 if name in ("AddProcessor", "DifferenceProcessor") else 2
    proc = factory()
    phases = []
    for phase, (id_a, id_b) in enumerate([(11, 21), (12, 21), (12, 22)]):
        outs = []
        for i in range(phase * N_MSGS, (phase + 1) * N_MSGS):
            a = replace(msg(i), config_id=id_a)
            b = replace(msg(i, n_ch=n_ch_b), config_id=id_b)
            outs += feed(proc, a, b)
        phases.append({o.config_id for o in outs if o.data.size})
    if name == "AlignAlongAxisProcessor":
        assert phases == [{11, 21}, {12, 21}, {12, 22}]
    else:
        assert all(len(ids) == 1 and 0 not in ids for ids in phases)
        assert len(set.union(*phases)) == 3


def _b_cases():
    """Stages whose output layout depends on their settings: (factory, messages, layout change, other change, driver).

    The first settings change alters the output layout and must change the output's config_id; the second
    (None where the class has none) leaves the layout alone and must not.
    """
    from ezmsg.sigproc import (
        activation,
        affinetransform,
        aggregate,
        asarray,
        binned_aggregate,
        coordinatespaces,
        digitize,
        downsample,
        extract_axis,
        filterbank,
        filterbankdesign,
        flatten,
        kaiser,
        multibandpow,
        quantize,
        resample,
        slicer,
        spectrum,
        transpose,
        wavelets,
        window,
    )
    from ezmsg.sigproc.math import scale

    k = kaiser.KaiserFilterSettings
    band = k(cutoff=[10.0, 20.0], ripple=40.0, width=5.0, pass_zero=False)
    return {
        "AffineTransformTransformer": (
            lambda: affinetransform.AffineTransformTransformer(weights=np.eye(3).tolist()),
            stream,
            {"weights": np.ones((3, 2)).tolist()},
            {"weights": (2 * np.eye(3)).tolist()},
            run,
        ),
        "RangedAggregateTransformer": (
            lambda: aggregate.RangedAggregateTransformer(axis="freq", bands=[(10.0, 20.0)]),
            spectral,
            {"bands": [(10.0, 20.0), (20.0, 30.0)]},
            {"operation": aggregate.AggregationFunction.MAX},
            run,
        ),
        "BinnedAggregateTransformer": (
            lambda: binned_aggregate.BinnedAggregateTransformer(),
            stream,
            {"bin_duration": 0.05},
            {"operation": aggregate.AggregationFunction.MAX},
            run,
        ),
        "DownsampleTransformer": (
            lambda: downsample.DownsampleTransformer(factor=2),
            stream,
            {"factor": 5},
            {"factor": None, "target_rate": FS / 2},
            run,
        ),
        "FilterbankTransformer": (
            lambda: filterbank.FilterbankTransformer(kernels=[[0.2] * 5, [0.1] * 10]),
            stream,
            {"kernels": [[0.2] * 5, [0.1] * 10, [0.5] * 2]},
            {"kernels": [[0.25] * 4, [0.125] * 8]},
            run,
        ),
        "FilterbankDesignTransformer": (
            lambda: filterbankdesign.FilterbankDesignTransformer(filters=[band]),
            stream,
            {"filters": [band, k(cutoff=[30.0, 40.0], ripple=40.0, width=5.0, pass_zero=False)]},
            {"filters": [k(cutoff=[10.0, 20.0], ripple=60.0, width=5.0, pass_zero=False)]},
            run,
        ),
        "AnalyticFilterbankTransformer": (
            lambda: multibandpow.AnalyticFilterbankTransformer(bands=[(10.0, 20.0)]),
            stream,
            {"bands": [(10.0, 20.0), (30.0, 40.0)]},
            {"transition_ratio": 0.4},
            run,
        ),
        "FlattenTransformer": (
            lambda: flatten.FlattenTransformer(),
            stream,
            {"sample_axis": "sample"},
            # One axis is flattened, so there is nothing to separate.
            {"label_separator": "_"},
            run,
        ),
        "SlicerTransformer": (
            lambda: slicer.SlicerTransformer(selection="0:2", axis="ch"),
            stream,
            {"selection": "1:3"},
            {"on_empty": "raise"},
            run,
        ),
        "SpectrumTransformer": (
            lambda: spectrum.SpectrumTransformer(),
            stream,
            {"output": spectrum.SpectralOutput.FULL},
            {"window": spectrum.WindowFunction.HANNING},
            run,
        ),
        "TransposeTransformer": (
            lambda: transpose.TransposeTransformer(axes=("ch", "time")),
            stream,
            {"axes": ("time", "ch")},
            {"order": "C"},
            run,
        ),
        "CWTTransformer": (
            lambda: wavelets.CWTTransformer(frequencies=[10.0, 20.0], wavelet="morl"),
            stream,
            {"frequencies": [10.0, 20.0, 30.0]},
            None,
            run,
        ),
        "WindowTransformer": (
            lambda: window.WindowTransformer(window_dur=0.5, window_shift=0.25),
            stream,
            {"window_dur": 0.4},
            {"buffer_update_strategy": "on_demand"},
            run,
        ),
        "WindowTransformer-anchor": (
            lambda: window.WindowTransformer(window_dur=0.5, window_shift=0.25),
            stream,
            # Read live, without a reset, and moves the within-window axis.
            {"anchor": window.Anchor.END},
            None,
            run,
        ),
        "ResampleProcessor": (
            lambda: resample.ResampleProcessor(resample_rate=250.0),
            stream,
            {"resample_rate": 200.0},
            {"fill_value": "last"},
            _resample,
        ),
        # Stateless stages.
        "AggregateTransformer": (
            lambda: aggregate.AggregateTransformer(axis="ch"),
            stream,
            {"axis": "time"},
            None,
            run,
        ),
        "ExtractAxisData": (
            lambda: extract_axis.ExtractAxisData(axis="ch", reference="time"),
            # Undeclared stream_dim: extracting `ch` leaves no `time` dimension to declare.
            lambda start=0: [replace(m, stream_dim=None) for m in stream(start)],
            {"axis": "time", "reference": "time"},
            # Read only for a linear axis.
            {"reference": "ch"},
            run,
        ),
        "CoordinateSpacesTransformer": (
            lambda: coordinatespaces.CoordinateSpacesTransformer(axis="ch"),
            lambda start=0: stream(start, n_ch=2),
            {"mode": coordinatespaces.CoordinateMode.POL2CART},
            None,
            run,
        ),
        "AsArrayTransformer": (
            lambda: asarray.AsArrayTransformer(),
            stream,
            {"dtype": "float32"},
            {"mlx_cache_limit_mb": 256.0},
            run,
        ),
        "DigitizeTransformer": (
            lambda: digitize.DigitizeTransformer(min_val=-5.0, max_val=5.0),
            stream,
            {"dtype": "int32"},
            # Reaches only the attrs.
            {"max_val": 4.0},
            run,
        ),
        "QuantizeTransformer": (
            lambda: quantize.QuantizeTransformer(max_val=5.0),
            stream,
            {"bits": 16},
            # Still uint8.
            {"bits": 7},
            run,
        ),
        "ActivationTransformer": (
            lambda: activation.ActivationTransformer(function=activation.ActivationFunction.SIGMOID),
            int_stream,
            {"function": activation.ActivationFunction.NONE},
            # The same function.
            {"function": "expit"},
            run,
        ),
        "ScaleTransformer": (
            lambda: scale.ScaleTransformer(scale=2.0),
            int_stream,
            {"scale": 1.0},
            {"scale": 3.0},
            run,
        ),
    }


B_CASES = _b_cases()


@pytest.mark.parametrize("name", sorted(B_CASES))
def test_a_settings_change_reaches_config_id_only_if_it_changes_the_layout(name, monkeypatch, caplog):
    factory, messages, layout_change, other_change, drive = B_CASES[name]
    _forget_config_warnings(monkeypatch)
    start = 0

    def out_ids(proc):
        nonlocal start
        outs = drive(proc, _declared(messages, ID_A, start=start))
        start += N_MSGS
        assert outs
        return {o.config_id for o in outs}

    with warnings.catch_warnings(), caplog.at_level("WARNING", logger="ezmsg.baseproc.stateful"):
        warnings.simplefilter("ignore")
        proc = factory()
        before = out_ids(proc)
        assert len(before) == 1 and 0 not in before
        if other_change is not None:
            proc.update_settings(replace(proc.settings, **other_change))
            assert out_ids(proc) == before
        proc.update_settings(replace(proc.settings, **layout_change))
        after = out_ids(proc)
    assert len(after) == 1 and 0 not in after and after != before
    assert _config_warnings(caplog) == []


def test_binned_aggregate_passthrough_carries_the_input_config_id():
    """``passthrough`` changes the layout without a reset: pass-through output has the input's id, binned output not."""
    from ezmsg.sigproc.binned_aggregate import BinnedAggregateTransformer

    proc = BinnedAggregateTransformer()
    binned = {o.config_id for o in run(proc, _declared(stream, ID_A))}
    proc.update_settings(replace(proc.settings, passthrough=True))
    passed = {o.config_id for o in run(proc, _declared(stream, ID_A, start=N_MSGS))}
    proc.update_settings(replace(proc.settings, passthrough=False))
    rebinned = {o.config_id for o in run(proc, _declared(stream, ID_A, start=2 * N_MSGS))}
    assert passed == {ID_A}
    assert binned == rebinned and ID_A not in binned


def test_window_anchor_moves_the_within_window_axis_without_a_reset():
    from ezmsg.sigproc.window import Anchor, WindowTransformer

    switched, kept = (WindowTransformer(window_dur=0.5, window_shift=0.25) for _ in range(2))
    run(switched, _declared(stream, ID_A))
    run(kept, _declared(stream, ID_A))
    switched.update_settings(replace(switched.settings, anchor=Anchor.END))
    after = run(switched, _declared(stream, ID_A, start=N_MSGS))[-1]
    ref = run(kept, _declared(stream, ID_A, start=N_MSGS))[-1]
    assert after.axes["time"].offset == -0.5
    # Each window's first sample keeps its absolute time: the window axis moves by as much as the samples' offset.
    assert after.axes["win"].offset + after.axes["time"].offset == pytest.approx(ref.axes["win"].offset)


def test_spectrum_follows_the_stream_length_under_one_config_id():
    """Transforming along the stream dimension, whose length config_id leaves out: a new length is a new layout."""
    from ezmsg.sigproc.spectrum import SpectrumTransformer

    proc = SpectrumTransformer()
    outs = [proc(replace(msg(i, n_time=n), config_id=ID_A)) for i, n in enumerate((250, 250, 100, 100))]
    assert [o.data.shape[0] for o in outs] == [126, 126, 51, 51]
    ids = [o.config_id for o in outs]
    assert ids[0] == ids[1] != ids[2] == ids[3]


@pytest.mark.parametrize("name", sorted(B_CASES))
def test_a_steady_stream_derives_its_config_id_once(name, monkeypatch):
    """The output's id is derived when the input's changes, not per message; an undeclared input gives 0."""
    from ezmsg.sigproc.util import segment

    factory, messages, _, _, drive = B_CASES[name]
    calls = []

    def counted(*args):
        calls.append(args)
        return chain_config_id(*args)

    chain_config_id = segment.chain_config_id
    monkeypatch.setattr(segment, "chain_config_id", counted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = factory()
        drive(proc, _declared(messages, ID_A))
        n_calls = len(calls)
        steady = drive(proc, _declared(messages, ID_A, start=N_MSGS))
        undeclared = drive(factory(), messages())
    assert steady and len(calls) == n_calls
    assert {o.config_id for o in undeclared} == {0}


# ---------------------------------------------------------------------------
# End of stream: a source ends a segment with one message with no samples, in a new segment
# (baseproc's ``end_of_stream``), and the unit flushes before that message and before
# whatever follows it. The empty message must change nothing about what the real data
# around it gives, and a flush with nothing real held must not emit samples.
# ---------------------------------------------------------------------------


def _eos_cases():
    from ezmsg.sigproc import (
        adaptive_lnc,
        binned_aggregate,
        butterworthfilter,
        detrend,
        downsample,
        ewma,
        filterbank,
        filterbankdesign,
        fir_hilbert,
        kaiser,
        multibandpow,
        rollingscaler,
        scaler,
        window,
    )
    from ezmsg.sigproc.aggregate import AggregationFunction as AF
    from ezmsg.sigproc.util.startup import StartupMode as SM

    band = kaiser.KaiserFilterSettings(cutoff=[10.0, 20.0], ripple=40.0, width=5.0, pass_zero=False)
    variants = {
        # Start-up modes that build state from the first samples after a reset.
        **{
            f"ButterworthFilterTransformer-{m.name}": lambda m=m: butterworthfilter.ButterworthFilterTransformer(
                order=4, cutoff=5.0, startup=m
            )
            for m in (SM.FIRST_CHUNK, SM.BUFFERED)
        },
        **{
            f"FilterbankTransformer-{m.name}": lambda m=m: filterbank.FilterbankTransformer(
                kernels=[np.ones(5) / 5, np.ones(9) / 9], startup=m
            )
            for m in (SM.ZERO, SM.FIRST_CHUNK, SM.BUFFERED)
        },
        "FilterbankDesignTransformer-BUFFERED": lambda: filterbankdesign.FilterbankDesignTransformer(
            filters=[band], startup=SM.BUFFERED
        ),
        **{
            f"FIRHilbertEnvelopeTransformer-{m.name}": lambda m=m: fir_hilbert.FIRHilbertEnvelopeTransformer(startup=m)
            for m in (SM.FIRST_CHUNK, SM.BUFFERED)
        },
        "MultiBandEnvelopePowerTransformer-BUFFERED": lambda: multibandpow.MultiBandEnvelopePowerTransformer(
            bands=[(10.0, 20.0)], startup=SM.BUFFERED
        ),
        **{
            f"AdaptiveLNCTransformer-{w.name}-{c.name}": lambda w=w, c=c: adaptive_lnc.AdaptiveLNCTransformer(
                warm_start=w, cancel_method=c
            )
            for w in adaptive_lnc.WarmStart
            for c in (adaptive_lnc.CancelMethod.NOTCH, adaptive_lnc.CancelMethod.SUBTRACT)
        },
        # Statistics seeded from the first samples.
        "EWMATransformer-reset_on_resume": lambda: ewma.EWMATransformer(reset_on_resume=True),
        "DetrendTransformer-reset_on_resume": lambda: detrend.DetrendTransformer(reset_on_resume=True),
        "AdaptiveStandardScalerTransformer-reset_on_resume": lambda: scaler.AdaptiveStandardScalerTransformer(
            reset_on_resume=True
        ),
        "RollingScalerProcessor-update_with_signal": lambda: rollingscaler.RollingScalerProcessor(
            update_with_signal=True
        ),
        # Grids and phases placed from the first message.
        "BinnedAggregateTransformer-minmax": lambda: binned_aggregate.BinnedAggregateTransformer(
            operation=(AF.MIN, AF.MAX)
        ),
        "BinnedAggregateTransformer-fixed": lambda: binned_aggregate.BinnedAggregateTransformer(
            bin_duration=0.03, fractional=False
        ),
        "DownsampleTransformer-3": lambda: downsample.DownsampleTransformer(factor=3),
        "WindowTransformer-1to1": lambda: window.WindowTransformer(window_dur=0.5),
        "WindowTransformer-no-pad": lambda: window.WindowTransformer(
            window_dur=0.5, window_shift=0.2, zero_pad_until="none"
        ),
        # Shift longer than the window: samples between windows are skipped, counted across messages.
        "WindowTransformer-sparse": lambda: window.WindowTransformer(window_dur=0.2, window_shift=0.3),
    }
    return {**CASES, **{k: (v, stream, run) for k, v in variants.items()}}


EOS_CASES = _eos_cases()


def _one(drive):
    """Driver for a lone message, without per-stream extras such as the sampler's triggers."""
    return _resample if drive is _resample else run


def _no_samples(out):
    return out is None or out.data.size == 0


def _assert_finite(outs):
    """For what has no reference to compare with; the rest may be as non-finite as the data makes it."""
    for out in outs:
        if out is not None and np.asarray(out.data).dtype.kind in "fc":
            assert np.all(np.isfinite(np.asarray(out.data))), out


def _assert_same_ids(outs, expected):
    _assert_same(outs, expected)
    assert [o.config_id for o in outs] == [e.config_id for e in expected]


def _assert_same_tail(tail, expected):
    """A flush gives one message, or nothing (None or no samples) when nothing real is held."""
    if _no_samples(expected):
        assert _no_samples(tail), tail
    else:
        _assert_same_ids([tail], [expected])


@pytest.mark.parametrize("name", sorted(EOS_CASES))
def test_an_empty_first_message_changes_nothing(name):
    """S1: a fresh processor's first message has no samples (and an offset the data does not continue)."""
    factory, messages, drive = EOS_CASES[name]
    data = _declared(messages, ID_A)
    empty = _head(data[0], 0)
    ax = empty.axes[empty.stream_dim]
    empty = replace(empty, axes={**empty.axes, empty.stream_dim: replace(ax, offset=ax.offset - GAP)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = factory()
        first = _one(drive)(proc, [empty])
        outs = drive(proc, data)
        tail = proc.flush()
        ref = factory()
        expected = drive(ref, data)
        ref_tail = ref.flush()
    assert all(_no_samples(o) for o in first)
    _assert_finite(first + [tail])
    _assert_same_ids(outs, expected)
    _assert_same_tail(tail, ref_tail)


@pytest.mark.parametrize("short", [False, True], ids=["", "short"])
@pytest.mark.parametrize("third", [False, True], ids=["data-in-marker-segment", "data-in-third-segment"])
@pytest.mark.parametrize("name", sorted(EOS_CASES))
def test_an_end_of_stream_marker_changes_nothing(name, third, short):
    """S2/S3/S4: data in A, flush, the marker in B, (flush twice,) data in B or C, flush.

    The reference is the same sequence without the marker and its flushes, so a processor that
    keeps some state across a segment change is held to that too. ``short`` data after the
    marker is too little to finish a start-up, a window or a bin, so the final flush releases it.
    """
    from ezmsg.baseproc import end_of_stream

    factory, messages, drive = EOS_CASES[name]
    pre = [replace(m, config_id=ID_A) for m in _before_gap(messages)]
    post = [replace(m, config_id=ID_A, segment=SEGMENT + 1 + third) for m in _after_gap(messages)]
    if short:
        post = [_head(post[0], 7)]
    marker = replace(end_of_stream(pre[-1]), segment=SEGMENT + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = factory()
        drive(ref, pre)
        ref.flush()
        expected = drive(ref, post)
        ref_tail = ref.flush()
        proc = factory()
        drive(proc, pre)
        held = proc.flush()
        at_marker = _one(drive)(proc, [marker])
        # S4: the stream may end here; flushing again holds nothing either.
        gap_tails = [proc.flush(), proc.flush()] if third else []
        outs = drive(proc, post)
        tail = proc.flush()
    assert all(_no_samples(o) for o in at_marker + gap_tails)
    assert all(o.segment == SEGMENT + 1 for o in at_marker if o is not None)
    _assert_finite([held] + at_marker + gap_tails + [tail])
    _assert_same_ids(outs, expected)
    _assert_same_tail(tail, ref_tail)
