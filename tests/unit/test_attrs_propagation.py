"""Every processor carries the input message's ``attrs`` through to its output.

Stream-level metadata (for instance a counter marking time gaps) rides in
``AxisArray.attrs`` and has to survive every stage of a pipeline. Most stages
get this for free from ``replace(message, ...)``; the ones that build their
output from a template made at reset time have to take ``attrs`` from the
*current* message, since ``attrs`` are not part of the message hash and can
change without a reset.

The inventory is pinned: every concrete processor class in the package must be
either covered by a case below or listed, with a reason, in ``EXEMPT``.
"""

import asyncio
import importlib
import importlib.util
import inspect
import pkgutil
import re
import sys
import warnings

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

ATTRS = {"segment": 3, "unit": "uV"}
FS = 500.0
N_CHUNK = 250  # 0.5 s per message
N_MSGS = 6  # 3 s total: enough for every buffering stage to emit


def msg(i=0, n_time=N_CHUNK, n_ch=3, attrs=ATTRS):
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
    )


def stream(start=0, attrs=ATTRS, n_ch=3):
    return [msg(i, n_ch=n_ch, attrs=attrs) for i in range(start, start + N_MSGS)]


def windowed(start=0, attrs=ATTRS):
    """(win, time, ch), as FBCCA sees it downstream of a window stage."""
    return [
        AxisArray(
            np.random.default_rng(i).standard_normal((2, N_CHUNK, 3)),
            dims=["win", "time", "ch"],
            axes={"win": AxisArray.TimeAxis(fs=2.0, offset=i), "time": AxisArray.TimeAxis(fs=FS)},
            key="dev",
            attrs=attrs,
            stream_dim="win",
        )
        for i in range(start, start + N_MSGS)
    ]


def spectral(start=0, attrs=ATTRS):
    return [
        AxisArray(
            np.abs(np.random.default_rng(i).standard_normal((N_CHUNK, 40))),
            dims=["time", "freq"],
            axes={"time": AxisArray.TimeAxis(fs=FS, offset=i * N_CHUNK / FS), "freq": AxisArray.LinearAxis(gain=1.0)},
            key="dev",
            attrs=attrs,
            stream_dim="time",
        )
        for i in range(start, start + N_MSGS)
    ]


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
    return outs + run(proc, msgs[1:])


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
        "DenormalizeTransformer": lambda: denormalize.DenormalizeTransformer(),
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
        "SignalInjectorTransformer": lambda: signalinjector.SignalInjectorTransformer(frequency=10.0),
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
        lambda start=0, attrs=ATTRS: stream(start, attrs, n_ch=2),
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
    # Optional dependency (river) absent: covered above when it is installed.
    "RiverAdaptiveStandardScalerTransformer",
    # Two-input processors: covered by TestTwoInputProcessors.
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


def _assert_carries(outs, attrs=ATTRS):
    assert outs, "the processor produced nothing to check"
    n_nonempty = 0
    for out in outs:
        assert isinstance(out, AxisArray), type(out)
        assert attrs.items() <= out.attrs.items(), out.attrs
        n_nonempty += out.data.size > 0
    assert n_nonempty, "feed more data: no output had samples"


@pytest.mark.parametrize("name", sorted(CASES))
def test_attrs_propagate(name):
    factory, messages, drive = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = factory()
        outs = drive(proc, messages())
    _assert_carries(outs)


@pytest.mark.parametrize("name", sorted(CASES))
def test_attrs_follow_the_current_message(name):
    """``attrs`` are not in the message hash, so a change must not wait for a reset."""
    factory, messages, drive = CASES[name]
    second = {**ATTRS, "segment": 4}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = factory()
        outs_a = drive(proc, messages())
        outs_b = drive(proc, messages(start=N_MSGS, attrs=second))
    _assert_carries(outs_a)
    _assert_carries(outs_b, second)


def test_buffered_startup_releases_with_the_latest_attrs():
    from ezmsg.sigproc.butterworthfilter import ButterworthFilterTransformer
    from ezmsg.sigproc.util.startup import StartupMode

    proc = ButterworthFilterTransformer(order=4, cutoff=5.0, startup=StartupMode.BUFFERED)
    outs = [proc(msg(i, n_time=10, attrs={**ATTRS, "segment": i})) for i in range(N_MSGS * 10)]
    released = next(i for i, o in enumerate(outs) if o.data.size)
    assert released > 0, "nothing was held: lower the cutoff"
    assert all(o.attrs["segment"] == i for i, o in enumerate(outs))


class TestTwoInputProcessors:
    """Both inputs carry the same attrs, so whatever merge rule applies keeps them."""

    def test_concat(self):
        from ezmsg.sigproc.concat import ConcatProcessor, ConcatSettings

        proc = ConcatProcessor(ConcatSettings(axis="ch"))
        proc.push_a(msg(0))
        proc.push_b(msg(0))
        _assert_carries([asyncio.run(proc.__acall__())])

    def test_merge(self):
        from ezmsg.sigproc.merge import MergeProcessor, MergeSettings

        proc = MergeProcessor(MergeSettings(axis="ch"))
        outs = [proc(msg(0)), proc.push_b(msg(0)), proc(msg(1)), proc.push_b(msg(1))]
        _assert_carries([o for o in outs if o is not None])

    def test_align(self):
        from ezmsg.sigproc.align import AlignAlongAxisProcessor, AlignAlongAxisSettings

        proc = AlignAlongAxisProcessor(AlignAlongAxisSettings())
        assert proc(msg(0)) is None
        a, b = proc.push_b(msg(0, n_ch=2))
        _assert_carries([a, b])

    def test_resampleconcat(self):
        from ezmsg.sigproc.resampleconcat import ResampleConcatProcessor, ResampleConcatSettings

        proc = ResampleConcatProcessor(ResampleConcatSettings())
        outs = []
        for i in range(N_MSGS):
            proc.push_reference(msg(i))
            proc.push_signal(msg(i, n_ch=2))
            outs.append(next(proc))
        _assert_carries([o for o in outs if o is not None])

    @pytest.mark.parametrize("which", ["add", "difference"])
    def test_math(self, which):
        mod = importlib.import_module(f"ezmsg.sigproc.math.{which}")
        proc = getattr(mod, f"{which.capitalize()}Processor")()
        proc.push_a(msg(0))
        proc.push_b(msg(0))
        _assert_carries([proc()])
