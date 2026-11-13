"""Data-primed filter start-up (``startup``): FIRST_CHUNK and BUFFERED modes."""

import logging

import numpy as np
import pytest
import scipy.signal as sps
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.butterworthfilter import ButterworthFilterSettings, ButterworthFilterTransformer
from ezmsg.sigproc.butterworthzerophase import (
    ButterworthBackwardFilterTransformer,
    ButterworthZeroPhaseSettings,
    ButterworthZeroPhaseTransformer,
)
from ezmsg.sigproc.cheby import ChebyshevFilterSettings, ChebyshevFilterTransformer
from ezmsg.sigproc.combfilter import CombFilterSettings, CombFilterTransformer
from ezmsg.sigproc.filter import FilterSettings, FilterTransformer
from ezmsg.sigproc.filterbank import FilterbankMode, FilterbankSettings, FilterbankTransformer
from ezmsg.sigproc.filterbankdesign import FilterbankDesignSettings, FilterbankDesignTransformer
from ezmsg.sigproc.fir_hilbert import (
    FIRHilbertEnvelopeTransformer,
    FIRHilbertFilterSettings,
    FIRHilbertFilterTransformer,
)
from ezmsg.sigproc.fir_pmc import ParksMcClellanFIRSettings, ParksMcClellanFIRTransformer
from ezmsg.sigproc.firfilter import FIRFilterSettings, FIRFilterTransformer
from ezmsg.sigproc.gaussiansmoothing import GaussianSmoothingFilterTransformer, GaussianSmoothingSettings
from ezmsg.sigproc.kaiser import KaiserFilterSettings, KaiserFilterTransformer
from ezmsg.sigproc.util.minphase import MinPhaseMode
from ezmsg.sigproc.util.startup import StartupMode

FS = 500.0
T0 = 10.0
CONSTANT, FIRST_CHUNK, BUFFERED = StartupMode.CONSTANT, StartupMode.FIRST_CHUNK, StartupMode.BUFFERED


def _msg(x, fs: float = FS, offset: float = T0) -> AxisArray:
    x = x if x.ndim > 1 else x[:, None]
    return AxisArray(
        x,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=offset),
            "ch": AxisArray.CoordinateAxis(data=np.arange(x.shape[1]), dims=["ch"]),
        },
    )


def _feed(proc, x, chunk: int | None = None) -> list[AxisArray]:
    chunk = chunk or len(x)
    return [proc(_msg(x[s : s + chunk], offset=T0 + s / FS)) for s in range(0, len(x), chunk)]


def _cat(outs: list[AxisArray]) -> tuple[np.ndarray, np.ndarray]:
    """Data concatenated on time (moved to axis 0) and per-sample timestamps."""
    ys, ts = [], []
    for out in outs:
        t_ix = out.get_axis_idx("time")
        ax = out.axes["time"]
        ys.append(np.moveaxis(np.asarray(out.data), t_ix, 0))
        ts.append(ax.offset + np.arange(out.data.shape[t_ix]) * ax.gain)
    return np.concatenate(ys), np.concatenate(ts)


def _run(proc, x, chunk: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    return _cat(_feed(proc, x, chunk))


def _settle(y, ref, tol: float = 0.05) -> int:
    """Samples until ``y`` stays within ``tol`` of ``ref``."""
    bad = np.nonzero(np.abs(y - ref).reshape(len(y), -1).max(axis=1) > tol)[0]
    return int(bad[-1] + 1) if bad.size else 0


def _tone(n: int, lead: int = 0, f: float = 20.0) -> np.ndarray:
    """A tone that crosses zero upwards at sample ``lead``."""
    return np.sin(2 * np.pi * f * (np.arange(n + lead) - lead) / FS)


def _bandpass_iir(startup, **kw):
    kw.setdefault("coef_type", "sos")
    return ButterworthFilterTransformer(
        ButterworthFilterSettings(order=4, cuton=18.0, cutoff=22.0, startup=startup, **kw)
    )


def _bandpass_fir(startup, **kw):
    return FIRFilterTransformer(
        FIRFilterSettings(order=400, cutoff=[15.0, 25.0], pass_zero=False, startup=startup, **kw)
    )


# --- Settling ---------------------------------------------------------------------------------


@pytest.mark.parametrize("make", [_bandpass_iir, _bandpass_fir], ids=["iir", "fir"])
def test_stationary_tone_settles_quickly(make):
    """A pass-band tone that starts at its mean (an upward zero crossing).

    Measured settling to 5% of amplitude: IIR (4th-order 18-22 Hz band-pass) 272 samples unprimed vs 0
    primed; FIR (400-tap 15-25 Hz) 259 vs 0. The odd reflection continues a tone exactly only when it
    starts at its mean. Started at its peak instead, the reflection adds a DC step of 2 * x[0] and the
    primed transient is longer than the unprimed one (IIR 378 vs 267 samples, FIR 301 vs 265).
    """
    lead, n = 3000, 2000
    x = _tone(n, lead)
    ref = make(CONSTANT)(_msg(x)).data[lead:]
    settle_const = _settle(make(CONSTANT)(_msg(x[lead:])).data, ref)
    assert settle_const > 200
    for startup in (FIRST_CHUNK, BUFFERED):
        y, _ = _run(make(startup), x[lead:], chunk=None if startup == FIRST_CHUNK else 50)
        assert _settle(y, ref) <= 0.05 * settle_const


def test_sloped_start_has_small_transient():
    """A drifting signal (offset, slope and a slow wobble) through a 1 Hz high-pass.

    Measured max |error| against the response with real history: 0.0052 primed vs 0.17 unprimed.
    """
    lead, n = 5000, 2000
    t = (np.arange(n + lead) - lead) / FS
    x = 3.0 + 2.0 * t + 0.3 * np.sin(2 * np.pi * 0.2 * t)

    def make(startup):
        return ButterworthFilterTransformer(
            ButterworthFilterSettings(order=2, cuton=1.0, coef_type="sos", startup=startup)
        )

    ref = make(CONSTANT)(_msg(x)).data[lead:]
    err_const = np.abs(make(CONSTANT)(_msg(x[lead:])).data - ref).max()
    err_primed = np.abs(_run(make(BUFFERED), x[lead:], chunk=40)[0] - ref).max()
    assert err_primed < 0.1 * err_const


# --- BUFFERED accounting ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda s: _bandpass_iir(s),
        lambda s: _bandpass_iir(s, coef_type="ba"),
        lambda s: _bandpass_fir(s, fir_fft_min_taps=10**9),  # time-domain FIR path is chunk-invariant
    ],
    ids=["sos", "ba", "fir"],
)
def test_buffered_is_chunk_invariant(make):
    x = np.random.default_rng(0).standard_normal((900, 3))
    y_ref, t_ref = _run(make(BUFFERED), x)
    for chunk in (1, 7, 64):
        y, t = _run(make(BUFFERED), x, chunk)
        np.testing.assert_array_equal(y, y_ref)
        # Timestamps are rebuilt from per-chunk offsets, which this harness rounds differently per chunking.
        np.testing.assert_allclose(t, t_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(t_ref, T0 + np.arange(len(x)) / FS, rtol=0, atol=1e-9)


def test_buffered_holds_then_releases_everything():
    x = np.random.default_rng(1).standard_normal((600, 2)).astype(np.float32)
    proc = _bandpass_iir(BUFFERED)
    outs = _feed(proc, x, chunk=1)
    n_pad = proc.state.filter.state.startup_n_pad
    assert 0 < n_pad < len(x) - 1
    normal = outs[-1]
    for out in outs[:n_pad]:
        assert out.data.shape == (0, 2)
        assert out.dims == normal.dims
        assert out.data.dtype == normal.data.dtype
    released = outs[n_pad]
    assert released.data.shape == (n_pad + 1, 2)
    assert released.axes["time"].offset == T0
    assert all(o.data.shape[0] == 1 for o in outs[n_pad + 1 :])
    assert sum(o.data.shape[0] for o in outs) == len(x)


def test_first_chunk_adds_no_latency():
    x = np.random.default_rng(2).standard_normal((300, 2))
    for make in (_bandpass_iir, _bandpass_fir):
        outs = _feed(make(FIRST_CHUNK), x, chunk=70)
        assert outs[0].data.shape == (70, 2)
        assert outs[0].axes["time"].offset == T0
        assert all(o.data.shape[0] == min(70, len(x) - 70 * i) for i, o in enumerate(outs))


# --- Inferred pad length ----------------------------------------------------------------------


def _n_pad(proc, n: int = 10) -> int:
    proc(_msg(np.zeros(n)))
    return proc.state.filter.state.startup_n_pad


def test_fir_pad_is_taps_minus_one():
    proc = _bandpass_fir(FIRST_CHUNK)
    assert _n_pad(proc) == len(proc.state.filter.settings.coefs[0]) - 1 == 399
    # After min-phase conversion the applied kernel is half length.
    proc = _bandpass_fir(FIRST_CHUNK, min_phase=MinPhaseMode.HOMOMORPHIC)
    assert _n_pad(proc) == len(proc.state.filter.settings.coefs[0]) - 1 == 199
    raw = FilterTransformer(FilterSettings(coefs=(np.ones(33) / 33, np.array([1.0])), startup=FIRST_CHUNK))
    raw(_msg(np.zeros(10)))
    assert raw.state.startup_n_pad == 32


def test_iir_pad_grows_as_band_narrows_and_is_capped():
    def make(width, **kw):
        return ButterworthFilterTransformer(
            ButterworthFilterSettings(order=4, cuton=20 - width / 2, cutoff=20 + width / 2, startup=BUFFERED, **kw)
        )

    pads = [_n_pad(make(w, startup_max_dur=None)) for w in (20.0, 5.0, 2.0, 0.5)]
    assert pads == sorted(pads) and len(set(pads)) == len(pads)
    assert pads[-1] > FS  # longer than the default 1 s cap
    assert _n_pad(make(0.5)) == int(FS)  # default cap
    assert _n_pad(make(20.0, startup_max_dur=0.02)) == 10


def test_pole_on_unit_circle():
    coefs = (np.array([1.0]), np.array([1.0, -1.01]))  # pole outside the unit circle
    with pytest.raises(ValueError, match="never settles"):
        FilterTransformer(FilterSettings(coefs=coefs, startup=BUFFERED, startup_max_dur=None))(_msg(np.zeros(5)))
    proc = FilterTransformer(FilterSettings(coefs=coefs, startup=BUFFERED, startup_max_dur=0.2))
    proc(_msg(np.zeros(5)))
    assert proc.state.startup_n_pad == 100


def test_requires_linear_axis():
    msg = AxisArray(
        np.zeros((20, 1)), dims=["time", "ch"], axes={"time": AxisArray.CoordinateAxis(np.arange(20.0), ["time"])}
    )
    proc = FilterTransformer(FilterSettings(coefs=(np.ones(5) / 5, np.array([1.0])), startup=FIRST_CHUNK))
    with pytest.raises(TypeError, match="LinearAxis"):
        proc(msg)


# --- Composite processors ---------------------------------------------------------------------


def test_hilbert_envelope_primed():
    """With a tone starting at its mean the reflection is the tone's true past, so a primed envelope
    must match the one computed with real history from the first sample. That holds only if the
    Hilbert FIR state and the real part's delay buffer were primed from the same samples."""
    lead, n = 400, 1000
    x = _tone(n, lead)

    def make(startup):
        return FIRHilbertEnvelopeTransformer(FIRHilbertFilterSettings(order=170, f_lo=5.0, f_hi=100.0, startup=startup))

    ref = make(CONSTANT)(_msg(x)).data[lead:]
    y_const, _ = _run(make(CONSTANT), x[lead:], 50)
    assert _settle(y_const, ref) > 150
    proc = make(BUFFERED)
    outs = _feed(proc, x[lead:], 50)
    assert outs[0].data.shape == (0, 1) and outs[0].data.dtype == outs[-1].data.dtype == np.float32
    y, t = _cat(outs)
    np.testing.assert_allclose(y, ref, atol=1e-5)
    np.testing.assert_allclose(t, T0 + np.arange(n) / FS, rtol=0, atol=1e-9)
    # The envelope primes; its inner filter must not prime or hold on its own.
    assert proc.state.filter.settings.startup == CONSTANT
    assert proc.state.filter.state.filter.settings.startup == CONSTANT


KERNELS = [sps.firwin(101, [15.0, 25.0], fs=FS, pass_zero=False), sps.firwin(51, 40.0, fs=FS)]


def test_filterbank_conv_primed_and_chunk_invariant():
    lead, n = 300, 900
    x = _tone(n, lead)[:, None] * np.array([1.0, -2.0])
    ref = FilterbankTransformer(FilterbankSettings(kernels=KERNELS))(_msg(x)).data[..., lead:]

    def make(startup):
        return FilterbankTransformer(FilterbankSettings(kernels=KERNELS, startup=startup))

    y_const, _ = _run(make(CONSTANT), x[lead:], 50)
    assert _settle(y_const, np.moveaxis(ref, -1, 0)) > 50
    y_ref, t_ref = _run(make(BUFFERED), x[lead:])
    np.testing.assert_allclose(y_ref, np.moveaxis(ref, -1, 0), atol=1e-12)
    for chunk in (1, 7, 64):
        outs = _feed(make(BUFFERED), x[lead:], chunk)
        assert outs[0].data.shape[-1] == 0 and outs[0].dims == outs[-1].dims
        assert outs[0].data.dtype == outs[-1].data.dtype
        y, t = _cat(outs)
        np.testing.assert_allclose(y, y_ref, atol=1e-12)
        np.testing.assert_allclose(t, t_ref, rtol=0, atol=1e-12)


def test_filterbank_first_chunk_no_latency():
    x = np.random.default_rng(3).standard_normal((300, 2))
    outs = _feed(FilterbankTransformer(FilterbankSettings(kernels=KERNELS, startup=FIRST_CHUNK)), x, 120)
    assert outs[0].data.shape[-1] == 120 and outs[0].axes["time"].offset == T0


def test_filterbank_design_forwards_startup():
    proc = FilterbankDesignTransformer(
        FilterbankDesignSettings(
            filters=[KaiserFilterSettings(cutoff=[15.0, 25.0], ripple=40.0, width=5.0, pass_zero=False)],
            startup=BUFFERED,
        )
    )
    out = proc(_msg(np.zeros((5, 2))))
    assert out.data.shape[-1] == 0
    assert proc.state.filterbank.settings.startup == BUFFERED


# --- Delay compensation, logging, reset -------------------------------------------------------


def test_compensate_delay_timestamps_unchanged():
    x = np.random.default_rng(4).standard_normal((800, 2))
    _, t_const = _run(_bandpass_fir(CONSTANT, compensate_delay=True), x, 50)
    first = _feed(_bandpass_fir(FIRST_CHUNK, compensate_delay=True), x, 50)
    const = _feed(_bandpass_fir(CONSTANT, compensate_delay=True), x, 50)
    assert [o.axes["time"].offset for o in first] == [o.axes["time"].offset for o in const]
    buf = _feed(_bandpass_fir(BUFFERED, compensate_delay=True), x, 50)
    n_held = sum(o.data.shape[0] == 0 for o in buf)
    # The released message carries the first input's (compensated) time; later ones match one for one.
    assert buf[n_held].axes["time"].offset == const[0].axes["time"].offset
    assert [o.axes["time"].offset for o in buf[n_held + 1 :]] == [o.axes["time"].offset for o in const[n_held + 1 :]]
    np.testing.assert_allclose(_cat(buf)[1], t_const, rtol=0, atol=1e-12)
    assert t_const[0] == pytest.approx(T0 - 399 / 2 / FS)

    def fb(startup):
        return FilterbankTransformer(FilterbankSettings(kernels=KERNELS, compensate_delay=True, startup=startup))

    np.testing.assert_allclose(_run(fb(BUFFERED), x, 50)[1], _run(fb(CONSTANT), x, 50)[1], rtol=0, atol=1e-12)

    def env(startup):
        return FIRHilbertEnvelopeTransformer(
            FIRHilbertFilterSettings(order=170, f_lo=5.0, f_hi=100.0, compensate_delay=True, startup=startup)
        )

    np.testing.assert_allclose(_run(env(BUFFERED), x, 50)[1], _run(env(CONSTANT), x, 50)[1], rtol=0, atol=1e-12)


@pytest.mark.parametrize("startup", [FIRST_CHUNK, BUFFERED])
def test_delay_logged_once(caplog, startup):
    with caplog.at_level(logging.INFO, logger="ezmsg"):
        _run(_bandpass_fir(startup), np.zeros((600, 1)), 50)
    assert len([r for r in caplog.records if "FIR filter" in r.getMessage()]) == 1


def test_reset_reprimes():
    proc = _bandpass_iir(BUFFERED)
    _feed(proc, np.random.default_rng(5).standard_normal((600, 3)), 50)
    n_pad = proc.state.filter.state.startup_n_pad
    # A stream change (different channel count) resets: output is held again, then released whole.
    x_b = np.random.default_rng(6).standard_normal((600, 2))
    outs = _feed(proc, x_b, 50)
    n_held = (n_pad + 1 + 49) // 50 - 1
    assert all(o.data.shape == (0, 2) for o in outs[:n_held])
    y, t = _cat(outs)
    y_fresh, t_fresh = _run(_bandpass_iir(BUFFERED), x_b, 50)
    np.testing.assert_array_equal(y, y_fresh)
    np.testing.assert_array_equal(t, t_fresh)

    # A settings change that resets (a new filter order) re-primes too.
    fir = FIRFilterTransformer(FIRFilterSettings(order=9, cutoff=30.0, startup=BUFFERED))
    _feed(fir, x_b, 50)
    fir.update_settings(order=21)
    out = fir(_msg(x_b[:10]))
    assert out.data.shape == (0, 2) and fir.state.filter.state.startup_n_pad == 20


# --- Backends ---------------------------------------------------------------------------------


@pytest.mark.parametrize("make", [_bandpass_iir, _bandpass_fir], ids=["sos-metal", "fir-conv"])
def test_mlx(make):
    mx = pytest.importorskip("mlx.core")
    x = np.random.default_rng(7).standard_normal((900, 3)).astype(np.float32)
    y_np, t_np = _run(make(BUFFERED), x, 64)
    proc = make(BUFFERED)
    outs = [proc(_msg(mx.array(x[s : s + 64]), offset=T0 + s / FS)) for s in range(0, len(x), 64)]
    assert all(isinstance(o.data, mx.array) for o in outs)
    assert outs[0].data.shape == (0, 3) and outs[0].data.dtype == outs[-1].data.dtype == mx.float32
    y, t = _cat(outs)
    np.testing.assert_allclose(y, y_np, rtol=1e-3, atol=1e-4)
    np.testing.assert_array_equal(t, t_np)
    first = make(FIRST_CHUNK)(_msg(mx.array(x)))
    np.testing.assert_allclose(np.asarray(first.data), make(FIRST_CHUNK)(_msg(x)).data, rtol=1e-3, atol=1e-4)


# --- Where the setting applies, or raises -----------------------------------------------------


DESIGNS = {
    "butterworth": lambda s: ButterworthFilterTransformer(ButterworthFilterSettings(order=4, cutoff=30.0, startup=s)),
    "chebyshev": lambda s: ChebyshevFilterTransformer(
        ChebyshevFilterSettings(order=4, ripple_tol=1.0, Wn=30.0, coef_type="sos", startup=s)
    ),
    "comb": lambda s: CombFilterTransformer(CombFilterSettings(fundamental_freq=50.0, num_harmonics=2, startup=s)),
    "kaiser": lambda s: KaiserFilterTransformer(KaiserFilterSettings(cutoff=30.0, ripple=60.0, width=10.0, startup=s)),
    "fir": lambda s: FIRFilterTransformer(FIRFilterSettings(order=64, cutoff=30.0, startup=s)),
    "pmc": lambda s: ParksMcClellanFIRTransformer(
        ParksMcClellanFIRSettings(order=40, cuton=5.0, cutoff=30.0, startup=s)
    ),
    "gaussian": lambda s: GaussianSmoothingFilterTransformer(GaussianSmoothingSettings(startup=s)),
    "hilbert": lambda s: FIRHilbertFilterTransformer(FIRHilbertFilterSettings(order=64, f_lo=5.0, startup=s)),
}


@pytest.mark.parametrize("name", DESIGNS)
def test_design_transformers_forward_startup(name):
    proc = DESIGNS[name](BUFFERED)
    out = proc(_msg(np.zeros((1, 2))))
    assert out.data.shape == (0, 2)
    assert proc.state.filter.settings.startup == BUFFERED
    assert proc.state.filter.state.startup_n_pad > 0


@pytest.mark.parametrize("startup", [FIRST_CHUNK, BUFFERED])
def test_unsupported_raise(startup):
    x = np.zeros((2000, 2))
    with pytest.raises(ValueError, match="FFT mode"):
        FilterbankTransformer(FilterbankSettings(kernels=KERNELS, mode=FilterbankMode.FFT, startup=startup))(_msg(x))
    long_kernels = [sps.firwin(1001, 30.0, fs=FS)] * 3  # AUTO resolves to FFT
    with pytest.raises(ValueError, match="resolved from mode=FilterbankMode.AUTO"):
        FilterbankTransformer(FilterbankSettings(kernels=long_kernels, mode=FilterbankMode.AUTO, startup=startup))(
            _msg(x)
        )
    with pytest.raises(ValueError, match="FFT mode"):
        FilterbankDesignTransformer(
            FilterbankDesignSettings(
                filters=[KaiserFilterSettings(cutoff=30.0, ripple=60.0, width=10.0)],
                mode=FilterbankMode.FFT,
                startup=startup,
            )
        )(_msg(x))
    zp = ButterworthZeroPhaseSettings(order=4, cutoff=30.0, startup=startup)
    with pytest.raises(ValueError, match="zero-phase"):
        ButterworthZeroPhaseTransformer(zp)(_msg(x))
    with pytest.raises(ValueError, match="zero-phase"):
        ButterworthBackwardFilterTransformer(zp)(_msg(x))


@pytest.mark.parametrize("chunk", [7, 50, 400])
def test_filterbank_constant_startup_removes_offset_step(chunk):
    """A DC offset is a step to a zero pre-history; CONSTANT holds the first sample instead."""
    kernels = [sps.firwin(101, 30.0, fs=FS), sps.firwin(101, [8.0, 12.0], pass_zero=False, fs=FS)]
    x = np.full(400, 1000.0)

    def run(startup):
        proc = FilterbankTransformer(FilterbankSettings(kernels=kernels, startup=startup))
        outs = [proc(_msg(x[i : i + chunk])) for i in range(0, x.size, chunk)]
        return np.concatenate([o.data for o in outs], axis=-1), outs[0].axes["time"].offset

    y_zero, t_zero = run(StartupMode.ZERO)
    y_const, t_const = run(CONSTANT)
    assert t_const == t_zero and y_const.shape == y_zero.shape  # no latency, nothing withheld
    # Every kernel sits at its steady state (offset x DC gain) from the first sample on.
    for k_ix, k in enumerate(kernels):
        np.testing.assert_allclose(y_const[..., k_ix, :], 1000.0 * k.sum(), rtol=1e-9)
    assert np.abs(y_zero[..., 1, :]).max() > 100.0  # the step a zero pre-history rings with


def test_filter_rejects_zero_startup():
    with pytest.raises(ValueError, match="startup=ZERO"):
        _bandpass_fir(StartupMode.ZERO)(_msg(np.zeros(100)))
