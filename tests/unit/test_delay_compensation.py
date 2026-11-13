"""Filter delay reporting and compensation on the time axis (``compensate_delay``)."""

import logging

import numpy as np
import pytest
import scipy.signal as sps
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.butterworthfilter import ButterworthFilterTransformer
from ezmsg.sigproc.butterworthzerophase import (
    ButterworthBackwardFilterTransformer,
    ButterworthZeroPhaseSettings,
    ButterworthZeroPhaseTransformer,
)
from ezmsg.sigproc.cheby import ChebyshevFilterTransformer
from ezmsg.sigproc.combfilter import CombFilterTransformer
from ezmsg.sigproc.filter import FilterSettings, FilterTransformer
from ezmsg.sigproc.filterbank import FilterbankMode, FilterbankSettings, FilterbankTransformer
from ezmsg.sigproc.filterbankdesign import FilterbankDesignSettings, FilterbankDesignTransformer
from ezmsg.sigproc.fir_hilbert import FIRHilbertEnvelopeTransformer
from ezmsg.sigproc.firfilter import FIRFilterTransformer
from ezmsg.sigproc.kaiser import KaiserFilterSettings
from ezmsg.sigproc.util.minphase import MinPhaseMode, is_linear_phase, kernel_delay, min_phase_kernel

FS = 500.0
T0 = 10.0


def _msg(x: np.ndarray, fs: float = FS, offset: float = T0) -> AxisArray:
    x = x if x.ndim > 1 else x[:, None]
    return AxisArray(
        x,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=offset),
            "ch": AxisArray.CoordinateAxis(data=np.arange(x.shape[1]), dims=["ch"]),
        },
    )


def _run(proc, x: np.ndarray, chunk: int | None = None, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Feed ``x`` in chunks; return (data concatenated on time, per-sample timestamps)."""
    chunk = chunk or len(x)
    ys, ts = [], []
    for s in range(0, len(x), chunk):
        out = proc(_msg(x[s : s + chunk], fs, T0 + s / fs))
        ax = out.axes["time"]
        t_ix = out.get_axis_idx("time")
        ys.append(np.moveaxis(out.data, t_ix, 0))
        ts.append(ax.offset + np.arange(out.data.shape[t_ix]) * ax.gain)
    return np.concatenate(ys), np.concatenate(ts)


def _impulse(n: int = 2000, at: int = 600) -> tuple[np.ndarray, float]:
    x = np.zeros(n)
    x[at] = 1.0
    return x, T0 + at / FS


# --- kernel_delay -------------------------------------------------------------------------------


def test_kernel_delay():
    lp = sps.firwin(101, 20, fs=FS)
    assert kernel_delay(lp) == 50.0
    assert kernel_delay(sps.firwin(100, 20, fs=FS)) == 49.5
    hilb = sps.remez(171, [0, 1, 2, 248, 249, 250], [0, 1, 0], type="hilbert", fs=FS)  # antisymmetric
    assert kernel_delay(hilb) == 85.0
    t = np.arange(-50, 51) / FS
    morlet = np.exp(2j * np.pi * 40 * t) * np.exp(-(t**2) / 0.001)  # conjugate-symmetric
    assert is_linear_phase(morlet) and kernel_delay(morlet) == 50.0
    # Not linear phase: group delay at the peak; leading zeros add to it.
    assert not is_linear_phase(np.r_[0.0, 0.0, 0.0, lp])
    assert kernel_delay(np.r_[0.0, 0.0, 0.0, lp]) == pytest.approx(53.0, abs=1e-6)
    assert kernel_delay(np.r_[0.0, 0.0, morlet]) == pytest.approx(52.0, abs=1e-6)
    assert kernel_delay(min_phase_kernel(lp, MinPhaseMode.HOMOMORPHICFULL)) < 50.0


# --- single filters -----------------------------------------------------------------------------


@pytest.mark.parametrize("n_taps", [101, 100])
def test_fir_impulse_peak_carries_input_timestamp(n_taps):
    x, t_imp = _impulse()
    b = sps.firwin(n_taps, 40, fs=FS)
    d = (n_taps - 1) / 2

    off = FilterTransformer(FilterSettings(coefs=(b, np.array([1.0]))))
    y_off, t_off = _run(off, x)
    assert off.delay_samples == d
    on = FilterTransformer(FilterSettings(coefs=(b, np.array([1.0])), compensate_delay=True))
    y_on, t_on = _run(on, x)

    np.testing.assert_array_equal(y_on, y_off)  # only the stamps move
    np.testing.assert_allclose(t_off - t_on, d / FS, atol=1e-12)
    assert abs(t_on[np.argmax(y_on[:, 0])] - t_imp) <= 0.5 / FS + 1e-12
    # Off: the peak's stamp is late by the reported delay (within half a sample for even taps).
    assert abs(t_off[np.argmax(y_off[:, 0])] - t_imp - d / FS) <= 0.5 / FS + 1e-12


def test_empty_message_is_stamped_like_its_neighbours():
    b = sps.firwin(101, 40, fs=FS)
    for first in (True, False):  # empty before, and after, the first real chunk
        proc = FilterTransformer(FilterSettings(coefs=(b, np.array([1.0])), compensate_delay=True))
        if not first:
            proc(_msg(np.zeros(10)))
        out = proc(_msg(np.zeros(0)))
        assert out.data.shape[0] == 0
        assert out.axes["time"].offset == pytest.approx(T0 - 50 / FS, abs=1e-12)


def test_design_filter_compensates_and_reports_delay():
    x, t_imp = _impulse()
    proc = FIRFilterTransformer(order=151, cutoff=40.0, compensate_delay=True)
    assert proc.delay_samples is None
    y, t = _run(proc, x, chunk=37)
    assert proc.delay_samples == 75.0
    assert t[np.argmax(y[:, 0])] == pytest.approx(t_imp, abs=1e-9)


def test_chained_fir_delays_add():
    x, t_imp = _impulse()
    first = FIRFilterTransformer(order=101, cutoff=40.0, compensate_delay=True)
    second = FIRFilterTransformer(order=61, cutoff=60.0, compensate_delay=True)
    out1 = first(_msg(x))
    out2 = second(out1)
    assert out2.axes["time"].offset == pytest.approx(T0 - (50 + 30) / FS, abs=1e-12)
    t = out2.axes["time"].value(np.arange(len(x)))
    assert t[np.argmax(out2.data[:, 0])] == pytest.approx(t_imp, abs=1e-9)


def test_fir_hilbert_envelope_aligned_with_modulation():
    t = np.arange(int(4 * FS)) / FS
    true_env = 1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    x = true_env * np.cos(2 * np.pi * 40.0 * t)
    settings = dict(order=170, f_lo=10.0, f_hi=100.0, trans_lo=5.0, trans_hi=5.0)

    def best_lag(env: np.ndarray, stamps: np.ndarray) -> float:
        # Lag (samples, 0.1 resolution) at which the true envelope, read at the output stamps, best fits the output.
        keep = slice(300, -300)  # skip the filter's start-up transient
        lags = np.round(np.arange(-1200, 1201) * 0.1, 1)
        errs = [np.mean((env[keep] - np.interp(stamps[keep] - T0 + lag / FS, t, true_env)) ** 2) for lag in lags]
        return float(lags[np.argmin(errs)])

    env_on, t_on = _run(FIRHilbertEnvelopeTransformer(compensate_delay=True, **settings), x, chunk=50)
    env_off, t_off = _run(FIRHilbertEnvelopeTransformer(**settings), x, chunk=50)
    assert best_lag(env_on[:, 0], t_on) == 0.0
    assert best_lag(env_off[:, 0], t_off) == -85.0  # uncompensated stamps are early by (171 - 1) / 2


def test_min_phase_filter_burst_envelope_aligned():
    """A narrow-band burst through a min-phase bandpass keeps its envelope peak's timestamp."""
    t = np.arange(int(6 * FS)) / FS
    t_c = 3.0
    x = np.exp(-0.5 * ((t - t_c) / 0.3) ** 2) * np.cos(2 * np.pi * 10.0 * t)
    proc = FIRFilterTransformer(
        order=401,
        cutoff=[7.0, 13.0],
        pass_zero=False,
        min_phase=MinPhaseMode.HOMOMORPHICFULL,
        compensate_delay=True,
    )
    y, stamps = _run(proc, x, chunk=100)
    d = proc.delay_samples
    assert 0 < d < 200  # well below the linear-phase delay of 200 samples
    env = np.abs(sps.hilbert(y[:, 0]))
    err_samples = (stamps[np.argmax(env)] - (T0 + t_c)) * FS
    assert abs(err_samples) <= 2.0, err_samples


def test_compensated_fir_is_chunk_size_invariant():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1500, 3))
    y1, t1 = _run(FIRFilterTransformer(order=101, cutoff=40.0, compensate_delay=True), x)
    y2, t2 = _run(FIRFilterTransformer(order=101, cutoff=40.0, compensate_delay=True), x, chunk=37)
    np.testing.assert_allclose(y2, y1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t2, t1, rtol=0, atol=1e-9)


def test_live_coefficient_update_changes_delay():
    x = np.zeros(200)
    proc = FilterTransformer(FilterSettings(coefs=(sps.firwin(21, 40, fs=FS), np.array([1.0])), compensate_delay=True))
    proc(_msg(x))
    assert proc.delay_samples == 10.0
    proc.update_coefficients((sps.firwin(41, 40, fs=FS), np.array([1.0])))
    out = proc(_msg(x, offset=T0 + 200 / FS))
    assert proc.delay_samples == 20.0
    assert out.axes["time"].offset == pytest.approx(T0 + (200 - 20) / FS, abs=1e-12)

    # Same-length swap (state kept) is picked up too.
    b = sps.firwin(41, 40, fs=FS)
    proc.update_coefficients((np.r_[0.0, 0.0, b[1:-1]], np.array([1.0])))
    proc(_msg(x, offset=T0 + 400 / FS))
    assert proc.delay_samples == pytest.approx(21.0, abs=1e-6)

    # Live redesign of a design-based filter.
    design = FIRFilterTransformer(order=21, cutoff=40.0, compensate_delay=True)
    design(_msg(x))
    assert design.delay_samples == 10.0
    design.update_settings(order=61)
    design(_msg(x, offset=T0 + 200 / FS))
    assert design.delay_samples == 30.0


def test_design_forwards_execution_settings_to_child():
    """use_fast_sosfilt, thread_min_bytes and fir_fft_min_taps used to be dropped."""
    x = np.random.default_rng(1).standard_normal((300, 2))
    proc = FIRFilterTransformer(
        order=101, cutoff=40.0, fir_fft_min_taps=10**6, use_fast_sosfilt=False, thread_min_bytes=0
    )
    proc(_msg(x))
    child = proc.state.filter
    assert child.state.fir_method == "conv1d"  # 101 taps would take the FFT path at the default of 64
    assert child.settings.use_fast_sosfilt is False
    assert child.settings.thread_min_bytes == 0
    assert child.settings.fir_fft_min_taps == 10**6

    # A live settings change reaches the child on the sync redesign path too.
    proc.update_settings(compensate_delay=True)
    out = proc(_msg(x, offset=T0 + 300 / FS))
    assert out.axes["time"].offset == pytest.approx(T0 + (300 - 50) / FS, abs=1e-12)


def test_fir_delay_logged(caplog):
    with caplog.at_level(logging.INFO, logger="ezmsg"):
        proc = FIRFilterTransformer(order=101, cutoff=40.0)
        for s in range(0, 600, 100):
            proc(_msg(np.zeros(100), offset=T0 + s / FS))
    msgs = [r.getMessage() for r in caplog.records if "FIR filter" in r.getMessage()]
    assert len(msgs) == 1  # once, not per message
    assert "101 taps" in msgs[0] and "delay 50 samples (100 ms)" in msgs[0] and "not compensated" in msgs[0]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ezmsg"):
        FIRFilterTransformer(order=101, cutoff=40.0, min_phase=MinPhaseMode.HOMOMORPHICFULL, compensate_delay=True)(
            _msg(np.zeros(100))
        )
    msgs = [r.getMessage() for r in caplog.records if "FIR filter" in r.getMessage()]
    assert len(msgs) == 1 and "approximate delay" in msgs[0] and ", compensated" in msgs[0]


@pytest.mark.parametrize(
    "make",
    [
        lambda: FilterTransformer(FilterSettings(coefs=sps.butter(4, 0.2), compensate_delay=True)),
        lambda: ButterworthFilterTransformer(order=4, cutoff=40.0, compensate_delay=True),
        lambda: ButterworthFilterTransformer(order=4, cutoff=40.0, coef_type="sos", compensate_delay=True),
        lambda: ChebyshevFilterTransformer(order=4, ripple_tol=0.5, Wn=40.0, compensate_delay=True),
        lambda: CombFilterTransformer(fundamental_freq=60.0, compensate_delay=True),
        lambda: ButterworthZeroPhaseTransformer(order=4, cutoff=40.0, compensate_delay=True),
        lambda: ButterworthBackwardFilterTransformer(
            ButterworthZeroPhaseSettings(order=4, cutoff=40.0, compensate_delay=True)
        ),
        lambda: FIRFilterTransformer(order=101, cutoff=40.0, coef_type="sos", compensate_delay=True),
    ],
)
def test_iir_or_sos_with_compensate_delay_raises(make):
    with pytest.raises(ValueError, match="compensate_delay"):
        make()(_msg(np.zeros(100)))


def test_update_to_iir_with_compensate_delay_raises():
    proc = FilterTransformer(FilterSettings(coefs=(sps.firwin(21, 40, fs=FS), np.array([1.0])), compensate_delay=True))
    proc(_msg(np.zeros(100)))
    with pytest.raises(ValueError, match="compensate_delay"):
        proc.update_coefficients(sps.butter(4, 0.2))


def test_compensate_delay_requires_linear_axis():
    proc = FilterTransformer(FilterSettings(coefs=(sps.firwin(21, 40, fs=FS), np.array([1.0])), compensate_delay=True))
    msg = AxisArray(
        np.zeros((50, 1)),
        dims=["time", "ch"],
        axes={"time": AxisArray.CoordinateAxis(data=np.arange(50) / FS, dims=["time"])},
    )
    with pytest.raises(TypeError, match="LinearAxis"):
        proc(msg)


def test_defaults_leave_timestamps_alone():
    x, _ = _impulse()
    out = FIRFilterTransformer(order=101, cutoff=40.0)(_msg(x))
    assert out.axes["time"].offset == T0


# --- filterbank ---------------------------------------------------------------------------------


def _lp_kernels():
    return [sps.firwin(n, c, fs=FS) for n, c in ((21, 100.0), (101, 40.0), (251, 15.0))]


@pytest.mark.parametrize("mode", [FilterbankMode.CONV, FilterbankMode.AUTO])
@pytest.mark.parametrize("chunk", [None, 37])
def test_filterbank_bands_peak_at_input_timestamp(mode, chunk):
    x, t_imp = _impulse()
    proc = FilterbankTransformer(FilterbankSettings(kernels=_lp_kernels(), mode=mode, compensate_delay=True))
    y, t = _run(proc, x, chunk=chunk)  # y: (time, ch, kernel)
    assert proc.state.mode == FilterbankMode.CONV
    assert proc.state.delay == 125.0
    for k_ix in range(3):
        assert t[np.argmax(y[:, 0, k_ix])] == pytest.approx(t_imp, abs=1e-9)


def test_filterbank_design_bands_peak_at_input_timestamp():
    x, t_imp = _impulse()
    filters = [KaiserFilterSettings(cutoff=c, ripple=60.0, width=w) for c, w in ((100.0, 40.0), (30.0, 5.0))]
    proc = FilterbankDesignTransformer(FilterbankDesignSettings(filters=filters, compensate_delay=True))
    y, t = _run(proc, x, chunk=50)
    lens = [len(k) for k in proc.state.filterbank.settings.kernels]
    assert lens[0] != lens[1]
    for k_ix in range(2):
        assert t[np.argmax(y[:, 0, k_ix])] == pytest.approx(t_imp, abs=0.5 / FS + 1e-9)


def test_filterbank_complex_kernels_aligned():
    """Complex (analytic) kernels of different lengths: band envelopes peak at the input timestamp."""
    x, t_imp = _impulse()
    kernels = []
    for f, n in ((60.0, 41), (20.0, 151)):
        tt = (np.arange(n) - (n - 1) / 2) / FS
        kernels.append(np.exp(2j * np.pi * f * tt) * np.hanning(n))
    proc = FilterbankTransformer(FilterbankSettings(kernels=kernels, compensate_delay=True))
    y, t = _run(proc, x, chunk=64)
    for k_ix in range(2):
        assert t[np.argmax(np.abs(y[:, 0, k_ix]))] == pytest.approx(t_imp, abs=1e-9)


def test_filterbank_compensated_is_chunk_size_invariant():
    x = np.random.default_rng(2).standard_normal((1200, 2))

    def make():
        return FilterbankTransformer(FilterbankSettings(kernels=_lp_kernels(), compensate_delay=True))

    y1, t1 = _run(make(), x)
    y2, t2 = _run(make(), x, chunk=29)
    np.testing.assert_allclose(y2, y1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t2, t1, rtol=0, atol=1e-9)


def test_filterbank_selective_min_phase_conversion(caplog):
    long_k = sps.firwin(401, [8.0, 12.0], fs=FS, pass_zero=False)
    short = [
        sps.firwin(31, [80.0, 120.0], fs=FS, pass_zero=False),
        sps.firwin(41, [150.0, 200.0], fs=FS, pass_zero=False),
    ]
    kernels = [long_k, *short]
    with caplog.at_level(logging.INFO, logger="ezmsg"):
        proc = FilterbankTransformer(
            FilterbankSettings(kernels=kernels, min_phase=MinPhaseMode.HOMOMORPHICFULL, compensate_delay=True)
        )
        proc(_msg(np.zeros(100)))
    prep = proc.state.prep_kerns

    # Only the bottleneck was converted...
    mp_long = min_phase_kernel(long_k, MinPhaseMode.HOMOMORPHICFULL)
    np.testing.assert_array_equal(prep[0], mp_long)
    # ...the others are untouched apart from leading zeros.
    for k, p in zip(short, prep[1:]):
        n_pad = len(p) - len(k)
        assert n_pad > 0
        np.testing.assert_array_equal(p[:n_pad], 0.0)
        np.testing.assert_array_equal(p[n_pad:], k)
    # The common delay is what converting every kernel would give.
    all_converted = max(kernel_delay(min_phase_kernel(k, MinPhaseMode.HOMOMORPHICFULL)) for k in kernels)
    assert proc.state.delay == pytest.approx(all_converted, abs=1e-12)
    assert proc.state.delay == pytest.approx(kernel_delay(mp_long), abs=1e-12)

    msg = next(r.getMessage() for r in caplog.records if "Filterbank" in r.getMessage())
    assert msg.count("min-phase") == 1 and "0: " in msg and "min-phase" in msg.split("1: ")[0]
    assert "approximate" in msg and "common approximate delay" in msg
    assert "worst residual misalignment" in msg and "compensated on the time axis" in msg


def test_filterbank_half_length_min_phase_converts_all():
    kernels = _lp_kernels()
    proc = FilterbankTransformer(
        FilterbankSettings(kernels=kernels, min_phase=MinPhaseMode.HOMOMORPHIC, compensate_delay=True)
    )
    proc(_msg(np.zeros(100)))
    for k, p in zip(kernels, proc.state.prep_kerns):
        mp = min_phase_kernel(k, MinPhaseMode.HOMOMORPHIC)
        np.testing.assert_array_equal(p[len(p) - len(mp) :], mp)


def test_filterbank_log_when_not_compensated(caplog):
    with caplog.at_level(logging.INFO, logger="ezmsg"):
        FilterbankTransformer(FilterbankSettings(kernels=_lp_kernels()))(_msg(np.zeros(100)))
    msg = next(r.getMessage() for r in caplog.records if "Filterbank" in r.getMessage())
    assert "0: 10 (20 ms)" in msg and "2: 125 (250 ms)" in msg and "Not compensated" in msg
    assert "approximate" not in msg


@pytest.mark.parametrize("mode", [FilterbankMode.FFT, FilterbankMode.AUTO])
def test_filterbank_fft_mode_with_compensate_delay_raises(mode):
    # A long kernel makes AUTO choose FFT.
    kernels = [sps.firwin(1501, 10.0, fs=FS), sps.firwin(1001, 20.0, fs=FS)]
    proc = FilterbankTransformer(FilterbankSettings(kernels=kernels, mode=mode, compensate_delay=True))
    with pytest.raises(ValueError, match="FFT mode"):
        proc(_msg(np.zeros(100)))
