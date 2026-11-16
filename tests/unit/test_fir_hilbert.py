import numpy as np
import pytest
import scipy.signal as sps
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ezmsg.sigproc.fir_hilbert import (
    FIRHilbertEnvelopeTransformer,
    FIRHilbertFilterTransformer,
)
from tests.helpers.empty_time import check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg


def axisarray(x: np.ndarray, fs: float, t0: float = 0.0) -> AxisArray:
    if x.ndim == 1:
        x = x[:, None]
    return AxisArray(
        data=x,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=t0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(x.shape[1]), dims=["ch"]),
        },
    )


def _ncorr(x, y, lag):
    if lag > 0:
        x2, y2 = x[lag:], y[:-lag]
    elif lag < 0:
        x2, y2 = x[:lag], y[-lag:]
    else:
        x2, y2 = x, y
    if x2.size < 100:
        return -1.0
    x2 = x2 - x2.mean()
    y2 = y2 - y2.mean()
    denom = np.linalg.norm(x2) * np.linalg.norm(y2)
    return float(np.dot(x2, y2) / denom) if denom > 0 else -1.0


def _best_corr_with_small_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 20) -> tuple[float, int]:
    assert a.size == b.size

    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.array([_ncorr(a, b, L) for L in lags])
    i = int(np.argmax(corrs))
    return float(corrs[i]), int(lags[i])


# FIR Hilbert tests


def test_fir_hilbert_filter_design_properties():
    fs = 1000.0
    tr = FIRHilbertFilterTransformer(
        axis="time",
        order=170,
        f_lo=1.0,
        f_hi=None,
        trans_lo=1.0,
        trans_hi=1.0,
    )

    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    msg = axisarray(x, fs)
    _ = tr(msg)

    coefs = tr.state.filter.settings.coefs
    taps = tr.get_taps()
    b, a = coefs

    # odd taps, antisymmetric impulse response, a == [1]
    assert taps % 2 == 1
    assert np.allclose(a, np.array([1.0]))
    assert np.allclose(b, -b[::-1], atol=1e-10)
    # near-zero DC gain
    w, h = sps.freqz(b, a, worN=2048, fs=fs)
    dc_mag = np.abs(h[w == 0.0])[0] if np.any(w == 0.0) else np.abs(h[0])
    assert dc_mag < 1e-3


def test_fir_hilbert_matches_scipy_oracle_with_delay():
    fs = 1000.0
    t = np.arange(12000) / fs
    x = (1 + 0.25 * np.sin(2 * np.pi * 2 * t)) * np.sin(2 * np.pi * 90 * t)

    Nbp = 391
    bp = sps.firwin(Nbp, [82, 98], pass_zero=False, fs=fs)
    xb = sps.filtfilt(bp, [1.0], x)
    trim_bp = min(3 * (Nbp - 1), 2 * Nbp)
    xb = xb[trim_bp:-trim_bp]
    assert xb.size > 4000

    tr = FIRHilbertFilterTransformer(axis="time", order=500, f_lo=82, f_hi=98, trans_lo=2, trans_hi=2)
    y_class = tr(axisarray(xb, fs)).data.squeeze()

    taps = tr.get_taps()
    assert taps is not None and taps % 2 == 1
    dly = (taps - 1) // 2

    imag_oracle = np.imag(sps.hilbert(xb))
    y_c = y_class[dly:]
    o_c = imag_oracle[:-dly]

    guard = max(Nbp, dly)
    L = min(y_c.size, o_c.size) - 2 * guard
    assert L > 2000

    a = (y_c[guard : guard + L] - y_c[guard : guard + L].mean()).astype(float)
    b = (o_c[guard : guard + L] - o_c[guard : guard + L].mean()).astype(float)

    MAX_TIGHT = min(50, dly)
    co1, lag1 = _best_corr_with_small_lag(a, b, max_lag=MAX_TIGHT)
    co2, lag2 = _best_corr_with_small_lag(a, -b, max_lag=MAX_TIGHT)
    co, lag = (co2, lag2) if co2 > co1 else (co1, lag1)

    assert co > 0.99 and abs(lag) <= 2


# FIR Hilbert Envelope tests


def test_fir_hilbert_envelope_tone_constant():
    fs = 1000.0
    f0 = 80.0
    t = np.arange(int(5.0 * fs)) / fs
    A = 1.0
    x = A * np.sin(2 * np.pi * f0 * t)

    bw = 16.0
    Nbp = 391
    bp = sps.firwin(Nbp, [f0 - bw / 2, f0 + bw / 2], pass_zero=False, fs=fs)
    x_bp = sps.filtfilt(bp, [1.0], x)

    tr = FIRHilbertEnvelopeTransformer(
        axis="time",
        order=300,
        f_lo=60.0,
        f_hi=200.0,
        trans_lo=2.0,
        trans_hi=2.0,
        norm_freq=f0,
    )
    y = tr(axisarray(x_bp, fs)).data[:, 0]

    taps = tr._state.filter.get_taps()
    dly = taps // 2
    guard = max(dly, Nbp)
    yv = y[guard:-guard] if y.size > 2 * guard else y[dly:]
    assert yv.size > 500

    env_med = float(np.median(yv))
    p05, p95 = np.percentile(yv, [5, 95])
    ripple_robust = (p95 - p05) / max(env_med, 1e-12)

    assert np.isclose(env_med, A, rtol=0.05)
    assert ripple_robust < 0.05


def test_fir_hilbert_envelope_internal_consistency_oracle():
    fs = 1000.0
    t = np.arange(int(6.0 * fs)) / fs
    f_c, f_m, m = 90.0, 3.0, 0.5

    env_true = 1.0 + m * np.sin(2 * np.pi * f_m * t)
    x = env_true * np.sin(2 * np.pi * f_c * t)

    bw = 16.0
    Nbp = 391
    bp = sps.firwin(Nbp, [f_c - bw / 2, f_c + bw / 2], pass_zero=False, fs=fs)
    x_bp = sps.filtfilt(bp, [1.0], x)
    trim_bp = min(3 * (Nbp - 1), 2 * Nbp)
    x_bp = x_bp[trim_bp:-trim_bp]

    tr = FIRHilbertEnvelopeTransformer(
        axis="time",
        order=500,
        f_lo=f_c - bw / 2,
        f_hi=f_c + bw / 2,
        trans_lo=2.0,
        trans_hi=2.0,
        norm_band=(f_c - 1.0, f_c + 1.0),
    )
    env_node = np.asarray(tr(axisarray(x_bp[:, None], fs)).data).squeeze()

    taps = tr._state.filter.get_taps()
    dly = (taps - 1) // 2
    assert env_node.size == x_bp.size

    bH, aH = tr._state.filter.state.filter.settings.coefs
    y_imag_full = sps.lfilter(bH, aH, x_bp)

    x_real_full = np.concatenate([np.zeros(dly, dtype=x_bp.dtype), x_bp[:-dly]])

    L = min(env_node.size, x_real_full.size, y_imag_full.size)
    env_oracle_full = np.abs(x_real_full[:L].astype(np.complex64) + 1j * y_imag_full[:L].astype(np.complex64))
    env_cmp_full = env_node[:L]

    guard = max(dly, Nbp)
    if 2 * guard < L:
        env_oracle = env_oracle_full[guard:-guard]
        env_cmp = env_cmp_full[guard:-guard]
    else:
        env_oracle = env_oracle_full
        env_cmp = env_cmp_full

    co, lag = _best_corr_with_small_lag(env_cmp.astype(float), env_oracle.astype(float), max_lag=5)
    assert co > 0.995 and abs(lag) <= 1


def test_fir_hilbert_envelope_multichannel_shapes():
    fs = 1000.0
    t = np.arange(4000) / fs
    f0 = 70.0
    x1 = np.sin(2 * np.pi * f0 * t)
    x2 = 0.5 * np.sin(2 * np.pi * (f0 + 10.0) * t)
    X = np.vstack([x1, x2]).T
    msg = axisarray(X, fs)

    tr = FIRHilbertEnvelopeTransformer(
        axis="time",
        order=170,
    )
    y = tr(msg).data
    assert y.shape[0] == X.shape[0]
    assert y.shape[1] == X.shape[1]
    assert np.min(y) >= 0


def test_fir_hilbert_envelope_constant_tones_gain():
    fs = 1000.0
    t = np.arange(4000) / fs
    x1 = 1.0 * np.sin(2 * np.pi * 70 * t)
    x2 = 0.5 * np.sin(2 * np.pi * 90 * t)
    X = np.vstack([x1, x2]).T

    tr = FIRHilbertEnvelopeTransformer(
        axis="time",
        order=500,
        f_lo=60.0,
        f_hi=100.0,
        trans_lo=2.0,
        trans_hi=2.0,
        norm_band=(75.0, 85.0),
    )
    y = tr(axisarray(X, fs)).data

    taps = tr._state.filter.get_taps()
    dly = (taps - 1) // 2
    guard = dly
    if y.shape[0] > 2 * guard:
        y_mid = y[guard:-guard, :]
    else:
        y_mid = y[guard:, :]

    m0, m1 = float(np.mean(y_mid[:, 0])), float(np.mean(y_mid[:, 1]))
    s0, s1 = float(np.std(y_mid[:, 0])), float(np.std(y_mid[:, 1]))

    assert abs(m0 - 1.0) < 0.03
    assert abs(m1 - 0.5) < 0.03

    assert s0 < 0.02 and s1 < 0.02

    assert np.min(y) >= -1e-6


def test_hilbert_filter_empty_after_init():
    from ezmsg.sigproc.fir_hilbert import FIRHilbertFilterSettings, FIRHilbertFilterTransformer

    proc = FIRHilbertFilterTransformer(FIRHilbertFilterSettings(order=20, f_lo=5.0, f_hi=15.0, axis="time"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_hilbert_envelope_empty_after_init():
    from ezmsg.sigproc.fir_hilbert import FIRHilbertEnvelopeTransformer, FIRHilbertFilterSettings

    proc = FIRHilbertEnvelopeTransformer(FIRHilbertFilterSettings(order=20, f_lo=5.0, f_hi=15.0, axis="time"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_hilbert_filter_empty_first():
    from ezmsg.sigproc.fir_hilbert import FIRHilbertFilterSettings, FIRHilbertFilterTransformer

    proc = FIRHilbertFilterTransformer(FIRHilbertFilterSettings(order=20, f_lo=5.0, f_hi=15.0, axis="time"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_hilbert_envelope_empty_first():
    from ezmsg.sigproc.fir_hilbert import FIRHilbertEnvelopeTransformer, FIRHilbertFilterSettings

    proc = FIRHilbertEnvelopeTransformer(FIRHilbertFilterSettings(order=20, f_lo=5.0, f_hi=15.0, axis="time"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


# --- flush: the samples the filter delay holds back ------------------------------------------

FLUSH_FS = 500.0


def _env(**kw):
    from ezmsg.sigproc.fir_hilbert import FIRHilbertFilterSettings

    return FIRHilbertEnvelopeTransformer(FIRHilbertFilterSettings(order=100, f_lo=5.0, f_hi=100.0, **kw))


def _env_msg(x: np.ndarray, start: int, segment: int = 3) -> AxisArray:
    return replace(axisarray(x, FLUSH_FS, t0=start / FLUSH_FS), segment=segment)


def _env_feed(proc, x: np.ndarray, chunk: int) -> list[AxisArray]:
    return [proc(_env_msg(x[s : s + chunk], s)) for s in range(0, len(x), chunk)]


def _env_cat(outs: list[AxisArray]) -> tuple[np.ndarray, np.ndarray]:
    outs = [o for o in outs if o.data.shape[0]]
    t = np.concatenate([o.axes["time"].offset + np.arange(o.data.shape[0]) / FLUSH_FS for o in outs])
    return np.concatenate([o.data for o in outs]), t


def _flush_signal(n: int = 1000) -> np.ndarray:
    t = np.arange(n) / FLUSH_FS
    tone = np.sin(2 * np.pi * 20 * t) * (1 + 0.5 * np.sin(2 * np.pi * 1.3 * t))
    return np.stack([tone, 3.0 + tone], axis=1)  # the second channel ends on a DC offset


def _offline_envelope(x: np.ndarray, b: np.ndarray, end: np.ndarray) -> np.ndarray:
    """CONSTANT start-up (inner FIR from the held first sample, real part from zeros), ``end`` past the end."""
    d = len(end)
    padded = np.concatenate([np.repeat(x[:1], 2 * d, axis=0), x, end])
    imag = np.stack([np.convolve(padded[:, c], b, mode="valid") for c in range(x.shape[1])], axis=1)
    real = np.concatenate([np.zeros((d, x.shape[1])), x])
    return np.abs(real + 1j * imag)


@pytest.mark.parametrize("compensate_delay", [False, True])
def test_hilbert_envelope_flush_emits_delayed_tail(compensate_delay):
    """Streamed output plus the flushed tail is the whole envelope: every input sample's content
    comes out once, on a contiguous time base, matching an offline convolution in which the input
    continues past the end as zeros, as the first sample is held before the start."""
    x = _flush_signal()
    proc = _env(compensate_delay=compensate_delay)
    assert proc.flush() is None  # nothing held before the first message
    outs = _env_feed(proc, x, 73)
    tail = proc.flush()
    d = proc.state.dly
    assert d == 50 and tail.data.shape == (d, 2) and tail.segment == 3
    y, t = _env_cat(outs + [tail])
    # One output per input sample, delayed by d; the tail carries the last d samples' content.
    t0 = -d / FLUSH_FS if compensate_delay else 0.0
    np.testing.assert_allclose(t, t0 + np.arange(len(x) + d) / FLUSH_FS, rtol=0, atol=1e-9)

    b = proc.state.filter.state.filter.settings.coefs[0]
    y_ref = _offline_envelope(x, b, np.zeros((d, 2)))
    np.testing.assert_allclose(y, y_ref, rtol=0, atol=1e-5)  # float32 output
    assert proc.flush() is None  # nothing left


def test_hilbert_envelope_flush_shorter_than_delay():
    """A stream shorter than the delay still emits all d delayed samples on a contiguous time base."""
    x = _flush_signal(1020)[1000:]  # 20 samples, not starting at time zero
    proc = _env()
    outs = [proc(_env_msg(x[s : s + 10], 1000 + s)) for s in (0, 10)]
    tail = proc.flush()
    d = proc.state.dly
    assert d == 50 and tail.data.shape == (d, 2)
    assert tail.axes["time"].offset == pytest.approx(1020 / FLUSH_FS)
    y, t = _env_cat(outs + [tail])
    np.testing.assert_allclose(t, (1000 + np.arange(20 + d)) / FLUSH_FS, rtol=0, atol=1e-9)
    b = proc.state.filter.state.filter.settings.coefs[0]
    np.testing.assert_allclose(y, _offline_envelope(x, b, np.zeros((d, 2))), rtol=0, atol=1e-5)
    assert proc.flush() is None


def test_hilbert_envelope_flush_tail_accuracy():
    """On zero-mean band-limited tones that continue past the flush, the zero-padded tail's envelope
    is closer to the true one (median RMS error) than hold-last's or even reflection's. (With a DC
    offset zero padding is the worst of the three: the caveat in ``flush``.)"""
    rng = np.random.default_rng(0)
    n, d = 300, 50
    t = np.arange(n + d) / FLUSH_FS
    errs = {"zeros": [], "hold": [], "even": []}
    for _ in range(36):
        x = np.sin(2 * np.pi * rng.uniform(6.0, 100.0) * t + rng.uniform(0, 2 * np.pi))[:, None]
        proc = _env()
        proc(_env_msg(x[:n], 0))
        b = proc.state.filter.state.filter.settings.coefs[0]
        true = _offline_envelope(x[:n], b, x[n:])[-d:]
        tails = {
            "zeros": proc.flush().data,
            "hold": _offline_envelope(x[:n], b, np.repeat(x[n - 1 : n], d, axis=0))[-d:],
            "even": _offline_envelope(x[:n], b, x[n - 2 - np.arange(d)])[-d:],  # about the last sample
        }
        for k, env in tails.items():
            errs[k].append(np.sqrt(np.mean((env - true) ** 2)))
    med = {k: np.median(v) for k, v in errs.items()}
    assert med["zeros"] < med["hold"]  # measured 0.091 vs 0.163
    assert med["zeros"] < med["even"]  # measured 0.091 vs 0.130 (seeds 0-9: 0.082-0.096 vs 0.102-0.150)


def test_hilbert_envelope_flush_with_final_message():
    x = _flush_signal(400)
    proc = _env()
    _env_feed(proc, x[:350], 50)
    tail = proc.flush(_env_msg(x[350:], 350))
    ref = _env()
    y_ref, t_ref = _env_cat(_env_feed(ref, x, 50)[-1:] + [ref.flush()])
    y, t = _env_cat([tail])
    np.testing.assert_array_equal(y, y_ref)
    np.testing.assert_allclose(t, t_ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("segment", [3, 4], ids=["same-segment", "new-segment"])
def test_hilbert_envelope_restarts_after_flush(segment):
    """The padding the tail was pushed out with is not real input, so whatever comes next starts
    the filter afresh: no tail sample is emitted again."""
    x = _flush_signal(400)
    proc = _env()
    _env_feed(proc, x[:300], 50)
    proc.flush()
    nxt = _env_msg(x[300:], 300, segment=segment)
    np.testing.assert_array_equal(proc(nxt).data, _env()(nxt).data)


def test_hilbert_envelope_flush_releases_unfilled_buffer():
    """BUFFERED start-up that never filled: the held samples are primed from the reflection of all
    of them, which is what FIRST_CHUNK does with them as one message, and the delayed tail follows."""
    from ezmsg.sigproc.util.startup import StartupMode

    x = _flush_signal(60)  # fewer than the 100-sample pad
    proc = _env(startup=StartupMode.BUFFERED)
    assert proc.flush() is None
    outs = _env_feed(proc, x, 20)
    assert all(o.data.shape == (0, 2) for o in outs)
    y, t = _env_cat([proc.flush()])
    ref = _env(startup=StartupMode.FIRST_CHUNK)
    y_ref, t_ref = _env_cat([ref(_env_msg(x, 0)), ref.flush()])
    assert y.shape == (60 + 50, 2)
    np.testing.assert_allclose(y, y_ref, rtol=0, atol=1e-6)
    np.testing.assert_allclose(t, np.arange(110) / FLUSH_FS, rtol=0, atol=1e-9)
    assert proc.flush() is None
