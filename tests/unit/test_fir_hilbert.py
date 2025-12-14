import numpy as np
import scipy.signal as sps
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.fir_hilbert import (
    FIRHilbertEnvelopeTransformer,
    FIRHilbertFilterTransformer,
)


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
