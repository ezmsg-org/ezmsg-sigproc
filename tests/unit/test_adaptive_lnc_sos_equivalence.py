"""Locks the LMS <-> LTI-notch equivalence underpinning the SOS/MLX rewrite.

A fixed-frequency quadrature-LMS line canceller is exactly a 2nd-order notch
(Glover 1977 / Widrow). :func:`design_lnc_sos` encodes that mapping; this test
proves it reproduces the per-sample LMS so the recursion can be replaced by
``sosfilt`` (and thus an Array-API / MLX backend) without changing behaviour.

* Single harmonic: **exact** (to float64 round-off).
* Multi-harmonic: the SOS cascade matches the parallel LMS dB reduction to a
  fraction of a dB (the cascade is sequential, the LMS parallel; they differ
  only by tiny cross-harmonic coupling).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps

from ezmsg.sigproc.adaptive_lnc import design_lnc_sos

FS = 30000.0
LINE_FREQ = 60.0


def _lms_output(x: np.ndarray, tau: float, num_harmonics: int = 1) -> np.ndarray:
    """Independent per-sample quadrature-LMS canceller (fixed frequency,
    control=1), one channel -- the ground truth the SOS form must reproduce.

    This is the original recursion, kept here (not via the transformer, which
    now uses the SOS form) so the test still validates the equivalence.
    """
    mu = 2.0 / (tau * FS)
    omega = 2.0 * np.pi * LINE_FREQ / FS
    k = np.arange(1, num_harmonics + 1)
    w_sin = np.zeros(num_harmonics)
    w_cos = np.zeros(num_harmonics)
    y = np.empty_like(x)
    for n in range(x.shape[0]):
        ph = k * (omega * n)  # continuous NCO phase, harmonic k at k*phase
        rs, rc = np.sin(ph), np.cos(ph)
        nr = np.sum(w_sin * rs + w_cos * rc)
        err = nr - x[n]
        w_sin -= mu * rs * err
        w_cos -= mu * rc * err
        y[n] = x[n] - nr
    return y


def _sos_output(x: np.ndarray, tau: float, num_harmonics: int = 1) -> np.ndarray:
    omega = 2.0 * np.pi * LINE_FREQ / FS
    mu = 2.0 / (tau * FS)
    sos = design_lnc_sos(omega, mu, num_harmonics=num_harmonics)
    return sps.sosfilt(sos, x)


def _line_db(x: np.ndarray, y: np.ndarray, hz: float) -> float:
    def mag(s):
        h = len(s) // 2
        seg = s[h:] * np.hanning(len(s) - h)
        f = np.fft.rfftfreq(len(seg), 1 / FS)
        return np.abs(np.fft.rfft(seg))[np.argmin(np.abs(f - hz))]

    return 20 * np.log10(mag(y) / mag(x))


@pytest.mark.parametrize("tau", [0.02, 0.05, 0.1])
def test_single_harmonic_sos_matches_lms_exactly(tau):
    """The notch biquad reproduces the per-sample LMS to float64 round-off."""
    rng = np.random.default_rng(0)
    k = np.arange(30000)
    x = (
        200 * np.sin(2 * np.pi * LINE_FREQ / FS * k) + 40 * np.sin(2 * np.pi * 10 / FS * k) + rng.normal(0, 8, k.size)
    ).astype(np.float64)

    y_lms = _lms_output(x, tau)
    y_sos = _sos_output(x, tau)
    rel = np.max(np.abs(y_lms - y_sos)) / np.max(np.abs(x))
    assert rel < 1e-9, f"LMS vs SOS rel diff {rel:.2e} at tau={tau}"


def test_pole_radius_matches_mu():
    """Notch poles sit at radius sqrt(1 - mu), inside the unit circle."""
    omega = 2.0 * np.pi * LINE_FREQ / FS
    mu = 2.0 / (0.1 * FS)
    sos = design_lnc_sos(omega, mu, num_harmonics=1)
    # a = [1, -(2-mu)cos w, (1-mu)] -> pole product = 1-mu = r^2
    a2 = sos[0, 5]
    assert a2 == pytest.approx(1 - mu)
    assert np.sqrt(a2) < 1.0  # stable
    # zeros on the unit circle (perfect notch): b = [1, -2cos w, 1]
    assert sos[0, 2] == pytest.approx(1.0)


def test_multi_harmonic_cascade_matches_lms_db():
    """Cascade SOS and parallel LMS cancel each harmonic to within ~0.5 dB."""
    rng = np.random.default_rng(1)
    k = np.arange(30000)
    x = (
        200 * np.sin(2 * np.pi * 60 / FS * k)
        + 100 * np.sin(2 * np.pi * 180 / FS * k + 0.5)
        + 40 * np.sin(2 * np.pi * 10 / FS * k)
        + rng.normal(0, 8, k.size)
    ).astype(np.float64)

    y_lms = _lms_output(x, 0.05, num_harmonics=3)
    y_sos = _sos_output(x, 0.05, num_harmonics=3)
    for hz in (60.0, 180.0):
        assert _line_db(x, y_lms, hz) == pytest.approx(_line_db(x, y_sos, hz), abs=0.5)
