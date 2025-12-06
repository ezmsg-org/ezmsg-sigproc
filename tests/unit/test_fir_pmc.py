import numpy as np
import scipy.signal as sps

from ezmsg.sigproc.fir_pmc import parks_mcclellan_design_fun


def test_pmcfir_behavior():
    fs = 1000.0
    dur = 2.0
    t = np.arange(int(fs * dur)) / fs

    # 20 Hz + 100 Hz, equal amplitude
    x = np.sin(2 * np.pi * 20.0 * t) + np.sin(2 * np.pi * 100.0 * t)

    coefs = parks_mcclellan_design_fun(
        fs=fs,
        order=201,
        cuton=70.0,
        cutoff=150.0,
        transition=10.0,
        weight_pass=1.0,
        weight_stop_lo=2.0,
        weight_stop_hi=2.0,
    )
    assert coefs is not None
    b, a = coefs
    assert a.size == 1 and a[0] == 1.0

    y = np.convolve(x, b, mode="same")

    proj_20 = np.mean(y * np.sin(2 * np.pi * 20.0 * t)) * 2
    proj_100 = np.mean(y * np.sin(2 * np.pi * 100.0 * t)) * 2

    # Attenuation of 20 Hz, preservation of 100 Hz
    assert abs(proj_100) > 0.3
    assert abs(proj_20) < 0.1


def test_pmcfir_symmetry_and_delay():
    fs = 1000.0
    order = 201
    b, a = parks_mcclellan_design_fun(
        fs=fs,
        order=order,
        cuton=70.0,
        cutoff=150.0,
        transition=10.0,
        weight_pass=1.0,
        weight_stop_lo=1.0,
        weight_stop_hi=1.0,
    )
    assert a.size == 1 and a[0] == 1.0
    assert np.allclose(b, b[::-1], atol=1e-12)

    w, gd = sps.group_delay((b, a), fs=fs)
    k = np.argmin(np.abs(w - 110.0))
    assert np.isclose(gd[k], order / 2, rtol=0.1)


def test_pmcfir_design_quality():
    fs = 1000.0
    order = 201
    b, a = parks_mcclellan_design_fun(
        fs=fs,
        order=order,
        cuton=70.0,
        cutoff=150.0,
        transition=10.0,
        weight_pass=1.0,
        weight_stop_lo=2.0,
        weight_stop_hi=2.0,
    )
    # Frequency response
    N = 32768
    H = np.fft.rfft(np.r_[b, np.zeros(N - len(b))])
    f = np.fft.rfftfreq(N, 1 / fs)

    # Regions
    pass_mask = (f >= 75.0) & (f <= 145.0)
    stop_lo = f <= 60.0
    stop_hi = f >= 160.0

    mag = np.abs(H)
    pb = mag[pass_mask]
    pb_db = 20 * np.log10(pb / np.median(pb))
    assert np.ptp(pb_db) < 1.0  # Ripple within 1 dB

    # Stopband attenuation vs median passband gain
    ref = np.median(pb)
    att_lo_db = 20 * np.log10(np.maximum(mag[stop_lo], 1e-12) / ref)
    att_hi_db = 20 * np.log10(np.maximum(mag[stop_hi], 1e-12) / ref)
    assert np.median(att_lo_db) < -20
    assert np.median(att_hi_db) < -20
