import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.butterworthzerophase import (
    ButterworthZeroPhaseSettings,
    ButterworthZeroPhaseTransformer,
)


@pytest.mark.parametrize(
    "cutoff, cuton",
    [
        (30.0, None),  # lowpass
        (None, 30.0),  # highpass
        (45.0, 30.0),  # bandpass
        (30.0, 45.0),  # bandstop
    ],
)
@pytest.mark.parametrize("order", [2, 4, 8])
def test_butterworth_zp_filter_specs(cutoff, cuton, order):
    """Zero-phase settings inherit filter_specs logic from legacy Butterworth settings."""
    btype, Wn = ButterworthZeroPhaseSettings(order=order, cuton=cuton, cutoff=cutoff).filter_specs()
    if cuton is None:
        assert btype == "lowpass" and Wn == cutoff
    elif cutoff is None:
        assert btype == "highpass" and Wn == cuton
    elif cuton <= cutoff:
        assert btype == "bandpass" and Wn == (cuton, cutoff)
    else:
        assert btype == "bandstop" and Wn == (cutoff, cuton)


@pytest.mark.parametrize(
    "cutoff, cuton",
    [
        (30.0, None),  # lowpass
        (None, 30.0),  # highpass
        (45.0, 30.0),  # bandpass
        (30.0, 45.0),  # bandstop
    ],
)
@pytest.mark.parametrize("order", [0, 2, 4])
@pytest.mark.parametrize("fs", [200.0])
@pytest.mark.parametrize("n_chans", [3])
@pytest.mark.parametrize("n_dims, time_ax", [(1, 0), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("coef_type", ["ba", "sos"])
@pytest.mark.parametrize("padtype,padlen", [(None, 0), ("odd", None)])
def test_butterworth_zero_phase_matches_scipy(
    cutoff, cuton, order, fs, n_chans, n_dims, time_ax, coef_type, padtype, padlen
):
    dur = 2.0
    n_times = int(dur * fs)

    if n_dims == 1:
        dat_shape = [n_times]
        dims = ["time"]
        other_axes = {}
    else:
        dat_shape = [5, n_chans]
        dat_shape.insert(time_ax, n_times)
        dims = ["freq", "ch"]
        dims.insert(time_ax, "time")
        other_axes = {
            "freq": AxisArray.LinearAxis(unit="Hz", offset=0.0, gain=1.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(n_chans).astype(str), dims=["ch"]),
        }

    x = np.linspace(0, 1, np.prod(dat_shape), dtype=float).reshape(*dat_shape)

    msg = AxisArray(
        data=x,
        dims=dims,
        axes=frozendict({**other_axes, "time": AxisArray.TimeAxis(fs=fs, offset=0.0)}),
        key="test_butterworth_zero_phase",
    )

    # expected via SciPy
    btype, Wn = ButterworthZeroPhaseSettings(order=order, cuton=cuton, cutoff=cutoff).filter_specs()
    if order == 0:
        expected = x
    else:
        tmp = np.moveaxis(x, time_ax, -1)
        if coef_type == "ba":
            b, a = scipy.signal.butter(order, Wn, btype=btype, fs=fs, output="ba")
            y = scipy.signal.filtfilt(b, a, tmp, axis=-1, padtype=padtype, padlen=padlen)
        else:
            sos = scipy.signal.butter(order, Wn, btype=btype, fs=fs, output="sos")
            y = scipy.signal.sosfiltfilt(sos, tmp, axis=-1, padtype=padtype, padlen=padlen)
        expected = np.moveaxis(y, -1, time_ax)

    axis_name = "time" if time_ax != 0 else None
    zp = ButterworthZeroPhaseTransformer(
        axis=axis_name,
        order=order,
        cuton=cuton,
        cutoff=cutoff,
        coef_type=coef_type,
        wn_hz=True,
        padtype=padtype,
        padlen=padlen,
    )

    out = zp(msg).data
    assert np.allclose(out, expected, atol=1e-10, rtol=1e-7)


def test_butterworth_zero_phase_empty_msg():
    zp = ButterworthZeroPhaseTransformer(axis="time", order=4, cuton=0.1, cutoff=10.0, coef_type="sos")
    msg = AxisArray(
        data=np.zeros((0, 2)),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
            "ch": AxisArray.CoordinateAxis(data=np.array(["0", "1"]), dims=["ch"]),
        },
        key="test_butterworth_zero_phase_empty",
    )
    res = zp(msg)
    assert res.data.size == 0


def test_butterworth_zero_phase_update_settings_changes_output():
    fs = 200.0
    t = np.arange(int(2.0 * fs)) / fs
    x = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 40 * t)]).T

    msg = AxisArray(
        data=x,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=0.0),
            "ch": AxisArray.CoordinateAxis(data=np.array(["0", "1"]), dims=["ch"]),
        },
        key="test_butterworth_zero_phase_update",
    )

    zp = ButterworthZeroPhaseTransformer(axis="time", order=4, cutoff=30.0, coef_type="sos", padtype="odd", padlen=None)
    y1 = zp(msg).data
    # LP at 30 should pass 10 Hz and attenuate 40 Hz
    p_in = np.abs(np.fft.rfft(x, axis=0)) ** 2
    p1 = np.abs(np.fft.rfft(y1, axis=0)) ** 2
    f = np.fft.rfftfreq(x.shape[0], 1 / fs)

    def peak_power(power, f0):
        return power[np.argmin(np.abs(f - f0))]

    assert peak_power(p1[:, 0], 10.0) > 0.7 * peak_power(p_in[:, 0], 10.0)
    assert peak_power(p1[:, 1], 40.0) < 0.3 * peak_power(p_in[:, 1], 40.0)

    # Switch to HP at 25 Hz
    zp.update_settings(cutoff=None, cuton=25.0)
    y2 = zp(msg).data
    p2 = np.abs(np.fft.rfft(y2, axis=0)) ** 2
    assert peak_power(p2[:, 0], 10.0) < 0.3 * peak_power(p_in[:, 0], 10.0)
    assert peak_power(p2[:, 1], 40.0) > 0.7 * peak_power(p_in[:, 1], 40.0)

    zp.update_settings(coef_type="ba", order=2, cutoff=15.0, cuton=None)
    y3 = zp(msg).data
    # attenuate 40 more than 10
    p3 = np.abs(np.fft.rfft(y3, axis=0)) ** 2
    assert peak_power(p3[:, 1], 40.0) < peak_power(p3[:, 0], 10.0)
