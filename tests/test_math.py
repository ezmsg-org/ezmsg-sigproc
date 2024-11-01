import numpy as np
import pytest

from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.math.abs import abs
from ezmsg.sigproc.math.clip import clip
from ezmsg.sigproc.math.difference import const_difference
from ezmsg.sigproc.math.invert import invert
from ezmsg.sigproc.math.log import log
from ezmsg.sigproc.math.scale import scale


def test_abs():
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    proc = abs()
    msg_out = proc.send(msg_in)
    assert np.array_equal(msg_out.data, np.abs(in_dat))


@pytest.mark.parametrize("a_min", [1, 2])
@pytest.mark.parametrize("a_max", [133, 134])
def test_clip(a_min: float, a_max: float):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    proc = clip(a_min, a_max)
    msg_out = proc.send(msg_in)

    assert all(msg_out.data[np.where(in_dat < a_min)] == a_min)
    assert all(msg_out.data[np.where(in_dat > a_max)] == a_max)


@pytest.mark.parametrize("value", [-100, 0, 100])
@pytest.mark.parametrize("subtrahend", [False, True])
def test_const_difference(value: float, subtrahend: bool):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    proc = const_difference(value, subtrahend)
    msg_out = proc.send(msg_in)
    assert np.array_equal(
        msg_out.data, (in_dat - value) if subtrahend else (value - in_dat)
    )


def test_invert():
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    proc = invert()
    msg_out = proc.send(msg_in)
    assert np.array_equal(msg_out.data, 1 / in_dat)


@pytest.mark.parametrize("base", [np.e, 2, 10])
@pytest.mark.parametrize("dtype", [int, float])
@pytest.mark.parametrize("clip", [False, True])
def test_log(base: float, dtype, clip: bool):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans).astype(dtype)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    proc = log(base, clip_zero=clip)
    msg_out = proc.send(msg_in)
    if clip and dtype is float:
        in_dat = np.clip(in_dat, a_min=np.finfo(msg_in.data.dtype).tiny, a_max=None)
    assert np.array_equal(msg_out.data, np.log(in_dat) / np.log(base))


@pytest.mark.parametrize("scale_factor", [0.1, 0.5, 2.0, 10.0])
def test_scale(scale_factor: float):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    proc = scale(scale_factor)
    msg_out = proc.send(msg_in)

    assert msg_out.data.shape == (n_times, n_chans)
    assert np.array_equal(msg_out.data, in_dat * scale_factor)
