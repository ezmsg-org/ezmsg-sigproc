import numpy as np
import pytest

from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.math.clip import clip
from ezmsg.sigproc.math.difference import const_difference
from ezmsg.sigproc.math.invert import invert
from ezmsg.sigproc.math.log import log
from ezmsg.sigproc.math.scale import scale


@pytest.mark.parametrize("a_min", [1, 2])
@pytest.mark.parametrize("a_max", [133, 134])
def test_clip(a_min: float, a_max: float):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    axis_arr_in = AxisArray(in_dat, dims=["time", "ch"])

    proc = clip(a_min, a_max)
    axis_arr_out = proc.send(axis_arr_in)

    assert all(axis_arr_out.data[np.where(in_dat < a_min)] == a_min)
    assert all(axis_arr_out.data[np.where(in_dat > a_max)] == a_max)


@pytest.mark.parametrize("value", [-100, 0, 100])
@pytest.mark.parametrize("subtrahend", [False, True])
def test_const_difference(value: float, subtrahend: bool):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    axis_arr_in = AxisArray(in_dat, dims=["time", "ch"])

    proc = const_difference(value, subtrahend)
    axis_arr_out = proc.send(axis_arr_in)
    assert np.array_equal(axis_arr_out.data, (in_dat - value) if subtrahend else (value - in_dat))


def test_invert():
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    axis_arr_in = AxisArray(in_dat, dims=["time", "ch"])
    proc = invert()
    axis_arr_out = proc.send(axis_arr_in)
    assert np.array_equal(axis_arr_out.data, 1 / in_dat)


@pytest.mark.parametrize("base", [np.e, 2, 10])
def test_log(base: float):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    axis_arr_in = AxisArray(in_dat, dims=["time", "ch"])
    proc = log(base)
    axis_arr_out = proc.send(axis_arr_in)
    assert np.array_equal(axis_arr_out.data, np.log(in_dat) / np.log(base))


@pytest.mark.parametrize("scale_factor", [0.1, 0.5, 2.0, 10.0])
def test_scale(scale_factor: float):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    axis_arr_in = AxisArray(in_dat, dims=["time", "ch"])

    proc = scale(scale_factor)
    axis_arr_out = proc.send(axis_arr_in)

    assert axis_arr_out.data.shape == (n_times, n_chans)
    assert np.array_equal(axis_arr_out.data, in_dat * scale_factor)
