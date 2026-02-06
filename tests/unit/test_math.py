import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.math.abs import AbsTransformer
from ezmsg.sigproc.math.clip import ClipSettings, ClipTransformer
from ezmsg.sigproc.math.difference import ConstDifferenceSettings, ConstDifferenceTransformer
from ezmsg.sigproc.math.invert import InvertTransformer
from ezmsg.sigproc.math.log import LogSettings, LogTransformer
from ezmsg.sigproc.math.pow import PowSettings, PowTransformer
from ezmsg.sigproc.math.scale import ScaleSettings, ScaleTransformer
from tests.helpers.empty_time import check_empty_result, make_empty_msg


def test_abs():
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    xformer = AbsTransformer()
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, np.abs(in_dat))


@pytest.mark.parametrize("min_val", [1, 2])
@pytest.mark.parametrize("max_val", [133, 134])
def test_clip(min_val: float, max_val: float):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = ClipTransformer(ClipSettings(min=min_val, max=max_val))
    msg_out = xformer(msg_in)

    assert all(msg_out.data[np.where(in_dat < min_val)] == min_val)
    assert all(msg_out.data[np.where(in_dat > max_val)] == max_val)


@pytest.mark.parametrize("value", [-100, 0, 100])
@pytest.mark.parametrize("subtrahend", [False, True])
def test_const_difference(value: float, subtrahend: bool):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = ConstDifferenceTransformer(ConstDifferenceSettings(value=value, subtrahend=subtrahend))
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, (in_dat - value) if subtrahend else (value - in_dat))


def test_invert():
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    xformer = InvertTransformer()
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, 1 / in_dat)


@pytest.mark.parametrize("base", [np.e, 2, 10])
@pytest.mark.parametrize("dtype", [int, float])
@pytest.mark.parametrize("clip_zero", [False, True])
def test_log(base: float, dtype, clip_zero: bool):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans).astype(dtype)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    xformer = LogTransformer(LogSettings(base=base, clip_zero=clip_zero))
    msg_out = xformer(msg_in)
    if clip_zero and dtype is float:
        in_dat = np.clip(in_dat, a_min=np.finfo(msg_in.data.dtype).tiny, a_max=None)
    assert np.array_equal(msg_out.data, np.log(in_dat) / np.log(base))


@pytest.mark.parametrize("scale_factor", [0.1, 0.5, 2.0, 10.0])
def test_scale(scale_factor: float):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = ScaleTransformer(ScaleSettings(scale=scale_factor))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == (n_times, n_chans)
    assert np.array_equal(msg_out.data, in_dat * scale_factor)


@pytest.mark.parametrize("exponent", [0.5, 2.0, 3.0])
def test_pow(exponent: float):
    n_times = 130
    n_chans = 255
    in_dat = np.abs(np.arange(n_times * n_chans).reshape(n_times, n_chans)).astype(float) + 1.0
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = PowTransformer(PowSettings(exponent=exponent))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == (n_times, n_chans)
    assert np.allclose(msg_out.data, in_dat**exponent)


def test_abs_empty_time():
    from ezmsg.sigproc.math.abs import AbsTransformer

    proc = AbsTransformer()
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_clip_empty_time():
    from ezmsg.sigproc.math.clip import ClipSettings, ClipTransformer

    proc = ClipTransformer(ClipSettings(min=0.0, max=1.0))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_const_difference_empty_time():
    from ezmsg.sigproc.math.difference import ConstDifferenceSettings, ConstDifferenceTransformer

    proc = ConstDifferenceTransformer(ConstDifferenceSettings(value=5.0))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_invert_empty_time():
    from ezmsg.sigproc.math.invert import InvertTransformer

    proc = InvertTransformer()
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_log_empty_time():
    from ezmsg.sigproc.math.log import LogSettings, LogTransformer

    proc = LogTransformer(LogSettings(base=np.e, clip_zero=True))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_pow_empty_time():
    from ezmsg.sigproc.math.pow import PowSettings, PowTransformer

    proc = PowTransformer(PowSettings(exponent=2.0))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_scale_empty_time():
    from ezmsg.sigproc.math.scale import ScaleSettings, ScaleTransformer

    proc = ScaleTransformer(ScaleSettings(scale=2.0))
    result = proc(make_empty_msg())
    check_empty_result(result)
