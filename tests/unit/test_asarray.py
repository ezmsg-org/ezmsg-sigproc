import importlib.util

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.asarray import (
    ArrayBackend,
    AsArraySettings,
    AsArrayTransformer,
    _detect_backend,
    _get_backend_module,
)
from tests.helpers.empty_time import check_empty_result, make_empty_msg, make_msg

# -- Helpers ------------------------------------------------------------------

_NON_NUMPY_BACKENDS = [b for b in ArrayBackend if b != "numpy"]


def _to_backend(data: np.ndarray, backend: str):
    """Convert a numpy array to the given backend."""
    xp = _get_backend_module(backend)
    return xp.asarray(data)


# -- Enum tests ---------------------------------------------------------------


def test_numpy_always_present():
    assert "numpy" in ArrayBackend.__members__


def test_every_member_importable():
    for member in ArrayBackend:
        assert importlib.util.find_spec(str(member)) is not None


# -- _detect_backend tests ----------------------------------------------------


def test_detect_backend_numpy():
    assert _detect_backend(np.array([1.0])) == "numpy"


@pytest.mark.parametrize("backend", _NON_NUMPY_BACKENDS)
def test_detect_backend_other(backend):
    xp = _get_backend_module(str(backend))
    arr = xp.asarray([1.0])
    assert _detect_backend(arr) == str(backend)


# -- No-op fast path -----------------------------------------------------------


def test_noop_returns_same_message():
    msg = make_msg()
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend.numpy))
    result = proc(msg)
    assert result is msg


# -- Dtype cast (same backend) ------------------------------------------------


def test_dtype_cast_numpy():
    data = np.ones((10, 3), dtype=np.float32)
    axes = frozendict(
        {
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(3).astype(str), dims=["ch"]),
        }
    )
    msg = AxisArray(data, dims=["time", "ch"], axes=axes)
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend.numpy, dtype="float64"))
    result = proc(msg)
    assert result.data.dtype == np.float64
    assert result.dims == msg.dims
    assert result.axes == msg.axes


# -- Cross-backend: numpy → other → numpy ------------------------------------


@pytest.mark.parametrize("backend", _NON_NUMPY_BACKENDS)
def test_numpy_to_other(backend):
    msg = make_msg()
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend[str(backend)]))
    result = proc(msg)
    assert _detect_backend(result.data) == str(backend)
    np.testing.assert_allclose(np.asarray(result.data), msg.data)


@pytest.mark.parametrize("backend", _NON_NUMPY_BACKENDS)
def test_other_to_numpy(backend):
    orig_data = np.random.randn(10, 3).astype(np.float64)
    foreign_data = _to_backend(orig_data, str(backend))
    axes = frozendict(
        {
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(3).astype(str), dims=["ch"]),
        }
    )
    msg = AxisArray(foreign_data, dims=["time", "ch"], axes=axes)
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend.numpy))
    result = proc(msg)
    assert _detect_backend(result.data) == "numpy"
    np.testing.assert_allclose(result.data, orig_data)
    assert result.dims == msg.dims
    assert result.axes == msg.axes


@pytest.mark.parametrize("backend", _NON_NUMPY_BACKENDS)
def test_roundtrip(backend):
    msg = make_msg()
    to_other = AsArrayTransformer(AsArraySettings(backend=ArrayBackend[str(backend)]))
    to_numpy = AsArrayTransformer(AsArraySettings(backend=ArrayBackend.numpy))
    result = to_numpy(to_other(msg))
    assert _detect_backend(result.data) == "numpy"
    np.testing.assert_allclose(result.data, msg.data)


# -- Cross-backend + dtype ----------------------------------------------------


@pytest.mark.parametrize("backend", _NON_NUMPY_BACKENDS)
def test_cross_backend_with_dtype(backend):
    data = np.ones((10, 3), dtype=np.float64)
    axes = frozendict(
        {
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(3).astype(str), dims=["ch"]),
        }
    )
    msg = AxisArray(data, dims=["time", "ch"], axes=axes)
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend[str(backend)], dtype="float32"))
    result = proc(msg)
    assert _detect_backend(result.data) == str(backend)
    xp = _get_backend_module(str(backend))
    assert result.data.dtype == xp.float32
    assert result.dims == msg.dims
    assert result.axes == msg.axes


# -- Metadata / axes preservation ---------------------------------------------


def test_axes_preserved():
    msg = make_msg()
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend.numpy, dtype="float32"))
    result = proc(msg)
    assert result.dims == msg.dims
    assert result.axes == msg.axes


@pytest.mark.parametrize("backend", _NON_NUMPY_BACKENDS)
def test_axes_preserved_cross_backend(backend):
    msg = make_msg()
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend[str(backend)]))
    result = proc(msg)
    assert result.dims == msg.dims
    assert result.axes == msg.axes


# -- Empty time dimension ------------------------------------------------------


def test_empty_time_numpy():
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend.numpy))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_empty_time_numpy_dtype():
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend.numpy, dtype="float32"))
    result = proc(make_empty_msg())
    check_empty_result(result)
    assert result.data.dtype == np.float32


@pytest.mark.parametrize("backend", _NON_NUMPY_BACKENDS)
def test_empty_time_cross_backend(backend):
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend[str(backend)]))
    result = proc(make_empty_msg())
    assert _detect_backend(result.data) == str(backend)
    # The time dimension should still be 0.
    time_idx = result.dims.index("time")
    assert result.data.shape[time_idx] == 0
