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
from tests.helpers.util import requires_mlx

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


# -- MLX buffer cache limit ----------------------------------------------------


def test_mlx_cache_limit_default_is_set():
    """A default, not None: the growth it prevents is invisible in RSS.

    MLX buffers are IOKit allocations, so a graph whose message length varies
    can put gigabytes into the allocator's per-size cache while RSS stays flat.
    Users do not go looking for a knob they have no symptom for.
    """
    assert AsArraySettings().mlx_cache_limit_mb == 512.0


def test_mlx_cache_limit_not_applied_for_numpy_target(monkeypatch):
    """Converting TO numpy must not touch a process-global MLX setting."""
    import ezmsg.sigproc.asarray as asarray_module

    calls = []
    monkeypatch.setattr(asarray_module, "_apply_mlx_cache_limit", lambda mb: calls.append(mb))
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend.numpy, mlx_cache_limit_mb=128.0))
    proc(make_msg())
    assert calls == []


@requires_mlx
def test_mlx_cache_limit_applied_once_for_mlx_target(monkeypatch):
    import ezmsg.sigproc.asarray as asarray_module

    calls = []
    monkeypatch.setattr(asarray_module, "_apply_mlx_cache_limit", lambda mb: calls.append(mb))
    proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend.mlx, mlx_cache_limit_mb=128.0))
    for _ in range(3):
        proc(make_msg())
    # Called per message, but the applier itself is what dedupes -- see below.
    assert calls == [128.0, 128.0, 128.0]


@requires_mlx
def test_apply_mlx_cache_limit_is_idempotent_and_warns_on_conflict(monkeypatch):
    """The limit is process-global, so a second, different value is a conflict.

    Silently letting the last node win would make the effective limit depend on
    which unit happened to convert first.
    """
    import mlx.core as mx

    import ezmsg.sigproc.asarray as asarray_module

    sets = []
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: sets.append(n))
    monkeypatch.setattr(asarray_module, "_MLX_CACHE_LIMIT_APPLIED", None)
    warnings = []
    monkeypatch.setattr(asarray_module.ez.logger, "warning", lambda msg, *a: warnings.append(msg))

    asarray_module._apply_mlx_cache_limit(128.0)
    asarray_module._apply_mlx_cache_limit(128.0)  # same value: no-op
    assert sets == [128 * 1024 * 1024]
    assert warnings == []

    asarray_module._apply_mlx_cache_limit(256.0)  # different value: warn, override
    assert sets == [128 * 1024 * 1024, 256 * 1024 * 1024]
    assert len(warnings) == 1 and "process-global" in warnings[0]


@requires_mlx
def test_mlx_cache_limit_actually_bounds_the_cache():
    """End to end: the setting reaches the allocator and caps it."""
    import mlx.core as mx

    import ezmsg.sigproc.asarray as asarray_module

    previous = mx.set_cache_limit(2**40)
    applied = asarray_module._MLX_CACHE_LIMIT_APPLIED
    try:
        asarray_module._MLX_CACHE_LIMIT_APPLIED = None
        proc = AsArrayTransformer(AsArraySettings(backend=ArrayBackend.mlx, mlx_cache_limit_mb=32.0))
        proc(make_msg())
        mx.clear_cache()
        # Churn many distinct sizes; without a limit this cache grows unbounded.
        n_ch, step, n_sizes = 256, 300, 60
        for n in range(1, n_sizes):
            a = mx.zeros((n_ch, step * n))
            mx.eval(a)
            del a

        # MLX evicts down to the limit *before* admitting a freed buffer, then
        # admits it, so the cache is a high-water mark that one buffer can
        # exceed -- `set_cache_limit` never promised a hard ceiling. Which side
        # of the limit it lands on varies run to run: measured here it settles
        # at either the largest churned buffer (17.0 MB) or the largest two
        # (34.3 MB), against a 32 MB limit.
        #
        # Asserting `<= limit` therefore sampled a state MLX does not guarantee.
        # It failed every CI run from 2026-08-24; forcing a trim with one extra
        # allocation first (the previous attempt at this) worked 12/12 locally
        # and still failed one macOS job in four.
        #
        # The bound below is the one the allocator actually honours, and it is
        # not a loose one: the same churn with no limit leaves ~500 MB cached,
        # ten times this ceiling. It is what the test meant to prove.
        largest_buffer = n_ch * step * (n_sizes - 1) * 4  # float32
        assert mx.get_cache_memory() <= 32 * 1024 * 1024 + largest_buffer
    finally:
        mx.clear_cache()
        mx.set_cache_limit(previous)
        asarray_module._MLX_CACHE_LIMIT_APPLIED = applied
