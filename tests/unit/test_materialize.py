import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.materialize import (
    MaterializeMode,
    MaterializeSettings,
    MaterializeTransformer,
    materialize_array,
)
from tests.helpers.empty_time import check_empty_result, make_empty_msg, make_msg
from tests.helpers.util import requires_mlx

ALL_MODES = [MaterializeMode.SYNC, MaterializeMode.ASYNC, MaterializeMode.OFF]


def test_numpy_passthrough():
    msg = make_msg()
    xformer = MaterializeTransformer()
    result = xformer(msg)
    assert result is msg


@pytest.mark.parametrize("mode", ALL_MODES)
def test_numpy_passthrough_all_modes(mode):
    """Every mode is a no-op on a backend that has nothing to evaluate."""
    msg = make_msg()
    result = MaterializeTransformer(MaterializeSettings(mode=mode))(msg)
    assert result is msg


def test_default_mode_is_sync():
    assert MaterializeTransformer().settings.mode is MaterializeMode.SYNC


def test_mode_accepts_str():
    """Settings coming from YAML/JSON arrive as plain strings."""
    assert materialize_array(np.ones(3), "async") is not None
    assert MaterializeMode("async") is MaterializeMode.ASYNC


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        materialize_array(np.ones(3), "eventually")


@requires_mlx
def test_mlx_evaluates():
    mx = pytest.importorskip("mlx.core")
    a = mx.ones((10, 3))
    b = mx.ones((10, 3))
    lazy_sum = a + b  # lazy — not yet evaluated
    msg = AxisArray(lazy_sum, dims=["time", "ch"])
    xformer = MaterializeTransformer()
    result = xformer(msg)
    assert isinstance(result.data, mx.array)
    np.testing.assert_array_equal(np.array(result.data), np.full((10, 3), 2.0))


@requires_mlx
@pytest.mark.parametrize("mode", ALL_MODES)
def test_mlx_values_unchanged(mode):
    """Materializing must not alter the values, whichever mode is used."""
    mx = pytest.importorskip("mlx.core")
    lazy = mx.ones((10, 3)) + mx.ones((10, 3))
    msg = AxisArray(lazy, dims=["time", "ch"])
    result = MaterializeTransformer(MaterializeSettings(mode=mode))(msg)
    assert result is msg
    np.testing.assert_array_equal(np.array(result.data), np.full((10, 3), 2.0))


@requires_mlx
def test_mlx_empty_message():
    """A zero-length message is safe to materialize (and forces nothing)."""
    mx = pytest.importorskip("mlx.core")
    msg = AxisArray(mx.zeros((0, 3)), dims=["time", "ch"])
    result = MaterializeTransformer()(msg)
    check_empty_result(result)


@requires_mlx
@pytest.mark.parametrize("mode", [MaterializeMode.SYNC, MaterializeMode.ASYNC])
def test_bounds_lazy_graph_growth(mode):
    """SYNC and ASYNC both stop a carried-forward lazy graph from accumulating.

    This is the property the node exists for. A recurrence that feeds its own
    output back in as state -- every stateful transformer in this package -- grows
    an unbounded computation graph if nothing ever evaluates it, retaining one
    input's worth of buffers per call. Both modes detach the graph; OFF does not,
    and is the control that proves the measurement can see the difference.
    """
    mx = pytest.importorskip("mlx.core")

    n_iter = 100
    chunk_bytes = 256 * 512 * 4

    def run(m):
        state = mx.zeros((256, 1), dtype=mx.float32)
        mx.eval(state)
        base = mx.get_active_memory()
        peak = 0
        for i in range(n_iter):
            chunk = mx.ones((256, 512), dtype=mx.float32) * float(i)
            mx.eval(chunk)  # a real materialized buffer, not just a graph node
            state = state + chunk.sum(axis=-1, keepdims=True)
            materialize_array(state, m)
            del chunk  # only the lazy graph can still be holding it
            peak = max(peak, mx.get_active_memory() - base)
        mx.eval(state)
        return peak

    bounded = run(mode)
    unbounded = run(MaterializeMode.OFF)
    # Bounded retains ~one chunk; unbounded retains all of them.
    assert bounded < 4 * chunk_bytes, f"{mode} retained {bounded} bytes, expected ~{chunk_bytes}"
    assert unbounded > 10 * bounded, f"control failed to show growth: {mode}={bounded} bytes vs off={unbounded} bytes"


def test_empty_time():
    msg = make_empty_msg()
    xformer = MaterializeTransformer()
    result = xformer(msg)
    check_empty_result(result)
