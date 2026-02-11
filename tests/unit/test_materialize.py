import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.materialize import MaterializeTransformer
from tests.helpers.empty_time import check_empty_result, make_empty_msg, make_msg
from tests.helpers.util import requires_mlx


def test_numpy_passthrough():
    msg = make_msg()
    xformer = MaterializeTransformer()
    result = xformer(msg)
    assert result is msg


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


def test_empty_time():
    msg = make_empty_msg()
    xformer = MaterializeTransformer()
    result = xformer(msg)
    check_empty_result(result)
