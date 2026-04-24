"""Tests for ezmsg.sigproc.util.array backend-agnostic helpers.

Each helper is exercised against numpy, mlx, and torch-mps. MLX and torch
are skipped on non-Apple-Silicon platforms via the helpers in
tests/helpers/util.py.
"""

import numpy as np
import pytest

from ezmsg.sigproc.util.array import (
    array_device,
    is_complex_dtype,
    is_float_dtype,
    xp_asarray,
    xp_create,
    xp_empty,
    xp_flip,
    xp_itemsize,
)
from tests.helpers.util import requires_mlx, requires_torch_mps

# ---------------------------------------------------------------------------
# Backend fixture
# ---------------------------------------------------------------------------


def _numpy_backend():
    return {
        "name": "numpy",
        "xp": np,
        "to_array": lambda data, dtype=np.float32: np.asarray(data, dtype=dtype),
        "to_numpy": np.asarray,
        "float32": np.float32,
        "int32": np.int32,
        "complex64": np.complex64,
        "supports_empty": True,
        "supports_device_kw": False,  # numpy.asarray has no 'device' kwarg
    }


def _mlx_backend():
    import mlx.core as mx

    return {
        "name": "mlx",
        "xp": mx,
        "to_array": lambda data, dtype=mx.float32: mx.array(np.asarray(data, dtype=np.float32)).astype(dtype),
        "to_numpy": lambda a: np.asarray(a),
        "float32": mx.float32,
        "int32": mx.int32,
        "complex64": mx.complex64,
        "supports_empty": False,  # mlx.core has no `empty`; xp_empty falls back
        "supports_device_kw": False,  # mlx.asarray doesn't accept `device`
    }


def _torch_mps_backend():
    import torch

    def _to_mps(data, dtype=torch.float32):
        return torch.as_tensor(np.asarray(data), dtype=dtype).to("mps")

    return {
        "name": "torch-mps",
        "xp": torch,
        "to_array": _to_mps,
        "to_numpy": lambda a: a.detach().cpu().numpy(),
        "float32": torch.float32,
        "int32": torch.int32,
        "complex64": torch.complex64,
        "supports_empty": True,
        "supports_device_kw": True,
    }


@pytest.fixture(
    params=[
        pytest.param("numpy", id="numpy"),
        pytest.param("mlx", id="mlx", marks=requires_mlx),
        pytest.param("torch-mps", id="torch-mps", marks=requires_torch_mps),
    ]
)
def backend(request):
    return {
        "numpy": _numpy_backend,
        "mlx": _mlx_backend,
        "torch-mps": _torch_mps_backend,
    }[request.param]()


# ---------------------------------------------------------------------------
# array_device
# ---------------------------------------------------------------------------


def test_array_device_returns_something_or_none(backend):
    """``array_device`` must not raise on any supported backend; it returns
    either a device object (numpy 'cpu', torch device) or ``None`` for
    device-less MLX."""
    arr = backend["to_array"]([1.0, 2.0, 3.0])
    dev = array_device(arr)
    if backend["name"] == "mlx":
        # array_api_compat cannot introspect MLX arrays; helper swallows the
        # AttributeError and returns None.
        assert dev is None
    else:
        assert dev is not None


# ---------------------------------------------------------------------------
# xp_asarray
# ---------------------------------------------------------------------------


def test_xp_asarray_basic(backend):
    xp = backend["xp"]
    out = xp_asarray(xp, [1.0, 2.0, 3.0])
    # Round-trip through numpy for value comparison.
    np.testing.assert_array_equal(backend["to_numpy"](out), np.array([1.0, 2.0, 3.0]))


def test_xp_asarray_with_dtype(backend):
    xp = backend["xp"]
    out = xp_asarray(xp, [1, 2, 3], dtype=backend["float32"])
    assert backend["to_numpy"](out).dtype == np.float32


def test_xp_asarray_omits_unsupported_device_kwarg(backend):
    """Passing device=None should never be forwarded (MLX would reject it)."""
    xp = backend["xp"]
    # device=None is the default; explicitly pass it to exercise the skip.
    out = xp_asarray(xp, [1.0, 2.0], device=None)
    np.testing.assert_array_equal(backend["to_numpy"](out), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# xp_create
# ---------------------------------------------------------------------------


def test_xp_create_zeros(backend):
    xp = backend["xp"]
    out = xp_create(xp.zeros, (4, 3), dtype=backend["float32"])
    arr = backend["to_numpy"](out)
    assert arr.shape == (4, 3)
    assert arr.dtype == np.float32
    assert (arr == 0).all()


def test_xp_create_ones(backend):
    xp = backend["xp"]
    out = xp_create(xp.ones, (2, 2), dtype=backend["float32"])
    arr = backend["to_numpy"](out)
    assert arr.shape == (2, 2)
    assert (arr == 1).all()


# ---------------------------------------------------------------------------
# xp_empty
# ---------------------------------------------------------------------------


def test_xp_empty_shape_and_dtype(backend):
    xp = backend["xp"]
    out = xp_empty(xp, (5, 2), dtype=backend["float32"])
    arr = backend["to_numpy"](out)
    assert arr.shape == (5, 2)
    assert arr.dtype == np.float32


def test_xp_empty_mlx_falls_back_to_zeros():
    """MLX exposes no ``empty`` — xp_empty must route to ``zeros``."""
    mx = pytest.importorskip("mlx.core")
    try:
        from tests.helpers.util import _has_mlx  # noqa: F401
    except ImportError:
        pass

    out = xp_empty(mx, (3, 4), dtype=mx.float32)
    np.testing.assert_array_equal(np.asarray(out), np.zeros((3, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# xp_flip
# ---------------------------------------------------------------------------


def test_xp_flip_1d(backend):
    arr = backend["to_array"](np.arange(5, dtype=np.float32))
    out = xp_flip(arr, axis=0)
    np.testing.assert_array_equal(backend["to_numpy"](out), np.array([4, 3, 2, 1, 0], dtype=np.float32))


def test_xp_flip_axis_0(backend):
    arr = backend["to_array"](np.arange(12, dtype=np.float32).reshape(4, 3))
    out = xp_flip(arr, axis=0)
    np.testing.assert_array_equal(
        backend["to_numpy"](out),
        np.flip(np.arange(12, dtype=np.float32).reshape(4, 3), axis=0),
    )


def test_xp_flip_axis_1(backend):
    arr = backend["to_array"](np.arange(12, dtype=np.float32).reshape(4, 3))
    out = xp_flip(arr, axis=1)
    np.testing.assert_array_equal(
        backend["to_numpy"](out),
        np.flip(np.arange(12, dtype=np.float32).reshape(4, 3), axis=1),
    )


def test_xp_flip_preserves_backend(backend):
    """Output must stay in the input's namespace — no silent coercion."""
    arr = backend["to_array"]([1.0, 2.0, 3.0])
    out = xp_flip(arr, axis=0)
    assert type(out).__module__.split(".")[0] == type(arr).__module__.split(".")[0]


# ---------------------------------------------------------------------------
# xp_itemsize
# ---------------------------------------------------------------------------


def test_xp_itemsize_float32(backend):
    assert xp_itemsize(backend["float32"]) == 4


def test_xp_itemsize_int32(backend):
    assert xp_itemsize(backend["int32"]) == 4


def test_xp_itemsize_raises_on_unknown():
    class Bogus:
        pass

    with pytest.raises(TypeError, match="byte size"):
        xp_itemsize(Bogus())


# ---------------------------------------------------------------------------
# is_complex_dtype / is_float_dtype
# ---------------------------------------------------------------------------


def test_is_complex_dtype_true(backend):
    assert is_complex_dtype(backend["complex64"]) is True


def test_is_complex_dtype_false(backend):
    assert is_complex_dtype(backend["float32"]) is False
    assert is_complex_dtype(backend["int32"]) is False


def test_is_float_dtype_true(backend):
    assert is_float_dtype(backend["xp"], backend["float32"]) is True


def test_is_float_dtype_false(backend):
    assert is_float_dtype(backend["xp"], backend["int32"]) is False
    assert is_float_dtype(backend["xp"], backend["complex64"]) is False
