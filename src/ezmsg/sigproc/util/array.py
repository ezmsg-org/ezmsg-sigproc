"""Portable helpers for Array API interoperability.

These utilities smooth over differences between Array API libraries
(NumPy, PyTorch, MLX, CuPy, etc.) — in particular around ``device``
placement and ``dtype`` introspection, which are not uniformly supported.
"""

import numpy as np


def array_device(x):
    """Return the device of an array, or ``None`` for device-less libraries."""
    try:
        from array_api_compat import device

        return device(x)
    except (AttributeError, TypeError):
        return None


def xp_asarray(xp, obj, *, dtype=None, device=None):
    """Portable ``xp.asarray`` that omits unsupported kwargs.

    Some Array API libraries (e.g. MLX) don't accept a ``device`` keyword.
    This helper builds the kwargs dict dynamically so that only supported
    arguments are forwarded.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    return xp.asarray(obj, **kwargs)


def xp_create(fn, *args, dtype=None, device=None, **extra):
    """Call a creation function (``zeros``, ``ones``, ``eye``) portably.

    Omits ``device`` if it is ``None`` (for libraries that don't support it).
    """
    kwargs = dict(extra)
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    return fn(*args, **kwargs)


def xp_empty(xp, shape, *, dtype=None):
    """Portable ``xp.empty`` with a ``zeros`` fallback for backends (e.g. MLX)
    that don't expose ``empty``. MLX is lazy so the extra zero init is near-free;
    on eager backends ``empty`` is preferred when available."""
    fn = getattr(xp, "empty", None) or xp.zeros
    if dtype is not None:
        return fn(shape, dtype=dtype)
    return fn(shape)


def xp_flip(arr, axis):
    """Reverse ``arr`` along ``axis``, portable across backends.

    numpy/cupy/torch expose ``xp.flip``; MLX does not but supports negative-
    step slicing, which we use as a universal fallback.
    """
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(None, None, -1)
    return arr[tuple(idx)]


def xp_itemsize(dtype) -> int:
    """Bytes per element of ``dtype``, portable across backends.

    numpy/cupy/torch expose ``.itemsize``; MLX exposes ``.size``.
    """
    size = getattr(dtype, "itemsize", None)
    if size is None:
        size = getattr(dtype, "size", None)
    if size is None:
        raise TypeError(f"Cannot determine byte size of dtype {dtype!r}")
    return size


def is_complex_dtype(dtype) -> bool:
    """Check whether *dtype* is a complex type, portably across backends."""
    if hasattr(dtype, "kind"):
        return dtype.kind == "c"
    return "complex" in str(dtype).lower()


def is_float_dtype(xp, dtype) -> bool:
    """Check whether *dtype* is a real floating-point type, portably."""
    try:
        return xp.isdtype(dtype, "real floating")
    except AttributeError:
        pass
    # Fallback for libraries without isdtype (e.g. MLX).
    try:
        return xp.issubdtype(dtype, xp.floating)
    except (AttributeError, TypeError):
        return np.issubdtype(np.dtype(dtype), np.floating)
