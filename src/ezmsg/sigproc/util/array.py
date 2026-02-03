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
