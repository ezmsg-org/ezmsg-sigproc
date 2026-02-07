"""Convert AxisArray data to a target array backend.

This module provides a transformer that converts AxisArray payloads between
array backends (NumPy, MLX, PyTorch, CuPy, JAX). Useful for wiring a
conversion step between nodes — e.g., numpy → MLX before a GPU-accelerated
filter, or MLX → numpy before a scipy-dependent node.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

import enum
import importlib
import importlib.util

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ezmsg.sigproc.util.array import xp_asarray


def _build_backend_members():
    members = ["numpy"]
    for name in ("mlx", "torch", "cupy", "jax"):
        if importlib.util.find_spec(name) is not None:
            members.append(name)
    return members


ArrayBackend = enum.StrEnum("ArrayBackend", _build_backend_members())


_BACKEND_MODULE_MAP = {
    "numpy": "numpy",
    "mlx": "mlx.core",
    "torch": "torch",
    "cupy": "cupy",
    "jax": "jax.numpy",
}

_BACKEND_TYPE_PREFIX = {
    "numpy": "numpy",
    "mlx": "mlx",
    "torch": "torch",
    "cupy": "cupy",
    "jax": "jax",
}


def _get_backend_module(backend: str):
    """Lazily import and return the array namespace module for *backend*."""
    module_name = _BACKEND_MODULE_MAP[backend]
    return importlib.import_module(module_name)


def _detect_backend(data) -> str:
    """Identify which backend an array belongs to via its module prefix."""
    module = type(data).__module__
    for backend, prefix in _BACKEND_TYPE_PREFIX.items():
        if module == prefix or module.startswith(prefix + "."):
            return backend
    raise TypeError(f"Unrecognized array type: {type(data)} (module={module})")


class AsArraySettings(ez.Settings):
    backend: ArrayBackend = ArrayBackend.numpy
    """Target array backend."""

    dtype: str | None = None
    """Target dtype as a string (e.g. "float32", "float64"). None keeps the original dtype."""


class AsArrayTransformer(BaseTransformer[AsArraySettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        target_backend = str(self.settings.backend)
        dtype_str = self.settings.dtype
        data = message.data

        current_backend = _detect_backend(data)
        target_xp = _get_backend_module(target_backend)
        resolved_dtype = getattr(target_xp, dtype_str) if dtype_str is not None else None

        # No-op fast path: already correct backend and no dtype change.
        if current_backend == target_backend and resolved_dtype is None:
            return message

        # Same backend, dtype change only.
        if current_backend == target_backend:
            new_data = xp_asarray(target_xp, data, dtype=resolved_dtype)
            return replace(message, data=new_data)

        # Cross-backend: go through numpy as an intermediate.
        np_data = np.asarray(data)
        new_data = xp_asarray(target_xp, np_data, dtype=resolved_dtype)
        return replace(message, data=new_data)


class AsArray(BaseTransformerUnit[AsArraySettings, AxisArray, AxisArray, AsArrayTransformer]):
    SETTINGS = AsArraySettings
