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


_StrEnum = getattr(enum, "StrEnum", None)
if _StrEnum is None:
    # Python 3.10 backport: _generate_next_value_ makes the functional API
    # assign name-equal-to-value, and __str__ returns the value (3.11 changed
    # mixed-in Enum __str__ to use the data type's __str__, but 3.10 didn't).
    class _StrEnum(str, enum.Enum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name

        def __str__(self):
            return self.value


ArrayBackend = _StrEnum("ArrayBackend", _build_backend_members())


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


_MLX_CACHE_LIMIT_APPLIED: float | None = None
"""Value this process has already handed to ``mx.set_cache_limit``.

Module-level because the limit is a property of the process, not of a node: two
``AsArray`` nodes converting to MLX in one process share one allocator.
"""


def _apply_mlx_cache_limit(limit_mb: float) -> None:
    """Bound the MLX buffer cache for this process. Idempotent.

    MLX caches every freed buffer in a multimap keyed by *exact* byte size, and
    only reuses one within ``min(2 * size, size + 2 * page_size)`` of the
    request -- above ~32 KiB that is effectively an exact match. A graph whose
    message length varies therefore mints a permanent new size class per length,
    in a cache whose default limit is the whole machine (23 GiB on a 24 GiB
    host). Measured on a 30 kHz feature chain that sees 40 distinct message
    lengths over an hour: 6015 MiB of physical footprint unbounded, 966 MiB at
    512 MiB, with *higher* throughput at the limit because there is less memory
    pressure.

    Eviction is LRU, which is what makes a limit the right tool rather than a
    blunt one: the steady-state shape is re-touched every message and stays at
    the head, while one-off shapes from a stall fall to the tail and are freed
    first. Measured, the hot allocation is unaffected (0.97x) after 39 rare
    shapes have been evicted past a 128 MiB limit.
    """
    global _MLX_CACHE_LIMIT_APPLIED
    if _MLX_CACHE_LIMIT_APPLIED == limit_mb:
        return
    import mlx.core as mx

    if _MLX_CACHE_LIMIT_APPLIED is not None:
        ez.logger.warning(
            f"MLX cache limit already set to {_MLX_CACHE_LIMIT_APPLIED} MiB in this process; "
            f"overriding with {limit_mb} MiB. The limit is process-global, so the last AsArray "
            "node to convert wins -- give every MLX-targeting AsArray in a process the same "
            "mlx_cache_limit_mb."
        )
    mx.set_cache_limit(int(limit_mb * 1024 * 1024))
    _MLX_CACHE_LIMIT_APPLIED = limit_mb


class AsArraySettings(ez.Settings):
    backend: ArrayBackend = ArrayBackend.numpy
    """Target array backend."""

    dtype: str | None = None
    """Target dtype as a string (e.g. "float32", "float64"). None keeps the original dtype."""

    mlx_cache_limit_mb: float | None = 512.0
    """Cap the MLX buffer cache (MiB) for the process that runs this node.

    Applied only when :attr:`backend` is MLX, once per process, on the first
    message. ``None`` leaves MLX's default, which is the size of the machine.

    Sizing: one *distinct message shape* costs roughly **50x the message
    payload** in cached buffers -- about 20 intermediates across a typical chain,
    each keeping its own size class. Measured cache for a steady-state chain,
    against ``samples x channels x 4`` bytes per message: 46x at 256 ch x 1200
    samples, 67x at 256 ch x 300. So::

        limit_MiB ~= 50 * message_MiB * (distinct shapes to keep hot)

    A 256-channel, 300-sample float32 message is 0.29 MiB, so ~15 MiB per shape
    and the 512 MiB default holds ~26 distinct shapes. Steady state needs only
    one; the rest is headroom for the varying-length messages a stall produces.
    Raise it if the graph legitimately cycles through many shapes; the floor is
    one working set (~1x the ``50 *`` term), and 0 disables caching entirely at
    a measured 40% throughput cost.

    This is a process-global MLX setting, so it is shared with anything else
    using MLX in the same process. The default suits streaming graphs; large
    offline batch work in the same process may want it raised or set to
    ``None``."""


class AsArrayTransformer(BaseTransformer[AsArraySettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        target_backend = str(self.settings.backend)
        if target_backend == "mlx" and self.settings.mlx_cache_limit_mb is not None:
            # Here rather than in the Unit's initialize(): this transformer is
            # also used bare (offline chains, benchmarks), and the limit has to
            # be set in whichever process actually converts, which for a
            # multi-process graph is not the one that built it -- set_cache_limit
            # does not survive the spawn.
            _apply_mlx_cache_limit(self.settings.mlx_cache_limit_mb)
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
