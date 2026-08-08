"""Split scipy IIR filtering across threads along a non-sample axis.

Each channel of an IIR filter is an independent recurrence, so splitting the
non-sample axes across threads changes nothing about the arithmetic any one
channel sees -- the result is bit-identical to the single-threaded call, not
merely close. That is what makes this usable in a pipeline whose whole point is
that offline output matches online output exactly.

The win is real but strictly large-chunk. scipy's ``sosfilt``/``lfilter`` are
single-threaded C loops, so the speedup tracks chunk *bytes* -- measured
end-to-end through FilterTransformer, order-4 SOS, three independent reps:

    0.25 MiB   0.50x  0.82x  0.79x    loss
    0.50 MiB   0.99x  1.07x  1.18x    break-even
    1.00 MiB   1.65x  1.74x  2.25x    win
    2.00 MiB   2.28x  2.36x  2.43x    win
    8.00 MiB       ~3.3x               win
   32.00 MiB       ~3.4x               win

The ratio depends on total bytes rather than on the channel/sample split: at a
fixed 1 MiB, 256x512, 512x256 and 1024x128 all land within 1.53-1.63x. Below
~0.5 MiB the dispatch cost dominates and threading is a *loss* -- hence
:data:`DEFAULT_MIN_BYTES` and the hard gate in :func:`should_thread`. Online
chunk sizes stay single-threaded.

The pool is module-level and created once, so a graph with many filter units
shares one bounded set of workers rather than spawning a pool per unit, and no
per-chunk thread creation ever happens. Note that this is *explicit* threading
we control; it is unrelated to the implicit threading inside BLAS or
``scipy.fft``, which is governed by environment variables and can nest badly
underneath a pool like this one. scipy's IIR filters touch neither.
"""

import atexit
import math
import os
import threading
import typing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
from ezmsg.util.messages.axisarray import slice_along_axis

#: Chunks smaller than this (in bytes) are filtered single-threaded. On a
#: 12-core M-series machine 0.25 MiB loses (0.5-0.8x), 0.5 MiB is break-even
#: (0.99-1.18x) and 1 MiB is the first size that wins consistently
#: (1.65-2.25x), so that is the default. Tunable per filter via
#: ``FilterBaseSettings.thread_min_bytes``; raise it on machines with slower
#: memory or fewer cores.
DEFAULT_MIN_BYTES = 1 << 20

#: Upper bound on pool size. IIR filtering is memory-bandwidth bound well before
#: it is core bound, so past a handful of workers the curve flattens (measured
#: ~4.5x at 6 threads on 12 cores).
DEFAULT_MAX_WORKERS = 8

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _worker_count() -> int:
    return max(1, min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1))


def get_pool() -> ThreadPoolExecutor:
    """The shared worker pool, created on first use and closed at interpreter exit."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=_worker_count(), thread_name_prefix="ezmsg-filt")
            atexit.register(shutdown_pool)
        return _pool


def shutdown_pool() -> None:
    """Release the shared pool. Registered with ``atexit``; safe to call directly."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=True)


def _split_axis(shape: tuple[int, ...], axis_idx: int) -> int | None:
    """Pick the non-sample axis to divide across threads: the longest one.

    Returns None when there is nothing to split (1-D input, or every other axis
    is degenerate), in which case the caller must stay single-threaded.
    """
    best, best_len = None, 1
    for i, n in enumerate(shape):
        if i != axis_idx and n > best_len:
            best, best_len = i, n
    return best


def should_thread(
    data: npt.NDArray,
    axis_idx: int,
    min_bytes: int = DEFAULT_MIN_BYTES,
) -> bool:
    """Whether splitting ``data`` across threads is likely to pay for itself."""
    if min_bytes <= 0:  # explicitly disabled
        return False
    if not isinstance(data, np.ndarray) or data.nbytes < min_bytes:
        return False
    split = _split_axis(data.shape, axis_idx)
    # Need at least two blocks for threading to mean anything.
    return split is not None and data.shape[split] >= 2


def filt_threaded(
    filt_func: typing.Callable,
    coefs: tuple,
    data: npt.NDArray,
    axis_idx: int,
    zi: npt.NDArray,
    zi_axis_offset: int,
    min_bytes: int = DEFAULT_MIN_BYTES,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Apply ``filt_func`` in parallel over blocks of a non-sample axis.

    Args:
        filt_func: ``scipy.signal.sosfilt`` or ``scipy.signal.lfilter``.
        coefs: Positional coefficient args for ``filt_func``.
        data: Input array; the sample axis is ``axis_idx``.
        zi: Filter state.
        zi_axis_offset: How far ``zi``'s axes are shifted relative to ``data``'s.
            1 for SOS (leading ``n_sections`` axis), 0 for BA.
        min_bytes: Threshold below which this falls back to one call.

    Returns:
        ``(y, zf)``, bit-identical to the equivalent single-threaded call.
    """
    if not should_thread(data, axis_idx, min_bytes):
        return filt_func(*coefs, data, axis=axis_idx, zi=zi)

    split = _split_axis(data.shape, axis_idx)
    n_split = data.shape[split]
    n_workers = min(_worker_count(), n_split)
    block = math.ceil(n_split / n_workers)
    bounds = [(s, min(s + block, n_split)) for s in range(0, n_split, block)]

    def work(bound: tuple[int, int]):
        lo, hi = bound
        sl = slice(lo, hi)
        return filt_func(
            *coefs,
            slice_along_axis(data, sl, split),
            axis=axis_idx,
            zi=slice_along_axis(zi, sl, split + zi_axis_offset),
        )

    results = list(get_pool().map(work, bounds))
    y = np.concatenate([r[0] for r in results], axis=split)
    zf = np.concatenate([r[1] for r in results], axis=split + zi_axis_offset)
    return y, zf
