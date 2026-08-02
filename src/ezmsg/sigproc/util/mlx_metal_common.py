"""Shared scaffolding for MLX+Metal recurrence kernels.

Both :mod:`sosfilt_mlx_metal` and :mod:`ewma_mlx_metal` apply a per-channel
recurrence over time and need the same boilerplate around the kernel: float32
promotion, batch-axis flatten/restore, and a chunked launch loop that carries
the per-chunk state forward. Those pieces live here so the kernel modules can
focus on their actual Metal source and per-op state layout.
"""

import numbers

import mlx.core as mx


def normalize_chunk_sizes(chunk_sizes, max_chunk_size):
    """Return sorted, unique, validated Metal kernel chunk sizes."""
    try:
        requested = tuple(chunk_sizes)
    except TypeError as exc:
        raise ValueError("chunk_sizes must be an iterable of integers") from exc
    if not requested:
        raise ValueError("chunk_sizes must contain at least one size")
    if any(not isinstance(size, numbers.Integral) or isinstance(size, bool) for size in requested):
        raise ValueError(f"chunk_sizes must contain only integers; got {requested!r}")
    sizes = tuple(sorted({int(size) for size in requested}))
    if sizes[0] < 1:
        raise ValueError(f"chunk_sizes must contain only positive sizes; got {sizes!r}")
    if sizes[-1] > max_chunk_size:
        raise ValueError(f"chunk_sizes={sizes!r} exceeds MAX_CHUNK_SIZE={max_chunk_size}")
    return sizes


def to_float32(arr):
    """Return ``arr`` as float32, avoiding a copy when already float32."""
    return arr.astype(mx.float32) if arr.dtype != mx.float32 else arr


def flatten_batch(x):
    """Flatten ``(*batch, n_samples)`` to ``(n_channels, n_samples)``.

    Returns ``(x_flat, batch_shape, n_channels, n_samples)``. ``n_channels`` is
    the product of ``batch_shape`` (1 when ``x`` is 1D). ``x_flat`` is a 2D
    view suitable for the kernels' ``ch * CS + t`` indexing.
    """
    batch_shape = tuple(x.shape[:-1])
    n_samples = x.shape[-1]
    n_channels = 1
    for d in batch_shape:
        n_channels *= d
    x_flat = x.reshape(n_channels, n_samples) if batch_shape else x.reshape(1, n_samples)
    return x_flat, batch_shape, n_channels, n_samples


def restore_batch(y_combined, batch_shape, n_samples):
    """Inverse of :func:`flatten_batch` on the time axis only."""
    if batch_shape:
        return y_combined.reshape(*batch_shape, n_samples)
    return y_combined.reshape(n_samples)


def chunked_scan(x_flat, n_samples, chunk_sizes, state, launch_fn):
    """Drive a chunked recurrence kernel over the time axis.

    ``chunk_sizes`` is a sorted tuple of allowable compile-time kernel sizes.
    Each launch uses the smallest allowable size that fits all remaining
    samples, or the largest size when multiple launches are required. A short
    chunk is zero-padded to the selected size and its runtime ``valid_length``
    is passed as a one-item uint32 array. This bounds the set of Metal kernel
    specializations while avoiding unnecessary launches.

    The caller closes over any extra kernel inputs (coefficients, sizes) inside
    ``launch_fn``. Kernels must emit their state at ``valid_length - 1`` rather
    than after the padding.

    Returns ``(y_combined, final_state)``.
    """
    y_chunks = []
    start = 0
    max_chunk_size = chunk_sizes[-1]
    while start < n_samples:
        remaining = n_samples - start
        chunk_size = next((size for size in chunk_sizes if size >= remaining), max_chunk_size)
        valid = min(remaining, chunk_size)
        end = start + valid
        x_chunk = x_flat[:, start:end]
        if valid < chunk_size:
            x_chunk = mx.pad(x_chunk, [(0, 0), (0, chunk_size - valid)])
        valid_length = mx.array([valid], dtype=mx.uint32)
        y_chunk, state = launch_fn(x_chunk, state, chunk_size, valid_length)
        y_chunks.append(y_chunk[:, :valid])
        start = end
    y_combined = y_chunks[0] if len(y_chunks) == 1 else mx.concatenate(y_chunks, axis=-1)
    return y_combined, state
