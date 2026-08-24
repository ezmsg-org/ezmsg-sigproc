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

    Chunk sizes bound the set of Metal kernel *specializations*; they do not by
    themselves bound the set of *buffer sizes*. MLX caches freed buffers in a
    multimap keyed by exact byte size and only reuses one within
    ``min(2 * size, size + 2 * page_size)`` of the request -- effectively an
    exact match above ~32 KiB -- so every distinct intermediate length becomes a
    permanent new size class in a cache whose default limit is the whole
    machine. Concatenating the *padded* chunks and trimming once therefore
    allocates on the multiple-of-chunk_size grid rather than at ``n_samples``,
    which measured 31% less cached memory over 40 distinct input lengths.

    Trimming once at the end is only correct because padding can occur on the
    final chunk alone: every earlier iteration has ``remaining > chunk_size``,
    so ``valid == chunk_size`` and the chunk is emitted whole. Were an interior
    chunk padded, its padding would sit between two runs of valid samples and
    the single trim would return garbage, so the loop asserts that invariant
    rather than trusting whoever next edits the size-selection rule.

    Returns ``(y_combined, final_state)``.
    """
    y_chunks = []
    start = 0
    max_chunk_size = chunk_sizes[-1]
    padded_len = 0
    while start < n_samples:
        remaining = n_samples - start
        chunk_size = next((size for size in chunk_sizes if size >= remaining), max_chunk_size)
        valid = min(remaining, chunk_size)
        end = start + valid
        x_chunk = x_flat[:, start:end]
        if valid < chunk_size:
            if end != n_samples:
                raise AssertionError(
                    f"chunked_scan padded an interior chunk (valid={valid}, chunk_size={chunk_size}, "
                    f"end={end}, n_samples={n_samples}); the single trim below would return padding as data."
                )
            x_chunk = mx.pad(x_chunk, [(0, 0), (0, chunk_size - valid)])
        valid_length = mx.array([valid], dtype=mx.uint32)
        y_chunk, state = launch_fn(x_chunk, state, chunk_size, valid_length)
        y_chunks.append(y_chunk)
        padded_len += chunk_size
        start = end
    y_padded = y_chunks[0] if len(y_chunks) == 1 else mx.concatenate(y_chunks, axis=-1)
    # Exact length is part of the contract: callers reshape to n_samples.
    y_combined = y_padded if padded_len == n_samples else y_padded[:, :n_samples]
    return y_combined, state
