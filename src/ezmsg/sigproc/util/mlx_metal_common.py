"""Shared scaffolding for MLX+Metal recurrence kernels.

Both :mod:`sosfilt_mlx_metal` and :mod:`ewma_mlx_metal` apply a per-channel
recurrence over time and need the same boilerplate around the kernel: float32
promotion, batch-axis flatten/restore, and a chunked launch loop that carries
the per-chunk state forward. Those pieces live here so the kernel modules can
focus on their actual Metal source and per-op state layout.
"""

import mlx.core as mx


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


def chunked_scan(x_flat, n_samples, chunk_size, state, launch_fn):
    """Drive a chunked recurrence kernel over the time axis.

    Calls ``launch_fn(x_chunk, state, cs) -> (y_chunk, new_state)`` once per
    chunk of up to ``chunk_size`` samples, threading ``state`` through. The
    caller closes over any extra kernel inputs (coefficients, sizes) inside
    ``launch_fn``.

    Returns ``(y_combined, final_state)``.
    """
    y_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        cs = end - start
        y_chunk, state = launch_fn(x_flat[:, start:end], state, cs)
        y_chunks.append(y_chunk)
    y_combined = y_chunks[0] if len(y_chunks) == 1 else mx.concatenate(y_chunks, axis=-1)
    return y_combined, state
