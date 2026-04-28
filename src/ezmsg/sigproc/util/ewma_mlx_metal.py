"""EWMA filtering on Apple Silicon via MLX + Metal."""

import mlx.core as mx

MAX_CHUNK_SIZE = 1024


def ewma_mlx_metal(x, alpha: float, zi, chunk_size: int = MAX_CHUNK_SIZE):
    """Apply ``y[n] = alpha*x[n] + (1-alpha)*y[n-1]`` on-device.

    Args:
        x: MLX array with time on the last axis.
        alpha: EWMA update coefficient.
        zi: Initial scipy-style filter state, i.e. ``(1-alpha) * y[-1]``,
            shaped like ``x.shape[:-1] + (1,)``.
        chunk_size: Maximum samples per Metal launch.

    Returns:
        ``(y, zf)`` where ``zf`` has the same shape/convention as ``zi``.
    """
    if chunk_size > MAX_CHUNK_SIZE:
        raise ValueError(f"chunk_size={chunk_size} exceeds MAX_CHUNK_SIZE={MAX_CHUNK_SIZE}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dimension; got {x.ndim}")
    if zi.shape != tuple(x.shape[:-1]) + (1,):
        raise ValueError(f"zi shape {tuple(zi.shape)} does not match expected {tuple(x.shape[:-1]) + (1,)}")

    x_f32 = x.astype(mx.float32) if x.dtype != mx.float32 else x
    zi_f32 = zi.astype(mx.float32) if zi.dtype != mx.float32 else zi
    coef = mx.array([float(alpha), float(1.0 - alpha)], dtype=mx.float32)

    batch_shape = tuple(x_f32.shape[:-1])
    n_samples = x_f32.shape[-1]
    n_channels = 1
    for dim in batch_shape:
        n_channels *= dim

    x_flat = x_f32.reshape(n_channels, n_samples) if batch_shape else x_f32.reshape(1, n_samples)
    zi_flat = zi_f32.reshape(n_channels)

    y_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        cs = end - start
        y_chunk, zi_flat = _launch_kernel(x_flat[:, start:end], coef, zi_flat, n_channels, cs)
        y_chunks.append(y_chunk)

    y_combined = y_chunks[0] if len(y_chunks) == 1 else mx.concatenate(y_chunks, axis=-1)

    if batch_shape:
        y_out = y_combined.reshape(*batch_shape, n_samples)
        zf_out = zi_flat.reshape(*batch_shape, 1)
    else:
        y_out = y_combined.reshape(n_samples)
        zf_out = zi_flat.reshape(1)

    return y_out, zf_out


_KERNEL_SOURCE = r"""
    uint t  = thread_position_in_threadgroup.x;
    uint ch = threadgroup_position_in_grid.x;

    float alpha = coef[0];
    float beta  = coef[1];

    float x_val = x_in[ch * CS + t];
    float zi = zi_in[ch];

    // Recurrence in scipy-lfilter state form:
    //   y[n]   = alpha*x[n] + z[n]
    //   z[n+1] = beta*y[n] = beta*z[n] + beta*alpha*x[n]
    float myM = beta;
    float myv = beta * alpha * x_val;
    if (t == 0) {
        myv += beta * zi;
    }

    threadgroup float s_M[2 * CS];
    threadgroup float s_v[2 * CS];

    uint src = 0;
    s_M[t] = myM;
    s_v[t] = myv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inclusive prefix scan over affine transforms z -> M*z + v.
    for (uint stride = 1; stride < CS; stride *= 2) {
        uint r = src * CS;
        uint w = (1 - src) * CS;

        if (t >= stride) {
            float sM = s_M[r + t - stride];
            float sv = s_v[r + t - stride];
            float mM = s_M[r + t];
            float mv = s_v[r + t];

            s_M[w + t] = mM * sM;
            s_v[w + t] = mM * sv + mv;
        } else {
            s_M[w + t] = s_M[r + t];
            s_v[w + t] = s_v[r + t];
        }

        src = 1 - src;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float z_prev = t == 0 ? zi : s_v[src * CS + t - 1];
    y_out[ch * CS + t] = alpha * x_val + z_prev;

    if (t == CS - 1) {
        zf_out[ch] = s_v[src * CS + t];
    }
"""


_kernel = mx.fast.metal_kernel(
    name="ewma_first_order",
    input_names=["x_in", "coef", "zi_in"],
    output_names=["y_out", "zf_out"],
    source=_KERNEL_SOURCE,
)


def _launch_kernel(x_chunk, coef, zi_flat, n_channels: int, cs: int):
    y_chunk, zf = _kernel(
        inputs=[x_chunk, coef, zi_flat],
        template=[
            ("CS", cs),
        ],
        grid=(n_channels * cs, 1, 1),
        threadgroup=(cs, 1, 1),
        output_shapes=[
            (n_channels, cs),
            (n_channels,),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    return y_chunk, zf
