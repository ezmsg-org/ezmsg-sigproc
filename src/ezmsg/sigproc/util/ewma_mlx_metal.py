"""EWMA filtering on Apple Silicon via MLX + Metal."""

import mlx.core as mx

from .mlx_metal_common import chunked_scan, flatten_batch, normalize_chunk_sizes, restore_batch, to_float32

MAX_CHUNK_SIZE = 1024


def ewma_mlx_metal(x, alpha: float, zi, chunk_size: int = MAX_CHUNK_SIZE, *, chunk_sizes=None):
    """Apply ``y[n] = alpha*x[n] + (1-alpha)*y[n-1]`` on-device.

    Args:
        x: MLX array with time on the last axis.
        alpha: EWMA update coefficient.
        zi: Initial scipy-style filter state, i.e. ``(1-alpha) * y[-1]``,
            shaped like ``x.shape[:-1] + (1,)``.
        chunk_size: Maximum samples per Metal launch. Ignored when
            ``chunk_sizes`` is provided. Retained for backward compatibility.
        chunk_sizes: Allowable compile-time kernel sizes. The smallest size
            that fits the remaining samples is selected for each launch, and
            each specialization is compiled lazily and cached by MLX.

    Returns:
        ``(y, zf)`` where ``zf`` has the same shape/convention as ``zi``.
    """
    allowed_chunk_sizes = normalize_chunk_sizes(
        (chunk_size,) if chunk_sizes is None else chunk_sizes,
        MAX_CHUNK_SIZE,
    )
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dimension; got {x.ndim}")
    if zi.shape != tuple(x.shape[:-1]) + (1,):
        raise ValueError(f"zi shape {tuple(zi.shape)} does not match expected {tuple(x.shape[:-1]) + (1,)}")

    x_f32 = to_float32(x)
    zi_f32 = to_float32(zi)
    coef = mx.array([float(alpha), float(1.0 - alpha)], dtype=mx.float32)

    x_flat, batch_shape, n_channels, n_samples = flatten_batch(x_f32)
    zi_flat = zi_f32.reshape(n_channels)

    def launch(x_chunk, state, cs, valid_length):
        return _launch_kernel(x_chunk, coef, state, n_channels, cs, valid_length)

    y_combined, zi_flat = chunked_scan(x_flat, n_samples, allowed_chunk_sizes, zi_flat, launch)

    y_out = restore_batch(y_combined, batch_shape, n_samples)
    zf_out = zi_flat.reshape(*batch_shape, 1) if batch_shape else zi_flat.reshape(1)
    return y_out, zf_out


_KERNEL_SOURCE = r"""
    uint t  = thread_position_in_threadgroup.x;
    uint ch = threadgroup_position_in_grid.x;
    uint valid = valid_length[0];

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
    if (t < valid) {
        y_out[ch * CS + t] = alpha * x_val + z_prev;
    }

    if (t == valid - 1) {
        zf_out[ch] = s_v[src * CS + t];
    }
"""


_kernel = mx.fast.metal_kernel(
    name="ewma_first_order",
    input_names=["x_in", "coef", "zi_in", "valid_length"],
    output_names=["y_out", "zf_out"],
    source=_KERNEL_SOURCE,
)


def _launch_kernel(x_chunk, coef, zi_flat, n_channels: int, cs: int, valid_length):
    y_chunk, zf = _kernel(
        inputs=[x_chunk, coef, zi_flat, valid_length],
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
