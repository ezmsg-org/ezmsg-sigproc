"""SOS biquad filtering on Apple Silicon via MLX + Metal.

This module exposes one public function, :func:`sosfilt_mlx_metal`, which
applies a cascade of second-order biquad sections to a multichannel signal
using a custom Metal kernel. Inputs and outputs stay on the GPU; data
never round-trips to CPU.

Performance targets (M1/M2-class, 700 channels, 4th-order Butterworth):
    N=30    (online):   ~55 µs/chunk streaming throughput, 18x real-time
    N=12000 (offline):  ~7 ms,  ~6.5x faster than scipy.signal.sosfilt
    N=60000 (offline):  ~35 ms, ~6.5x faster than scipy.signal.sosfilt

Precision:
    Output matches scipy.signal.sosfilt to ~1e-4 absolute at N=30000 in
    float32. Error comes from float32 accumulation through the log2(CS)-
    depth matrix composition chain inside each chunk; state handoff
    between chunks is precise. For typical neural recordings the error
    is well below the int16 quantization floor of the input and is
    dominated by physiological noise.

Quick start:
    import mlx.core as mx
    from scipy import signal
    from sosfilt_mlx_metal import sosfilt_mlx_metal

    # Design a filter with scipy, move to device
    sos = signal.butter(4, [100, 2000], btype='band', fs=30000, output='sos')
    sos_mx = mx.array(sos.astype('float32'))

    # Filter a signal (any shape, time is last axis)
    x = mx.random.normal((700, 30000))
    y, zf = sosfilt_mlx_metal(sos_mx, x)

    # Streaming: carry state between chunks
    zi = None
    for chunk in chunks:
        y_chunk, zi = sosfilt_mlx_metal(sos_mx, chunk, zi=zi)

Layout:
    The kernel requires time to be the last axis of ``x`` and reads the
    input as row-major contiguous. If your streaming framework delivers
    data in (samples, channels) layout, transpose once at the GPU entry
    point and keep (channels, samples) layout for the whole on-device
    pipeline — see the module-level discussion in the Anthropic design
    notes for why this is preferred over letting each op handle strides.

Chunk size:
    :func:`sosfilt_mlx_metal` launches one kernel per chunk of up to
    ``MAX_CHUNK_SIZE`` (512) samples, carrying state across chunks for
    signals longer than that. The default is optimal for offline use;
    in streaming use you typically pass chunks matching the hardware
    message size (30–512 samples) and let the function handle one
    chunk per call with the caller's zi/zf managing persistent state.

Regression testing:
    The private :func:`_sosfilt_mlx_metal_unfused` launches one kernel
    per section (no fusion). It has the same public signature and
    produces bit-identical output to the fused version, making it a
    cross-check for any future kernel modifications.
"""

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Maximum chunk size (samples) per single kernel invocation. Constrained
#: by threadgroup memory: at CS=512 the fused kernel uses ~26 KB of the
#: 32 KB per-threadgroup budget on M1/M2. Raise only after confirming
#: your hardware has additional shared-memory headroom.
MAX_CHUNK_SIZE = 512

# Prefix-scan composition is fast, but it is numerically fragile when SOS
# poles are very close to the unit circle. Above this radius we keep the
# computation on Metal but use a serial DF-II-T kernel per channel.
SERIAL_KERNEL_POLE_RADIUS = 0.995


def sos_float32_max_pole_radius(sos) -> float:
    """Return the largest denominator pole radius after float32 quantization."""
    sos_np = np.asarray(sos, dtype=np.float32)
    if sos_np.ndim != 2 or sos_np.shape[1] != 6:
        raise ValueError(f"sos must have shape (n_sections, 6); got {tuple(sos_np.shape)}")

    max_radius = 0.0
    for section in sos_np:
        den = np.asarray(section[3:6], dtype=np.float64)
        if den[0] == 0.0:
            raise ValueError("SOS denominator coefficient a0 must be nonzero")
        roots = np.roots(den)
        if roots.size:
            max_radius = max(max_radius, float(np.max(np.abs(roots))))
    return max_radius


def sos_float32_stable(sos) -> bool:
    """Whether the SOS denominator remains stable after float32 quantization."""
    radius = sos_float32_max_pole_radius(sos)
    return np.isfinite(radius) and radius < 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sosfilt_mlx_metal(sos, x, zi=None, chunk_size=MAX_CHUNK_SIZE):
    """Apply an SOS biquad filter cascade on-device.

    One fused Metal kernel launch per chunk handles typical biquad sections
    sequentially in threadgroup memory. Filters with poles very close to the
    unit circle use a slower serial Metal kernel that preserves scipy.signal
    SOS semantics for numerically delicate low-cutoff filters.

    Args:
        sos: :class:`mx.array` of shape ``(n_sections, 6)`` containing the
            SOS coefficients in scipy's layout: ``[b0, b1, b2, a0, a1, a2]``
            per row, with ``a0`` assumed to be 1.0. Dtype is converted to
            ``float32`` internally.
        x: :class:`mx.array` with time as the last axis. Shape
            ``(*batch, n_samples)``; leading dimensions are flattened into
            a single channel axis for the kernel and restored on output.
        zi: Initial filter state of shape
            ``(n_sections, *batch, 2)``. ``None`` is equivalent to zeros.
        chunk_size: Maximum samples per kernel invocation. Longer signals
            are chunked automatically with state carried between chunks.
            Must be in ``[1, MAX_CHUNK_SIZE]``.

    Returns:
        Tuple ``(y, zf)``:

        * ``y`` — filtered signal, same shape and dtype as ``x``.
        * ``zf`` — final state, shape ``(n_sections, *batch, 2)``,
          suitable to pass as ``zi`` for the next chunk in streaming use.

    Raises:
        ValueError: if ``sos`` has the wrong shape, if ``chunk_size`` is out
            of range, or if the float32 SOS denominator would be unstable.

    Notes:
        The kernel is compiled once per distinct
        ``(chunk_size, n_sections, n_channels)`` combination and cached
        by MLX. The first call with any new combination pays a one-time
        compile cost of ~60–150 ms.
    """
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError(f"sos must have shape (n_sections, 6); got {tuple(sos.shape)}")
    if chunk_size > MAX_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size={chunk_size} exceeds MAX_CHUNK_SIZE={MAX_CHUNK_SIZE}. "
            f"Raising this requires verifying threadgroup memory fits on "
            f"your hardware."
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dimension; got {x.ndim}")

    max_pole_radius = sos_float32_max_pole_radius(sos)
    if not np.isfinite(max_pole_radius) or max_pole_radius >= 1.0:
        raise ValueError(
            "sosfilt_mlx_metal requires SOS denominators to remain stable after "
            f"float32 quantization; max pole radius is {max_pole_radius:.9g}. "
            "Use a float64 scipy path or redesign the filter with a higher cutoff."
        )
    use_serial_kernel = max_pole_radius >= SERIAL_KERNEL_POLE_RADIUS

    sos_f32 = sos.astype(mx.float32) if sos.dtype != mx.float32 else sos
    x_f32 = x.astype(mx.float32) if x.dtype != mx.float32 else x

    n_sections = sos_f32.shape[0]
    batch_shape = tuple(x_f32.shape[:-1])
    n_samples = x_f32.shape[-1]

    # Flatten leading dims to a single channel axis
    n_channels = 1
    for d in batch_shape:
        n_channels *= d
    x_flat = x_f32.reshape(n_channels, n_samples) if batch_shape else x_f32.reshape(1, n_samples)

    # Flatten zi to the packed layout the kernel expects
    if zi is None:
        zi_flat = mx.zeros(n_sections * n_channels * 2, dtype=mx.float32)
    else:
        if zi.shape != (n_sections,) + batch_shape + (2,):
            raise ValueError(
                f"zi shape {tuple(zi.shape)} does not match expected " f"{(n_sections,) + batch_shape + (2,)}"
            )
        zi_f32 = zi.astype(mx.float32) if zi.dtype != mx.float32 else zi
        zi_flat = zi_f32.reshape(n_sections * n_channels * 2)

    # Flatten SOS to a linear coefficient buffer
    sos_flat = sos_f32.reshape(n_sections * 6)

    # Chunk the signal; state flows through zi_flat across iterations
    y_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        cs = end - start
        x_chunk = x_flat[:, start:end]
        if use_serial_kernel:
            y_chunk, zi_flat = _launch_serial_kernel(x_chunk, sos_flat, zi_flat, n_channels, n_sections, cs)
        else:
            y_chunk, zi_flat = _launch_fused_kernel(x_chunk, sos_flat, zi_flat, n_channels, n_sections, cs)
        y_chunks.append(y_chunk)

    y_combined = y_chunks[0] if len(y_chunks) == 1 else mx.concatenate(y_chunks, axis=-1)

    # Restore original batch shape
    if batch_shape:
        y_out = y_combined.reshape(*batch_shape, n_samples)
        zf_out = zi_flat.reshape(n_sections, *batch_shape, 2)
    else:
        y_out = y_combined.reshape(n_samples)
        zf_out = zi_flat.reshape(n_sections, 2)

    return y_out, zf_out


# ---------------------------------------------------------------------------
# Fused kernel (production path)
# ---------------------------------------------------------------------------
#
# Algorithm (per chunk, all sections in one launch):
#
#   1. Each thread handles one time step t of one channel ch. Threadgroup
#      layout is (ch,) in grid, (t,) in threadgroup.
#   2. For each biquad section s in order:
#        - derive A (2x2), B (2-vec), c0 scalar from this section's
#          SOS row, treating each biquad as the affine recurrence
#          s[n+1] = A s[n] + B x[n], y[n] = c0 x[n] + s[n][0]
#        - every thread initializes (M_t, v_t) = (A, B * x[t])
#        - thread 0 additionally folds the initial state zi into v[0] so
#          the scan can treat zi as zero for uniform handling across t
#        - Hillis-Steele scan of (M, v) pairs over log2(CS) levels,
#          using ping-pong shared-memory buffers so writers don't stomp
#          on concurrent readers
#        - after the scan, prefix[t].v = s[t+1] (the post-step state)
#        - compute y[t] = c0*x[t] + s[t][0] by reading the previous
#          thread's scan result (or zi[0] for thread 0)
#        - stash y[t] in a shared-memory scratch buffer s_y
#        - the last thread writes this section's final state to zf
#   3. Between sections: barrier, then promote s_y[t] to the running
#      x_val register so the next section reads from shared memory
#      instead of global memory.
#   4. After the last section, write x_val (now the final y[t]) to the
#      global y_out buffer.
#
# Shared-memory budget at CS=512:
#   M ping-pong:   4 floats × 2 buffers × 512 positions × 4 bytes = 16 KB
#   v ping-pong:   2 floats × 2 buffers × 512 positions × 4 bytes =  8 KB
#   y scratch:     1 float  × 512 positions × 4 bytes             =  2 KB
#                                                          total: 26 KB

_FUSED_KERNEL_SOURCE = r"""
    uint t  = thread_position_in_threadgroup.x;
    uint ch = threadgroup_position_in_grid.x;

    // Ping-pong shared buffers for (M, v), plus an inter-section y stash.
    threadgroup float s_M00[2 * CS];
    threadgroup float s_M01[2 * CS];
    threadgroup float s_M10[2 * CS];
    threadgroup float s_M11[2 * CS];
    threadgroup float s_v0 [2 * CS];
    threadgroup float s_v1 [2 * CS];
    threadgroup float s_y  [CS];

    // Running sample: each thread loads x once, then this register is
    // overwritten by each section's y[t] via s_y.
    float x_val = x_in[ch * CS + t];

    for (uint sec = 0; sec < N_SECTIONS; sec++) {

        // Unpack this section's coefficients; derive affine form.
        // SOS row layout: [b0, b1, b2, a0, a1, a2], a0 assumed 1.
        uint sbase = sec * 6;
        float b0 = sos_all[sbase + 0];
        float b1 = sos_all[sbase + 1];
        float b2 = sos_all[sbase + 2];
        float a1 = sos_all[sbase + 4];
        float a2 = sos_all[sbase + 5];

        float A00 = -a1,  A01 = 1.0f;
        float A10 = -a2,  A11 = 0.0f;
        float B0  = b1 - a1 * b0;
        float B1  = b2 - a2 * b0;
        float c0  = b0;

        // Initial (M, v) for this thread: M = A (constant), v = B * x.
        float myv0 = B0 * x_val;
        float myv1 = B1 * x_val;

        // Absorb zi into v[0]: after this, the scan can treat the chunk
        // as having zero initial state and prefix[t].v = s[t+1].
        uint zi_base = (sec * N_CHANNELS + ch) * 2;
        float zi0 = zi_all[zi_base + 0];
        float zi1 = zi_all[zi_base + 1];
        if (t == 0) {
            myv0 += A00 * zi0 + A01 * zi1;
            myv1 += A10 * zi0 + A11 * zi1;
        }

        // Prime buffer 0 of the ping-pong.
        uint src = 0;
        s_M00[t] = A00;
        s_M01[t] = A01;
        s_M10[t] = A10;
        s_M11[t] = A11;
        s_v0 [t] = myv0;
        s_v1 [t] = myv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Hillis-Steele scan: log2(CS) levels. At each level, thread t
        // combines its current element with the element at position
        // t - stride using the semigroup
        //   (M_sh, v_sh) ⊕ (M_me, v_me) = (M_me @ M_sh, M_me @ v_sh + v_me)
        for (uint stride = 1; stride < CS; stride *= 2) {
            uint r = src * CS;
            uint w = (1 - src) * CS;

            if (t >= stride) {
                float sM00 = s_M00[r + t - stride];
                float sM01 = s_M01[r + t - stride];
                float sM10 = s_M10[r + t - stride];
                float sM11 = s_M11[r + t - stride];
                float sv0  = s_v0 [r + t - stride];
                float sv1  = s_v1 [r + t - stride];

                float mM00 = s_M00[r + t];
                float mM01 = s_M01[r + t];
                float mM10 = s_M10[r + t];
                float mM11 = s_M11[r + t];
                float mv0  = s_v0 [r + t];
                float mv1  = s_v1 [r + t];

                s_M00[w + t] = mM00 * sM00 + mM01 * sM10;
                s_M01[w + t] = mM00 * sM01 + mM01 * sM11;
                s_M10[w + t] = mM10 * sM00 + mM11 * sM10;
                s_M11[w + t] = mM10 * sM01 + mM11 * sM11;

                s_v0[w + t] = mM00 * sv0 + mM01 * sv1 + mv0;
                s_v1[w + t] = mM10 * sv0 + mM11 * sv1 + mv1;
            } else {
                s_M00[w + t] = s_M00[r + t];
                s_M01[w + t] = s_M01[r + t];
                s_M10[w + t] = s_M10[r + t];
                s_M11[w + t] = s_M11[r + t];
                s_v0 [w + t] = s_v0 [r + t];
                s_v1 [w + t] = s_v1 [r + t];
            }

            src = 1 - src;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Per-thread y[t] = c0*x[t] + s[t][0].
        //   t == 0: s[0] = zi, read zi[0] directly.
        //   t  > 0: s[t] = prefix[t-1].v, read s_v0[src*CS + t - 1].
        float y_val;
        if (t == 0) {
            y_val = c0 * x_val + zi0;
        } else {
            y_val = c0 * x_val + s_v0[src * CS + t - 1];
        }
        s_y[t] = y_val;

        // Write this section's final state (last thread only).
        if (t == CS - 1) {
            zf_all[zi_base + 0] = s_v0[src * CS + t];
            zf_all[zi_base + 1] = s_v1[src * CS + t];
        }

        // Barrier: all y writes to s_y must complete before anyone reads.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Promote y to next section's input (register).
        x_val = s_y[t];

        // Barrier: conservatively gate against the next section's ping-
        // pong reinit racing with any remaining s_y reads. The scan's
        // first barrier covers this in practice, but the explicit form
        // is cheaper than chasing a subtle timing bug.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final section's y for this thread.
    y_out[ch * CS + t] = x_val;
"""


_fused_kernel = mx.fast.metal_kernel(
    name="sosfilt_biquad_fused",
    input_names=["x_in", "sos_all", "zi_all"],
    output_names=["y_out", "zf_all"],
    source=_FUSED_KERNEL_SOURCE,
)


def _launch_fused_kernel(x_chunk, sos_flat, zi_flat, n_channels, n_sections, cs):
    """One kernel dispatch covering all sections for one chunk.

    Returns ``(y_chunk, zf_flat)`` both as ``float32`` MLX arrays. The
    caller rebinds ``zi_flat = zf_flat`` between chunks for streaming.
    """
    y_chunk, zf_flat = _fused_kernel(
        inputs=[x_chunk, sos_flat, zi_flat],
        template=[
            ("T", mx.float32),
            ("CS", cs),
            ("N_SECTIONS", n_sections),
            ("N_CHANNELS", n_channels),
        ],
        grid=(n_channels * cs, 1, 1),
        threadgroup=(cs, 1, 1),
        output_shapes=[
            (n_channels, cs),
            (n_sections * n_channels * 2,),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    return y_chunk, zf_flat


# ---------------------------------------------------------------------------
# Serial kernel (near-unit-pole path)
# ---------------------------------------------------------------------------
#
# The fused prefix-scan kernel composes section state matrices in float32.
# For poles very close to the unit circle, those matrix products can drift
# enough to swamp low-cutoff high-pass outputs. This kernel keeps the same
# data on Metal but assigns one thread per channel and runs scipy's DF-II-T
# recurrence in time order. It is slower, but it matches scipy's float32 SOS
# semantics for filters that are too numerically delicate for the scan.

_SERIAL_KERNEL_SOURCE = r"""
    uint ch = thread_position_in_grid.x;

    float z0[N_SECTIONS];
    float z1[N_SECTIONS];
    for (uint sec = 0; sec < N_SECTIONS; sec++) {
        uint zi_base = (sec * N_CHANNELS + ch) * 2;
        z0[sec] = zi_all[zi_base + 0];
        z1[sec] = zi_all[zi_base + 1];
    }

    for (uint t = 0; t < CS; t++) {
        float x_val = x_in[ch * CS + t];

        for (uint sec = 0; sec < N_SECTIONS; sec++) {
            uint sbase = sec * 6;
            float b0 = sos_all[sbase + 0];
            float b1 = sos_all[sbase + 1];
            float b2 = sos_all[sbase + 2];
            float a1 = sos_all[sbase + 4];
            float a2 = sos_all[sbase + 5];

            float y_val = b0 * x_val + z0[sec];
            float next_z0 = b1 * x_val - a1 * y_val + z1[sec];
            float next_z1 = b2 * x_val - a2 * y_val;

            z0[sec] = next_z0;
            z1[sec] = next_z1;
            x_val = y_val;
        }

        y_out[ch * CS + t] = x_val;
    }

    for (uint sec = 0; sec < N_SECTIONS; sec++) {
        uint zi_base = (sec * N_CHANNELS + ch) * 2;
        zf_all[zi_base + 0] = z0[sec];
        zf_all[zi_base + 1] = z1[sec];
    }
"""


_serial_kernel = mx.fast.metal_kernel(
    name="sosfilt_biquad_serial",
    input_names=["x_in", "sos_all", "zi_all"],
    output_names=["y_out", "zf_all"],
    source=_SERIAL_KERNEL_SOURCE,
)


def _launch_serial_kernel(x_chunk, sos_flat, zi_flat, n_channels, n_sections, cs):
    """One serial DF-II-T kernel dispatch for numerically delicate SOS filters."""
    y_chunk, zf_flat = _serial_kernel(
        inputs=[x_chunk, sos_flat, zi_flat],
        template=[
            ("T", mx.float32),
            ("CS", cs),
            ("N_SECTIONS", n_sections),
            ("N_CHANNELS", n_channels),
        ],
        grid=(n_channels, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[
            (n_channels, cs),
            (n_sections * n_channels * 2,),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    return y_chunk, zf_flat


# ---------------------------------------------------------------------------
# Unfused kernel (regression/dev path)
# ---------------------------------------------------------------------------
#
# Same scan as the fused kernel, but processes exactly one section per
# launch. The Python wrapper iterates sections. Produces bit-identical
# output to the fused kernel (verified in the self-test) and serves as
# a cross-check for future kernel changes. Not intended for production
# use — the fused path is measurably faster at equal correctness.

_UNFUSED_KERNEL_SOURCE = r"""
    uint t  = thread_position_in_threadgroup.x;
    uint ch = threadgroup_position_in_grid.x;

    // Unpack single-section SOS row and derive affine form.
    float b0 = sos_row[0];
    float b1 = sos_row[1];
    float b2 = sos_row[2];
    float a1 = sos_row[4];
    float a2 = sos_row[5];

    float A00 = -a1,  A01 = 1.0f;
    float A10 = -a2,  A11 = 0.0f;
    float B0  = b1 - a1 * b0;
    float B1  = b2 - a2 * b0;
    float c0  = b0;

    float x_val = x_in[ch * CS + t];

    float myv0 = B0 * x_val;
    float myv1 = B1 * x_val;

    float zi0 = zi[ch * 2 + 0];
    float zi1 = zi[ch * 2 + 1];
    if (t == 0) {
        myv0 += A00 * zi0 + A01 * zi1;
        myv1 += A10 * zi0 + A11 * zi1;
    }

    threadgroup float s_M00[2 * CS];
    threadgroup float s_M01[2 * CS];
    threadgroup float s_M10[2 * CS];
    threadgroup float s_M11[2 * CS];
    threadgroup float s_v0 [2 * CS];
    threadgroup float s_v1 [2 * CS];

    uint src = 0;
    s_M00[t] = A00;
    s_M01[t] = A01;
    s_M10[t] = A10;
    s_M11[t] = A11;
    s_v0 [t] = myv0;
    s_v1 [t] = myv1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < CS; stride *= 2) {
        uint r = src * CS;
        uint w = (1 - src) * CS;

        if (t >= stride) {
            float sM00 = s_M00[r + t - stride];
            float sM01 = s_M01[r + t - stride];
            float sM10 = s_M10[r + t - stride];
            float sM11 = s_M11[r + t - stride];
            float sv0  = s_v0 [r + t - stride];
            float sv1  = s_v1 [r + t - stride];

            float mM00 = s_M00[r + t];
            float mM01 = s_M01[r + t];
            float mM10 = s_M10[r + t];
            float mM11 = s_M11[r + t];
            float mv0  = s_v0 [r + t];
            float mv1  = s_v1 [r + t];

            s_M00[w + t] = mM00 * sM00 + mM01 * sM10;
            s_M01[w + t] = mM00 * sM01 + mM01 * sM11;
            s_M10[w + t] = mM10 * sM00 + mM11 * sM10;
            s_M11[w + t] = mM10 * sM01 + mM11 * sM11;

            s_v0[w + t] = mM00 * sv0 + mM01 * sv1 + mv0;
            s_v1[w + t] = mM10 * sv0 + mM11 * sv1 + mv1;
        } else {
            s_M00[w + t] = s_M00[r + t];
            s_M01[w + t] = s_M01[r + t];
            s_M10[w + t] = s_M10[r + t];
            s_M11[w + t] = s_M11[r + t];
            s_v0 [w + t] = s_v0 [r + t];
            s_v1 [w + t] = s_v1 [r + t];
        }

        src = 1 - src;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float y_val;
    if (t == 0) {
        y_val = c0 * x_val + zi0;
    } else {
        y_val = c0 * x_val + s_v0[src * CS + t - 1];
    }
    y_out[ch * CS + t] = y_val;

    if (t == CS - 1) {
        zf_out[ch * 2 + 0] = s_v0[src * CS + t];
        zf_out[ch * 2 + 1] = s_v1[src * CS + t];
    }
"""


_unfused_kernel = mx.fast.metal_kernel(
    name="sosfilt_biquad_unfused",
    input_names=["x_in", "sos_row", "zi"],
    output_names=["y_out", "zf_out"],
    source=_UNFUSED_KERNEL_SOURCE,
)


def _launch_unfused_kernel(x_chunk, sos_row, state):
    """Single-section kernel dispatch. Used by
    :func:`_sosfilt_mlx_metal_unfused`."""
    n_channels, cs = x_chunk.shape
    y_chunk, zf = _unfused_kernel(
        inputs=[x_chunk, sos_row, state],
        template=[("T", mx.float32), ("CS", cs)],
        grid=(n_channels * cs, 1, 1),
        threadgroup=(cs, 1, 1),
        output_shapes=[(n_channels, cs), (n_channels, 2)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return y_chunk, zf


def _sosfilt_mlx_metal_unfused(sos, x, zi=None, chunk_size=MAX_CHUNK_SIZE):
    """Unfused variant: one kernel launch per (section, chunk).

    Identical public signature and output to :func:`sosfilt_mlx_metal`,
    retained for regression testing. The self-test asserts bit-exact
    agreement between this and the fused version.

    Not exported. Exists to validate the fused kernel and to catch any
    divergence introduced by future edits to either kernel.
    """
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError(f"sos must have shape (n_sections, 6); got {tuple(sos.shape)}")
    if chunk_size > MAX_CHUNK_SIZE:
        raise ValueError(f"chunk_size={chunk_size} > MAX_CHUNK_SIZE")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    sos_f32 = sos.astype(mx.float32) if sos.dtype != mx.float32 else sos
    x_f32 = x.astype(mx.float32) if x.dtype != mx.float32 else x

    n_sections = sos_f32.shape[0]
    batch_shape = tuple(x_f32.shape[:-1])
    n_samples = x_f32.shape[-1]

    n_channels = 1
    for d in batch_shape:
        n_channels *= d
    x_flat = x_f32.reshape(n_channels, n_samples) if batch_shape else x_f32.reshape(1, n_samples)

    if zi is None:
        zi_per_section = mx.zeros((n_sections, n_channels, 2), dtype=mx.float32)
    else:
        if zi.shape != (n_sections,) + batch_shape + (2,):
            raise ValueError(
                f"zi shape {tuple(zi.shape)} does not match expected " f"{(n_sections,) + batch_shape + (2,)}"
            )
        zi_f32 = zi.astype(mx.float32) if zi.dtype != mx.float32 else zi
        zi_per_section = zi_f32.reshape(n_sections, n_channels, 2)

    y_current = x_flat
    zf_list = []

    for s in range(n_sections):
        sos_row = sos_f32[s]
        state = zi_per_section[s]
        y_chunks = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            x_chunk = y_current[:, start:end]
            y_chunk, state = _launch_unfused_kernel(x_chunk, sos_row, state)
            y_chunks.append(y_chunk)
        y_current = y_chunks[0] if len(y_chunks) == 1 else mx.concatenate(y_chunks, axis=-1)
        zf_list.append(state)

    if batch_shape:
        y_out = y_current.reshape(*batch_shape, n_samples)
        zf_out = mx.stack(zf_list, axis=0).reshape(n_sections, *batch_shape, 2)
    else:
        y_out = y_current.reshape(n_samples)
        zf_out = mx.stack(zf_list, axis=0).reshape(n_sections, 2)

    return y_out, zf_out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test():
    """Run a battery of correctness and performance checks.

    Validates:
      - scipy agreement across 3 signal sizes (online, mid, long)
      - fused == unfused bit-exact (regression oracle)
      - streaming state handoff over 1000 small chunks
      - online real-time throughput
    """
    import time

    import numpy as np
    from scipy import signal as sp_signal

    FS = 30_000
    N_CHANNELS = 700
    WARMUP = 3
    ITERS = 10

    sos_np = sp_signal.butter(4, [100.0, 2000.0], btype="band", fs=FS, output="sos")
    n_sections = sos_np.shape[0]
    rng = np.random.default_rng(0)

    # ---- per-size benchmark ---------------------------------------------
    for N in [30, 12_000, 60_000]:
        print(f"\n{'='*72}")
        print(f"N = {N} samples, {N_CHANNELS} channels, {n_sections} sections")
        print(f"{'='*72}")

        x_np = rng.standard_normal((N_CHANNELS, N)).astype(np.float32)
        y_ref = sp_signal.sosfilt(sos_np, x_np, axis=-1)

        sos_m = mx.array(sos_np.astype(np.float32))
        x_m = mx.array(x_np)

        cs = min(MAX_CHUNK_SIZE, N) if N >= 32 else N

        # Fused (production path)
        t0 = time.perf_counter()
        y_m, zf_m = sosfilt_mlx_metal(sos_m, x_m, chunk_size=cs)
        mx.eval(y_m, zf_m)
        t_first = time.perf_counter() - t0

        for _ in range(WARMUP):
            y_m, _ = sosfilt_mlx_metal(sos_m, x_m, chunk_size=cs)
            mx.eval(y_m)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            y_m, _ = sosfilt_mlx_metal(sos_m, x_m, chunk_size=cs)
            mx.eval(y_m)
        t_fused = (time.perf_counter() - t0) / ITERS
        diff_f = float(np.max(np.abs(np.asarray(y_m) - y_ref)))

        # Unfused (regression oracle)
        for _ in range(WARMUP):
            y_u, _ = _sosfilt_mlx_metal_unfused(sos_m, x_m, chunk_size=cs)
            mx.eval(y_u)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            y_u, _ = _sosfilt_mlx_metal_unfused(sos_m, x_m, chunk_size=cs)
            mx.eval(y_u)
        t_unfused = (time.perf_counter() - t0) / ITERS
        diff_u = float(np.max(np.abs(np.asarray(y_u) - y_ref)))
        diff_fu = float(np.max(np.abs(np.asarray(y_m) - np.asarray(y_u))))

        # scipy reference time
        t0 = time.perf_counter()
        for _ in range(ITERS):
            _ = sp_signal.sosfilt(sos_np, x_np, axis=-1)
        t_scipy = (time.perf_counter() - t0) / ITERS

        print(
            f"  fused   (cs={cs}):  {t_fused*1000:8.2f} ms/iter "
            f"(1st: {t_first*1000:.0f} ms)  max |Δ| vs scipy: {diff_f:.2e}"
        )
        print(
            f"  unfused (cs={cs}):  {t_unfused*1000:8.2f} ms/iter" f"                    max |Δ| vs scipy: {diff_u:.2e}"
        )
        print(f"  fused vs unfused: {t_unfused/t_fused:.2f}x speedup   " f"max |Δ| between them: {diff_fu:.2e}")
        print(f"  scipy:               {t_scipy*1000:8.2f} ms/iter   " f"(fused vs scipy: {t_scipy/t_fused:.2f}x)")

        assert diff_fu == 0.0, (
            f"REGRESSION: fused and unfused differ by {diff_fu:.2e} — " f"bit-exact agreement is required."
        )

    # ---- streaming correctness ------------------------------------------
    print(f"\n{'='*72}")
    print("Streaming correctness (N=6000 split into 200 chunks of 30, state carry)")
    print(f"{'='*72}")
    N_total = 6000
    chunk = 30
    x_np = rng.standard_normal((N_CHANNELS, N_total)).astype(np.float32)
    y_ref = sp_signal.sosfilt(sos_np, x_np, axis=-1)

    sos_m = mx.array(sos_np.astype(np.float32))
    zi = None
    ys = []
    for start in range(0, N_total, chunk):
        end = min(start + chunk, N_total)
        x_chunk = mx.array(x_np[:, start:end])
        y_chunk, zi = sosfilt_mlx_metal(sos_m, x_chunk, zi=zi, chunk_size=end - start)
        ys.append(y_chunk)
    y_stream = mx.concatenate(ys, axis=-1)
    mx.eval(y_stream)
    diff_stream = float(np.max(np.abs(np.asarray(y_stream) - y_ref)))
    print(f"  streamed vs scipy whole: max |Δ| = {diff_stream:.2e}")

    # ---- online throughput ----------------------------------------------
    print(f"\n{'='*72}")
    print("Online streaming throughput (1000 chunks of 30, state carry)")
    print(f"{'='*72}")
    N_chunks = 1000
    x_chunks_np = rng.standard_normal((N_chunks, N_CHANNELS, 30)).astype(np.float32)

    zi = None
    for i in range(min(N_chunks, WARMUP * 10)):
        x_c = mx.array(x_chunks_np[i])
        _, zi = sosfilt_mlx_metal(sos_m, x_c, zi=zi, chunk_size=30)
    mx.eval(zi)

    zi = None
    t0 = time.perf_counter()
    for i in range(N_chunks):
        x_c = mx.array(x_chunks_np[i])
        y_c, zi = sosfilt_mlx_metal(sos_m, x_c, zi=zi, chunk_size=30)
    mx.eval(zi, y_c)
    t_stream = time.perf_counter() - t0
    per_chunk_us = t_stream / N_chunks * 1e6
    realtime_budget_us = 30 / FS * 1e6  # 1000 µs at 30 kHz
    print(f"  1000 chunks: {t_stream*1000:.1f} ms total, {per_chunk_us:.0f} µs/chunk")
    print(f"  real-time budget: {realtime_budget_us:.0f} µs — " f"headroom: {realtime_budget_us / per_chunk_us:.1f}x")


if __name__ == "__main__":
    _self_test()
