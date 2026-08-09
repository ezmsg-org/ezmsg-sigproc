"""Core IIR/FIR filtering infrastructure with BA and SOS coefficient support."""

import dataclasses
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import scipy.signal
from array_api_compat import get_namespace, is_numpy_array
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    SettingsType,
    TransformerType,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.messages.util import replace
from scipy.fft import next_fast_len as _next_fast_len

from .util import sosfilt_direct
from .util.array import array_device, xp_asarray, xp_create
from .util.threaded_filt import DEFAULT_MIN_BYTES as _DEFAULT_THREAD_MIN_BYTES
from .util.threaded_filt import filt_threaded, should_thread

try:
    import mlx.core as _mx

    from .util.sosfilt_mlx_metal import sos_float32_stable as _sos_float32_stable
    from .util.sosfilt_mlx_metal import sosfilt_mlx_metal as _sosfilt_mlx_metal_fn

    _HAS_MLX_METAL = True
except ImportError:
    _HAS_MLX_METAL = False


_MISSING = object()


def _changed_settings_fields(old_settings, new_settings) -> set[str]:
    """Names of settings fields that differ between two settings instances.

    Equivalent to ``ezmsg.baseproc``'s private helper, but tolerant of
    array-valued fields (e.g. ``cutoff`` as a sequence of band edges), whose
    ``!=`` yields an array rather than a bool.
    """
    changed = set()
    for settings_field in dataclasses.fields(new_settings):
        old_value = getattr(old_settings, settings_field.name, _MISSING)
        new_value = getattr(new_settings, settings_field.name)
        try:
            differs = bool(old_value != new_value)
        except (TypeError, ValueError):
            differs = not np.array_equal(old_value, new_value)
        if differs:
            changed.add(settings_field.name)
    return changed


@dataclass
class FilterCoefficients:
    b: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    a: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))


# Type aliases
BACoeffs = tuple[npt.NDArray, npt.NDArray]
SOSCoeffs = npt.NDArray
FilterCoefsType = typing.TypeVar("FilterCoefsType", BACoeffs, SOSCoeffs)


def _normalize_coefs(
    coefs: FilterCoefficients | tuple[npt.NDArray, npt.NDArray] | npt.NDArray | None,
) -> tuple[str, tuple[npt.NDArray, ...] | None]:
    coef_type = "ba"
    if coefs is not None:
        # scipy.signal functions called with first arg `*coefs`.
        # Make sure we have a tuple of coefficients.
        if isinstance(coefs, np.ndarray):
            coef_type = "sos"
            coefs = (coefs,)  # sos funcs just want a single ndarray.
        elif isinstance(coefs, FilterCoefficients):
            coefs = (coefs.b, coefs.a)
        elif not isinstance(coefs, tuple):
            coefs = (coefs,)
    return coef_type, coefs


#: Promoted dtypes for which the dedicated numpy FIR paths reproduce
#: ``lfilter``'s semantics. ``ndimage.correlate1d`` demotes its output to the
#: input dtype and ``rfft`` rejects complex outright, so anything else -- complex,
#: integer, extended precision -- goes to scipy rather than being mis-handled.
_FIR_NUMPY_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _is_fir(b: npt.NDArray, a: npt.NDArray) -> bool:
    """Whether BA coefficients describe a filter with no feedback."""
    return len(a) == 1 or bool(np.allclose(a[1:], 0))


def _fir_taps(b: npt.NDArray, a: npt.NDArray, data: typing.Any) -> tuple[npt.NDArray, typing.Any] | None:
    """Normalized FIR taps and the dtype to filter in, or None to use scipy.

    ``scipy.signal.lfilter`` divides through by ``a[0]`` and promotes to
    ``result_type(b, a, x)``; the dedicated paths are only a valid substitute if
    they do the same. Note ``a`` participates in the promotion even when it is
    just ``[1.0]``: float32 taps against the default float64 ``a`` still filter
    in float64 (verified against ``lfilter`` over all eight float32/float64
    combinations of b, a and x).

    Returns ``None`` for inputs the dedicated paths would silently mis-handle --
    ``a[0] == 0``, or a promotion outside :data:`_FIR_NUMPY_DTYPES` -- so the
    caller falls through to ``lfilter``, whose behavior is the reference.
    """
    a = np.asarray(a).reshape(-1)
    if a.size == 0:
        return None
    a0 = a[0]
    if a0 == 0:
        # scipy warns and yields inf here; defer to it rather than inventing.
        return None
    b = np.asarray(b)
    if a0 != 1:
        b = b / a0
    if not is_numpy_array(data):
        # Device-constrained backends keep filtering in their own dtype.
        return b, data.dtype
    dtype = np.result_type(b.dtype, a.dtype, data.dtype)
    if dtype not in _FIR_NUMPY_DTYPES:
        return None
    return b, dtype


def _sosfilt_mlx_metal_xp(sos_mx, data, axis_idx, zi, chunk_sizes):
    """Apply SOS filtering via the on-device Metal kernel.

    The kernel requires time as the last axis of its input; ``zi`` is kept
    in scipy layout ``(n_sections, ..., 2, ...)`` with the "2" at
    ``axis_idx + 1`` so state shape is portable across backends. Both
    tensors are moved to the kernel's ``(..., time-last)`` layout on the
    way in and restored on the way out.
    """
    x_mx = _mx.moveaxis(data, axis_idx, -1) if axis_idx != data.ndim - 1 else data
    zi_mx = _mx.moveaxis(zi, axis_idx + 1, -1) if axis_idx + 1 != zi.ndim - 1 else zi
    y_mx, zf_mx = _sosfilt_mlx_metal_fn(sos_mx, x_mx, zi=zi_mx, chunk_sizes=chunk_sizes)
    y = _mx.moveaxis(y_mx, -1, axis_idx) if axis_idx != data.ndim - 1 else y_mx
    zf = _mx.moveaxis(zf_mx, -1, axis_idx + 1) if axis_idx + 1 != zi.ndim - 1 else zf_mx
    return y, zf


def _fir_filt_fft(b, data, zi, axis_idx, xp):
    """FIR filtering via FFT convolution with streaming state.

    Args:
        b: FIR filter taps, shape (1, ..., M+1, ..., 1) with filter length at axis_idx.
        data: Input array.
        zi: State array holding the last M input samples along axis_idx.
        axis_idx: The axis along which to filter.
        xp: Array API namespace.

    Returns:
        (filtered_data, new_zi) tuple.
    """
    M = zi.shape[axis_idx]  # filter order (num taps - 1)

    if M == 0:
        # Zero-order FIR: just scale
        return data * b, zi

    N = data.shape[axis_idx]

    # Prepend state (last M input samples from previous chunk)
    extended = xp.concat([zi, data], axis=axis_idx)

    # FFT convolution. Round the transform length up to a 5-smooth size: the
    # natural N + 2M is frequently a bad length (a large prime factor), and
    # paying for a few extra samples is far cheaper than the resulting
    # transform. Measured worst case, 17 taps on a 8192-sample chunk:
    # 56.8 ms at N + 2M = 8224 (= 2**5 * 257) vs 24.9 ms at the next fast
    # length. next_fast_len itself costs ~40 ns, so it is called inline.
    fft_len = _next_fast_len(N + 2 * M)
    B = xp.fft.rfft(b, n=fft_len, axis=axis_idx)
    X = xp.fft.rfft(extended, n=fft_len, axis=axis_idx)
    full = xp.fft.irfft(B * X, n=fft_len, axis=axis_idx)

    # Extract valid output: length N starting at offset M
    out = slice_along_axis(full, slice(M, M + N), axis_idx)

    # Update state: last M samples of extended input
    new_zi = slice_along_axis(extended, slice(N, N + M), axis_idx)

    return out, new_zi


def _fir_filt_conv1d(b_1d, data, zi, axis_idx):
    """FIR filtering via scipy.ndimage's C-level 1-D correlation (numpy only).

    The scipy.signal route for a numpy FIR is ``lfilter``, which for ``len(a) == 1``
    degrades to ``np.apply_along_axis(np.convolve, ...)`` -- a Python-level loop
    with one ``np.convolve`` call per channel. ``ndimage.correlate1d`` does the
    same arithmetic in one C call over the whole array, which is 1.9-3.0x faster
    for short filters.

    It loses to the FFT path once the filter grows (see FIR_FFT_MIN_TAPS), so
    this is the short-filter branch only.

    Args:
        b_1d: 1-D FIR taps, shape (M+1,).
        data: Input array.
        zi: State holding the last M input samples along axis_idx.
        axis_idx: The axis along which to filter.

    Returns:
        (filtered_data, new_zi) tuple.
    """
    M = zi.shape[axis_idx]

    if M == 0:
        return data * b_1d[0], zi

    N = data.shape[axis_idx]
    extended = np.concatenate([zi, data], axis=axis_idx)

    # correlate1d centers its kernel; origin shifts it. Correlating with the
    # reversed taps at origin = -(len(b) // 2) makes output index n use the
    # window starting at n, i.e. y[n] = sum_k b[k] * extended[n + M - k], which
    # is the causal convolution whose first N samples are the valid output.
    # Verified for both odd and even tap counts.
    full = scipy.ndimage.correlate1d(
        extended,
        b_1d[::-1],
        axis=axis_idx,
        mode="constant",
        origin=-(len(b_1d) // 2),
    )
    out = slice_along_axis(full, slice(0, N), axis_idx)
    new_zi = slice_along_axis(extended, slice(N, N + M), axis_idx)
    return out, new_zi


def _fir_filt_conv(b_1d, data, zi, axis_idx, xp):
    """FIR filtering via direct convolution using xp.conv_general.

    Args:
        b_1d: 1D FIR filter taps, shape (M+1,).
        data: Input array.
        zi: State array holding the last M input samples along axis_idx.
        axis_idx: The axis along which to filter.
        xp: Array API namespace (must have conv_general).

    Returns:
        (filtered_data, new_zi) tuple.
    """
    M = zi.shape[axis_idx]  # filter order (num taps - 1)

    if M == 0:
        return data * b_1d[0], zi

    N = data.shape[axis_idx]

    # Prepend state (last M input samples from previous chunk)
    extended = xp.concat([zi, data], axis=axis_idx)

    # Reshape N-D data into (batch, length, channels) for conv_general
    shape = extended.shape
    batch_size = 1
    for i in range(axis_idx):
        batch_size *= shape[i]
    chan_size = 1
    for i in range(axis_idx + 1, len(shape)):
        chan_size *= shape[i]
    L = shape[axis_idx]  # M + N

    input_3d = xp.reshape(extended, (batch_size, L, chan_size))

    # conv_general expects weight shape (out_channels, kernel_size, in_channels/groups)
    # With groups=chan_size, each channel is convolved independently.
    # We want each output channel to use the same kernel b_1d.
    # Weight shape: (chan_size, M+1, 1)
    kernel = xp.reshape(b_1d, (1, M + 1, 1))
    weight = xp.broadcast_to(kernel, (chan_size, M + 1, 1))

    # conv_general with flip=True gives correlation->convolution
    # padding=0 (default "VALID"), groups=chan_size for per-channel conv
    # Input: (batch_size, M+N, chan_size), Weight: (chan_size, M+1, 1)
    # Output: (batch_size, N, chan_size)
    out_3d = xp.conv_general(input_3d, weight, groups=chan_size, flip=True)

    # Reshape back to original data shape
    out_shape = list(data.shape)
    out_shape[axis_idx] = N
    dat_out = xp.reshape(out_3d, tuple(out_shape))

    # Update state: last M samples of extended input
    new_zi = slice_along_axis(extended, slice(N, N + M), axis_idx)

    return dat_out, new_zi


class FilterBaseSettings(ez.Settings):
    axis: str | None = None
    """The name of the axis to operate on."""

    coef_type: str = "ba"
    """The type of filter coefficients. One of "ba" or "sos"."""

    use_fast_sosfilt: bool = True
    """If True (default), numpy SOS filtering calls scipy's Cython kernel
    directly instead of going through ``scipy.signal.sosfilt``.

    Output is bit-identical -- same kernel, same dtype promotion, same order of
    operations -- so this is purely a latency optimization. It removes scipy's
    fixed ~12-18 us of per-call validation and bookkeeping, which is invisible
    on offline-sized chunks but dominant online: measured 59.9% of the call at
    16 channels x 30 samples, 51.0% at 32x30, 16.6% at 256x30, 4.1% at 256x128.

    The fast path depends on a private scipy entry point. It is verified against
    the public function once per process, and falls back automatically if the
    private kernel is missing or disagrees. Set False to force the public path."""

    use_mlx_metal: bool = True
    """If True (default), SOS filtering on MLX inputs runs on the GPU via the
    bundled Metal kernel (``sosfilt_mlx_metal``) instead of round-tripping
    through scipy. Set to False to fall back to the scipy path (bit-exact
    with numpy at the cost of CPU round-trips and ~5-8x slowdown)."""

    mlx_metal_chunk_sizes: tuple[int, ...] = (512,)
    """Allowable compile-time chunk sizes for SOS Metal kernels. The smallest
    size that fits the remaining samples is selected on each launch; otherwise
    the largest size is repeated. Specializations compile lazily on first use.
    Values must be in ``[1, 512]``."""

    thread_min_bytes: int = _DEFAULT_THREAD_MIN_BYTES
    """Chunk size (bytes) at or above which scipy IIR filtering is split across
    threads on a non-sample axis. Channels are independent recurrences, so the
    result is bit-identical to the single-threaded call.

    Only large chunks benefit: scipy's filters are single-threaded C loops, and
    below ~1 MB the dispatch cost makes threading a net loss (measured 0.23x at
    30 samples x 256 channels, 4.5x at 8192). The default keeps online-sized
    chunks single-threaded while letting offline blocks use the cores. Set to 0
    to disable threading entirely."""

    fir_fft_min_taps: int = 64
    """Tap count at or above which a numpy FIR filter is applied by FFT
    convolution instead of scipy.ndimage's C-level correlation.

    Neither wins everywhere. Measured on 256 channels (best FFT vs ndimage),
    the FFT is 1.3-6.8x faster at 129-513 taps while ndimage is 1.6-3.0x
    faster at 17-33 taps, with the crossover near 64 taps.

    The two differ in more than speed, and the difference matters if you rely
    on an offline run reproducing an online one exactly:

    * The time-domain path is **chunk-size invariant** -- filtering in chunks
      of 250, chunks of 333, or one shot gives bit-identical output.
    * The FFT path is not. The transform length depends on the chunk length,
      so a different chunking gives a different rounding: measured 2.6e-07
      relative in float32 (~7e-16 in float64).

    Both match scipy's ``lfilter`` to roundoff, so this is about reproducibility
    across chunkings rather than accuracy. Set this very high to force the
    time-domain path everywhere and keep FIR output chunk-invariant."""


class FilterSettings(FilterBaseSettings):
    coefs: FilterCoefficients | None = None
    """The pre-calculated filter coefficients."""

    # Note: coef_type = "ba" is assumed for this class.


@processor_state
class FilterState:
    zi: npt.NDArray | None = None
    fir_b: typing.Any | None = None  # reshaped taps for FFT path (broadcast shape)
    fir_b_1d: typing.Any | None = None  # 1D taps for conv path
    fir_method: str | None = None  # "conv1d", "conv", "fft", or None (scipy)
    sos_method: str | None = None  # 'mlx_metal', 'scipy_numpy', or None (scipy)
    sos_mx: typing.Any | None = None  # cached mlx.core.array of SOS coefs
    sos_direct: typing.Any | None = None  # DirectSosfilt with per-call setup hoisted
    sos_direct_dtype: typing.Any | None = None  # promoted dtype sos_direct was built for


class FilterTransformer(BaseStatefulTransformer[FilterSettings, AxisArray, AxisArray, FilterState]):
    """
    Filter data using the provided coefficients.
    """

    NONRESET_SETTINGS_FIELDS = frozenset({"mlx_metal_chunk_sizes"})

    def __call__(self, message: AxisArray) -> AxisArray:
        if self.settings.coefs is None:
            return message
        # Empty chunks can't supply the first sample that zi edge-scaling
        # needs, so defer state creation until the first non-empty chunk.
        if message.data.size == 0:
            return message
        if self._state.zi is None:
            self._reset_state(message)
            self._hash = self._hash_message(message)
        return super().__call__(message)

    def _hash_message(self, message: AxisArray) -> int:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        axis_idx = message.get_axis_idx(axis)
        samp_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        return hash((message.key, samp_shape))

    def _build_fir_taps(self, b: npt.NDArray, work_dtype: typing.Any, data: typing.Any, axis_idx: int) -> None:
        """Cache the converted taps that both dedicated FIR paths read."""
        xp = get_namespace(data)
        dev = array_device(data)
        # 1D taps for the conv paths
        self.state.fir_b_1d = xp_asarray(xp, b, dtype=work_dtype, device=dev)
        # Reshape b to broadcast: (1, ..., M+1, ..., 1) for FFT path
        b_shape = [1] * data.ndim
        b_shape[axis_idx] = len(b)
        self.state.fir_b = xp.reshape(self.state.fir_b_1d, tuple(b_shape))

    def _refresh_fir_taps(self, message: AxisArray, axis_idx: int) -> None:
        """Rebuild the tap caches that ``update_coefficients`` dropped.

        A same-length coefficient swap deliberately keeps ``zi``, so the taps
        cannot be rebuilt there -- the dtype and device only become known when a
        message arrives. If the new coefficients no longer route here at all
        (complex taps, ``a[0] == 0``, a changed order), ``zi`` would mean
        something different on the path they do route to, so start over instead.
        """
        _, coefs = _normalize_coefs(self.settings.coefs)
        b, a = coefs
        fir = _fir_taps(b, a, message.data) if _is_fir(b, a) else None
        if fir is None or len(fir[0]) - 1 != self.state.zi.shape[axis_idx]:
            self._reset_state(message)
        else:
            self._build_fir_taps(fir[0], fir[1], message.data, axis_idx)

    def _reset_state(self, message: AxisArray) -> None:
        # __call__ guarantees a non-empty message here. Initial conditions are
        # edge-scaled by the first sample x0 -- the scipy ``lfilter_zi * x[0]``
        # idiom -- treating the pre-stream signal as constant x0 so that a DC
        # offset does not ring through as a start-up transient.
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        axis_idx = message.get_axis_idx(axis)
        n_tail = message.data.ndim - axis_idx - 1
        _, coefs = _normalize_coefs(self.settings.coefs)

        first_idx = tuple(slice(0, 1) if i == axis_idx else slice(None) for i in range(message.data.ndim))

        if self.settings.coef_type == "ba":
            b, a = coefs
            fir = _fir_taps(b, a, message.data) if _is_fir(b, a) else None

            if fir is not None:
                # Dedicated FIR paths. scipy's lfilter degrades to a per-channel
                # Python loop when len(a) == 1, so numpy comes here too rather
                # than falling through to the generic scipy branch below. Note
                # these paths define zi as the last M *input* samples, unlike
                # scipy's lfilter state, which is why the setup differs.
                b_fir, work_dtype = fir
                xp = get_namespace(message.data)
                dev = array_device(message.data)
                M = len(b_fir) - 1  # filter order
                zi_shape = list(message.data.shape)
                zi_shape[axis_idx] = M
                # zi holds input samples, but in the promoted dtype: it is
                # concatenated with the incoming chunk, so it is what carries
                # result_type(b, x) through to the output.
                zeros = xp_create(xp.zeros, tuple(zi_shape), dtype=work_dtype, device=dev)
                self.state.zi = zeros + message.data[first_idx]
                self._build_fir_taps(b_fir, work_dtype, message.data, axis_idx)
                # Choose method. Non-numpy backends prefer their own fused
                # convolution when they have one. For numpy the choice is
                # tap-count driven: ndimage's C loop wins for short filters,
                # the FFT wins once the filter grows (see FIR_FFT_MIN_TAPS).
                if is_numpy_array(message.data):
                    self.state.fir_method = "fft" if len(b_fir) >= self.settings.fir_fft_min_taps else "conv1d"
                else:
                    self.state.fir_method = "conv" if hasattr(xp, "conv_general") else "fft"
                self.state.sos_method = None
                self.state.sos_mx = None
                return

            if max(len(b), len(a)) < 2:
                # Single-tap filter: zero-length state; lfilter_zi needs len >= 2.
                zi = np.zeros(0)
            else:
                # Constant-unit-input steady state; valid for FIR (a=[1]) and IIR.
                zi = scipy.signal.lfilter_zi(b, a)
        else:
            # For second-order sections (SOS) filters, use sosfilt_zi
            zi = scipy.signal.sosfilt_zi(*coefs)

        zi_expand = (None,) * axis_idx + (slice(None),) + (None,) * n_tail
        n_tile = message.data.shape[:axis_idx] + (1,) + message.data.shape[axis_idx + 1 :]

        if self.settings.coef_type == "sos":
            zi_expand = (slice(None),) + zi_expand
            n_tile = (1,) + n_tile

        zi_tiled = np.tile(zi[zi_expand], n_tile)
        zi_tiled = zi_tiled * np.asarray(message.data[first_idx])  # edge-scale by x0

        self.state.fir_method = None
        self.state.fir_b = None
        self.state.fir_b_1d = None

        sos_method = None
        sos_mx = None
        is_mlx_input = not is_numpy_array(message.data) and get_namespace(message.data).__name__ == "mlx.core"

        # Route SOS on MLX inputs through the on-device Metal kernel when the
        # float32 SOS remains stable. Ultra-low high-pass filters can become
        # unstable after float32 quantization; keep those on scipy with a
        # float64 NumPy state instead of sending them to Metal.
        if (
            self.settings.coef_type == "sos"
            and self.settings.use_mlx_metal
            and _HAS_MLX_METAL
            and is_mlx_input
            and _sos_float32_stable(coefs[0])
        ):
            sos_method = "mlx_metal"
            sos_mx = _mx.array(np.asarray(coefs[0]).astype(np.float32))
        elif self.settings.coef_type == "sos" and is_mlx_input:
            sos_method = "scipy_numpy"

        if not is_numpy_array(message.data) and sos_method != "scipy_numpy":
            xp = get_namespace(message.data)
            zi_tiled = xp_asarray(xp, zi_tiled)
        self.state.zi = zi_tiled

        if sos_method == "mlx_metal":
            self.state.sos_method = "mlx_metal"
            self.state.sos_mx = sos_mx
        elif sos_method == "scipy_numpy":
            self.state.sos_method = "scipy_numpy"
            self.state.sos_mx = None
        else:
            self.state.sos_method = None
            self.state.sos_mx = None

        # Built lazily on first use, so that a coefficient update which does not
        # force a full reset still rebuilds it (see update_coefficients).
        self.state.sos_direct = None
        self.state.sos_direct_dtype = None

    def update_coefficients(
        self,
        coefs: FilterCoefficients | tuple[npt.NDArray, npt.NDArray] | npt.NDArray,
        coef_type: str | None = None,
    ) -> None:
        """
        Update filter coefficients.

        If the new coefficients have the same length as the current ones, only the coefficients are updated.
        If the lengths differ, the filter state is also reset to handle the new filter order.

        Args:
            coefs: New filter coefficients
        """
        old_coefs = self.settings.coefs

        # Update settings with new coefficients
        self.settings = replace(self.settings, coefs=coefs)
        if coef_type is not None:
            self.settings = replace(self.settings, coef_type=coef_type)

        # Check if we need to reset the state
        if self.state.zi is not None:
            reset_needed = False

            if self.settings.coef_type == "ba":
                if isinstance(old_coefs, FilterCoefficients) and isinstance(coefs, FilterCoefficients):
                    if len(old_coefs.b) != len(coefs.b) or len(old_coefs.a) != len(coefs.a):
                        reset_needed = True
                elif isinstance(old_coefs, tuple) and isinstance(coefs, tuple):
                    if len(old_coefs[0]) != len(coefs[0]) or len(old_coefs[1]) != len(coefs[1]):
                        reset_needed = True
                else:
                    reset_needed = True
                if not reset_needed:
                    # A same-length swap can still flip FIR <-> IIR (a=[1, 0] ->
                    # a=[1, -0.9]). The FIR paths keep zi as the last M *input*
                    # samples while lfilter keeps filter state, so the carried
                    # state is not transferable between them: reset.
                    _, old_ba = _normalize_coefs(old_coefs)
                    _, new_ba = _normalize_coefs(coefs)
                    if old_ba is None or _is_fir(*old_ba) != _is_fir(*new_ba):
                        reset_needed = True
            elif self.settings.coef_type == "sos":
                if isinstance(old_coefs, np.ndarray) and isinstance(coefs, np.ndarray):
                    if old_coefs.shape != coefs.shape:
                        reset_needed = True
                else:
                    reset_needed = True

            if reset_needed:
                self.state.zi = None  # This will trigger _reset_state on the next call

        # Always invalidate cached MLX SOS coefs; _reset_state or _process
        # will re-cache them from the new settings.coefs on the next call.
        self.state.sos_mx = None
        # Likewise the converted FIR taps. A same-length swap does not reset zi,
        # so without this the filter would keep running the old taps.
        # _refresh_fir_taps rebuilds them on the next chunk.
        self.state.fir_b = None
        self.state.fir_b_1d = None
        # Likewise the direct-kernel filter, which holds a converted copy of the
        # coefficients. A same-length coefficient change does not reset zi, so
        # without this it would keep filtering with the old coefficients.
        self.state.sos_direct = None
        self.state.sos_direct_dtype = None

    def _sos_direct_for(self, data_np: npt.NDArray, zi_np: npt.NDArray):
        """Cached direct-kernel SOS filter for the current stream, or None.

        Built on first use rather than in ``_reset_state`` so that a coefficient
        update which does not force a reset still picks up the new coefficients.

        The cache is keyed on ``result_type(sos, data, zi)``, because the promoted
        dtype is baked into the converted coefficients. A stream that starts in
        float32 and later receives float64 must not keep filtering in the dtype it
        began with, and one that later receives complex must not have its
        imaginary part cast away -- public ``sosfilt`` promotes in both cases.
        ``np.result_type`` costs ~0.09 us against the ~5-7 us this path saves, so
        it is checked every call rather than assumed to be stable.

        A cached ``None`` means "this dtype is not usable"; the caller falls back
        to public ``sosfilt``, which handles the dtypes the kernel does not.
        """
        if self.settings.coef_type != "sos" or not self.settings.use_fast_sosfilt:
            return None
        if not isinstance(data_np, np.ndarray) or not isinstance(zi_np, np.ndarray):
            return None
        _, coefs = _normalize_coefs(self.settings.coefs)
        dtype = np.result_type(coefs[0], data_np, zi_np)
        # NB: `dtype == None` is True for float64 -- np.dtype(None) is float64 --
        # so the sentinel has to be excluded explicitly.
        if self.state.sos_direct_dtype is not None and dtype == self.state.sos_direct_dtype:
            return self.state.sos_direct
        self.state.sos_direct_dtype = dtype
        self.state.sos_direct = None
        if dtype in sosfilt_direct.SUPPORTED_DTYPES and sosfilt_direct.available():
            try:
                self.state.sos_direct = sosfilt_direct.DirectSosfilt(coefs[0], dtype)
            except ValueError:
                # Malformed coefficients: leave the authoritative error to public
                # sosfilt rather than raising a lookalike from here.
                pass
        return self.state.sos_direct

    def _process(self, message: AxisArray) -> AxisArray:
        if message.data.size > 0:
            axis = message.dims[0] if self.settings.axis is None else self.settings.axis
            axis_idx = message.get_axis_idx(axis)
            if self.state.fir_method is not None and self.state.fir_b_1d is None:
                self._refresh_fir_taps(message, axis_idx)
            if self.state.fir_method == "conv1d":
                dat_out, self.state.zi = _fir_filt_conv1d(self.state.fir_b_1d, message.data, self.state.zi, axis_idx)
            elif self.state.fir_method == "conv":
                xp = get_namespace(message.data)
                dat_out, self.state.zi = _fir_filt_conv(self.state.fir_b_1d, message.data, self.state.zi, axis_idx, xp)
            elif self.state.fir_method == "fft":
                xp = get_namespace(message.data)
                dat_out, self.state.zi = _fir_filt_fft(self.state.fir_b, message.data, self.state.zi, axis_idx, xp)
            elif self.state.sos_method == "mlx_metal":
                if self.state.sos_mx is None:
                    _, coefs = _normalize_coefs(self.settings.coefs)
                    self.state.sos_mx = _mx.array(np.asarray(coefs[0]).astype(np.float32))
                dat_out, self.state.zi = _sosfilt_mlx_metal_xp(
                    self.state.sos_mx,
                    message.data,
                    axis_idx,
                    self.state.zi,
                    self.settings.mlx_metal_chunk_sizes,
                )
            elif self.state.sos_method == "scipy_numpy":
                data_np, zi_np = np.asarray(message.data), np.asarray(self.state.zi)
                _, coefs = _normalize_coefs(self.settings.coefs)
                if should_thread(data_np, axis_idx, self.settings.thread_min_bytes):
                    dat_out_np, self.state.zi = filt_threaded(
                        scipy.signal.sosfilt,
                        coefs,
                        data_np,
                        axis_idx,
                        zi_np,
                        zi_axis_offset=1,
                        min_bytes=self.settings.thread_min_bytes,
                    )
                else:
                    direct = self._sos_direct_for(data_np, zi_np)
                    if direct is not None:
                        dat_out_np, self.state.zi = direct.apply(data_np, axis_idx, zi_np)
                    else:
                        dat_out_np, self.state.zi = scipy.signal.sosfilt(*coefs, data_np, axis=axis_idx, zi=zi_np)
                dat_out = xp_asarray(get_namespace(message.data), dat_out_np)
            elif is_numpy_array(message.data) and self.settings.coef_type == "sos":
                _, coefs = _normalize_coefs(self.settings.coefs)
                if should_thread(message.data, axis_idx, self.settings.thread_min_bytes):
                    dat_out, self.state.zi = filt_threaded(
                        scipy.signal.sosfilt,
                        coefs,
                        message.data,
                        axis_idx,
                        self.state.zi,
                        zi_axis_offset=1,
                        min_bytes=self.settings.thread_min_bytes,
                    )
                else:
                    direct = self._sos_direct_for(message.data, self.state.zi)
                    if direct is not None:
                        dat_out, self.state.zi = direct.apply(message.data, axis_idx, self.state.zi)
                    else:
                        dat_out, self.state.zi = scipy.signal.sosfilt(
                            *coefs, message.data, axis=axis_idx, zi=self.state.zi
                        )
            else:
                _, coefs = _normalize_coefs(self.settings.coefs)
                filt_func = {"ba": scipy.signal.lfilter, "sos": scipy.signal.sosfilt}[self.settings.coef_type]
                input_xp = None if is_numpy_array(message.data) else get_namespace(message.data)
                if input_xp is not None:
                    # Convert coefs and zi to the input namespace so scipy's
                    # array_namespace sees a single backend and converts back.
                    # NOTE: scipy 1.17 bundles an array_api_compat that does
                    # not recognize MLX, so we also convert the output below.
                    # When scipy's bundled copy gains MLX support, the manual
                    # conversion will become a no-op.
                    coefs = tuple(xp_asarray(input_xp, c) for c in coefs)
                    # Non-numpy arrays are left to scipy's own dispatch;
                    # filt_threaded declines anything that is not an ndarray.
                    dat_out, self.state.zi = filt_func(*coefs, message.data, axis=axis_idx, zi=self.state.zi)
                    dat_out = xp_asarray(input_xp, dat_out)
                    self.state.zi = xp_asarray(input_xp, self.state.zi)
                else:
                    dat_out, self.state.zi = filt_threaded(
                        filt_func,
                        coefs,
                        message.data,
                        axis_idx,
                        self.state.zi,
                        zi_axis_offset=1 if self.settings.coef_type == "sos" else 0,
                        min_bytes=self.settings.thread_min_bytes,
                    )
        else:
            dat_out = message.data

        return replace(message, data=dat_out)


class Filter(BaseTransformerUnit[FilterSettings, AxisArray, AxisArray, FilterTransformer]):
    SETTINGS = FilterSettings


def filtergen(axis: str, coefs: npt.NDArray | tuple[npt.NDArray] | None, coef_type: str) -> FilterTransformer:
    """
    Filter data using the provided coefficients.

    Returns:
        :obj:`FilterTransformer`.
    """
    return FilterTransformer(FilterSettings(axis=axis, coefs=coefs, coef_type=coef_type))


@processor_state
class FilterByDesignState:
    filter: FilterTransformer | None = None
    needs_redesign: bool = False


class FilterByDesignTransformer(
    BaseStatefulTransformer[SettingsType, AxisArray, AxisArray, FilterByDesignState],
    ABC,
    typing.Generic[SettingsType, FilterCoefsType],
):
    """Abstract base class for filter design transformers."""

    @classmethod
    def get_message_type(cls, dir: str) -> type[AxisArray]:
        if dir in ("in", "out"):
            return AxisArray
        else:
            raise ValueError(f"Invalid direction: {dir}. Must be 'in' or 'out'.")

    @abstractmethod
    def get_design_function(self) -> typing.Callable[[float], FilterCoefsType | None]:
        """Return a function that takes sampling frequency and returns filter coefficients."""
        ...

    def update_settings(self, new_settings: typing.Optional[SettingsType] = None, **kwargs) -> None:
        """
        Update settings and mark that filter coefficients need to be recalculated.

        Args:
            new_settings: Complete new settings object to replace current settings
            **kwargs: Individual settings to update
        """
        old_settings = self.settings
        if new_settings is not None:
            self.settings = new_settings
        else:
            self.settings = replace(self.settings, **kwargs)

        changed = _changed_settings_fields(old_settings, self.settings)

        if self.state.filter is not None:
            if changed <= {"mlx_metal_chunk_sizes"}:
                self.state.filter.settings = replace(
                    self.state.filter.settings,
                    mlx_metal_chunk_sizes=self.settings.mlx_metal_chunk_sizes,
                )
            else:
                self.state.needs_redesign = True

    def __call__(self, message: AxisArray) -> AxisArray:
        # Offer a shortcut when there is no design function or order is 0.
        if hasattr(self.settings, "order") and not self.settings.order:
            return message
        design_fun = self.get_design_function()
        if design_fun is None:
            return message

        # Check if filter exists but needs redesign due to settings change
        if self.state.filter is not None and self.state.needs_redesign:
            axis = self.state.filter.settings.axis
            fs = 1 / message.axes[axis].gain
            coefs = design_fun(fs)

            # Convert BA to SOS if requested
            if coefs is not None and self.settings.coef_type == "sos":
                if isinstance(coefs, tuple) and len(coefs) == 2:
                    # It's BA format, convert to SOS
                    b, a = coefs
                    coefs = scipy.signal.tf2sos(b, a)

            self.state.filter.update_coefficients(coefs, coef_type=self.settings.coef_type)
            self.state.needs_redesign = False

        return super().__call__(message)

    def _hash_message(self, message: AxisArray) -> int:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        gain = message.axes[axis].gain if hasattr(message.axes[axis], "gain") else 1
        axis_idx = message.get_axis_idx(axis)
        samp_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        return hash((message.key, samp_shape, gain))

    def _reset_state(self, message: AxisArray) -> None:
        design_fun = self.get_design_function()
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        fs = 1 / message.axes[axis].gain
        coefs = design_fun(fs)

        # Convert BA to SOS if requested
        if coefs is not None and self.settings.coef_type == "sos":
            if isinstance(coefs, tuple) and len(coefs) == 2:
                # It's BA format, convert to SOS
                b, a = coefs
                coefs = scipy.signal.tf2sos(b, a)

        new_settings = FilterSettings(
            axis=axis,
            coef_type=self.settings.coef_type,
            coefs=coefs,
            use_mlx_metal=self.settings.use_mlx_metal,
            mlx_metal_chunk_sizes=self.settings.mlx_metal_chunk_sizes,
        )
        self.state.filter = FilterTransformer(settings=new_settings)
        self.state.needs_redesign = False

    def _process(self, message: AxisArray) -> AxisArray:
        # The settings-change redesign logic in __call__ above is only reached
        # by sync callers. Async callers (every BaseTransformerUnit) go
        # through _aprocess → _process, so we honor needs_redesign here too,
        # otherwise live setting updates never take effect on the filter
        # coefficients.
        if self.state.needs_redesign:
            self._reset_state(message)
        return self.state.filter(message)


class BaseFilterByDesignTransformerUnit(
    BaseTransformerUnit[SettingsType, AxisArray, AxisArray, FilterByDesignTransformer],
    typing.Generic[SettingsType, TransformerType],
): ...
