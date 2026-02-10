import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import scipy.signal
from array_api_compat import get_namespace, is_numpy_array
from ezmsg.baseproc import (
    BaseConsumerUnit,
    BaseStatefulTransformer,
    BaseTransformerUnit,
    SettingsType,
    TransformerType,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.messages.util import replace

from .util.array import array_device, xp_asarray, xp_create


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


def _sosfilt_xp(sos, x, axis_idx, zi, xp):
    """SOS filtering via parallel prefix scan (direct-form II transposed).

    Solves the IIR linear recurrence z[n+1] = A @ z[n] + B * x[n] using a
    Hillis-Steele inclusive prefix scan in O(log N) sequential steps instead
    of O(N), minimizing Python-level loop overhead for lazy-evaluation
    backends like MLX.

    Args:
        sos: (n_sections, 6) SOS coefficient array. Each row is [b0, b1, b2, a0, a1, a2].
            a0 is assumed to be 1.0 (standard for scipy.signal.butter output).
        x: Input data array.
        axis_idx: The axis along which to filter.
        zi: Initial conditions, shape (n_sections, *x.shape[:axis_idx], 2, *x.shape[axis_idx+1:]).
        xp: Array API namespace.

    Returns:
        (y, zf) tuple — filtered output and final filter state.
    """
    n_sections = sos.shape[0]
    N = x.shape[axis_idx]

    # Move time to axis 0 for uniform batch handling.
    x = xp.moveaxis(x, axis_idx, 0)  # (N, *batch)
    zi = xp.moveaxis(zi, axis_idx + 1, 1)  # (n_sections, 2, *batch)

    # Flatten batch dims into one.
    batch_shape = x.shape[1:]
    batch_size = 1
    for s in batch_shape:
        batch_size *= s
    x = xp.reshape(x, (N, batch_size))  # (N, B)
    zi = xp.reshape(zi, (n_sections, 2, batch_size))  # (S, 2, B)

    # Pre-allocate output zi.
    zi_out = xp.zeros((n_sections, 2, batch_size), dtype=x.dtype)

    for s in range(n_sections):
        _b0 = float(sos[s, 0])
        _b1 = float(sos[s, 1])
        _b2 = float(sos[s, 2])
        _a1 = float(sos[s, 4])
        _a2 = float(sos[s, 5])

        z_init = zi[s]  # (2, B)

        # State recurrence: z[n+1] = A @ z[n] + B_vec * x[n]
        #   A = [[-a1, 1], [-a2, 0]]
        #   B_vec = [b1 - a1*b0, b2 - a2*b0]
        # Output: y[n] = b0 * x[n] + z[n][0]
        A_mat = xp_asarray(xp, np.array([[-_a1, 1.0], [-_a2, 0.0]]))  # (2, 2)
        B_vec = xp_asarray(xp, np.array([_b1 - _a1 * _b0, _b2 - _a2 * _b0]))  # (2,)

        # Initialize scan elements:
        #   A_scan[n] = A for all n
        #   c_scan[n] = B_vec * x[n]
        A_scan = xp.zeros((N, 2, 2), dtype=A_mat.dtype)
        A_scan[:] = A_mat  # broadcast A_mat into every row
        c_scan = B_vec[None, :, None] * x[:, None, :]  # (N, 2, B)

        # Hillis-Steele inclusive prefix scan.
        # Operator: (A_r, c_r) ∘ (A_l, c_l) = (A_r @ A_l, A_r @ c_l + c_r)
        # After the scan, A_scan[n] = A^(n+1) and
        # c_scan[n] = Σ_{k=0..n} A^(n-k) @ B_vec * x[k].
        stride = 1
        while stride < N:
            right_A = A_scan[stride:]  # (N-stride, 2, 2)
            left_A = A_scan[:-stride]  # (N-stride, 2, 2)
            right_c = c_scan[stride:]  # (N-stride, 2, B)
            left_c = c_scan[:-stride]  # (N-stride, 2, B)

            A_scan[stride:] = right_A @ left_A
            c_scan[stride:] = right_A @ left_c + right_c
            stride *= 2

        # Recover all states: z[n+1] = A_scan[n] @ z_init + c_scan[n]
        z_from_scan = A_scan @ z_init[None, :, :] + c_scan  # (N, 2, B)

        # z[0..N-1] for output: prepend z_init, drop z[N].
        z_needed = xp.zeros((N, 2, batch_size), dtype=x.dtype)
        z_needed[0] = z_init
        z_needed[1:] = z_from_scan[:-1]

        # y[n] = b0 * x[n] + z[n][0]; output becomes input for the next section.
        x = _b0 * x + z_needed[:, 0, :]  # (N, B)

        # Final state for this section: z[N]
        zi_out[s] = z_from_scan[-1]

    # Restore shapes.
    x = xp.reshape(x, (N,) + batch_shape)
    zi_out = xp.reshape(zi_out, (n_sections, 2) + batch_shape)
    x = xp.moveaxis(x, 0, axis_idx)
    zi_out = xp.moveaxis(zi_out, 1, axis_idx + 1)
    return x, zi_out


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

    # FFT convolution
    fft_len = N + 2 * M
    B = xp.fft.rfft(b, n=fft_len, axis=axis_idx)
    X = xp.fft.rfft(extended, n=fft_len, axis=axis_idx)
    full = xp.fft.irfft(B * X, n=fft_len, axis=axis_idx)

    # Extract valid output: length N starting at offset M
    out = slice_along_axis(full, slice(M, M + N), axis_idx)

    # Update state: last M samples of extended input
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


class FilterSettings(FilterBaseSettings):
    coefs: FilterCoefficients | None = None
    """The pre-calculated filter coefficients."""

    # Note: coef_type = "ba" is assumed for this class.


@processor_state
class FilterState:
    zi: npt.NDArray | None = None
    fir_b: typing.Any | None = None  # reshaped taps for FFT path (broadcast shape)
    fir_b_1d: typing.Any | None = None  # 1D taps for conv path
    fir_method: str | None = None  # 'conv', 'fft', or None (scipy)


class FilterTransformer(BaseStatefulTransformer[FilterSettings, AxisArray, AxisArray, FilterState]):
    """
    Filter data using the provided coefficients.
    """

    def __call__(self, message: AxisArray) -> AxisArray:
        if self.settings.coefs is None:
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

    def _reset_state(self, message: AxisArray) -> None:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        axis_idx = message.get_axis_idx(axis)
        n_tail = message.data.ndim - axis_idx - 1
        _, coefs = _normalize_coefs(self.settings.coefs)

        if self.settings.coef_type == "ba":
            b, a = coefs
            is_fir = len(a) == 1 or np.allclose(a[1:], 0)

            if is_fir and not is_numpy_array(message.data):
                # FIR + non-numpy: use conv_general if available, else FFT
                xp = get_namespace(message.data)
                dev = array_device(message.data)
                M = len(b) - 1  # filter order
                zi_shape = list(message.data.shape)
                zi_shape[axis_idx] = M
                self.state.zi = xp_create(xp.zeros, tuple(zi_shape), dtype=message.data.dtype, device=dev)
                # 1D taps for conv path
                self.state.fir_b_1d = xp_asarray(xp, b, dtype=message.data.dtype, device=dev)
                # Reshape b to broadcast: (1, ..., M+1, ..., 1) for FFT path
                b_shape = [1] * message.data.ndim
                b_shape[axis_idx] = len(b)
                self.state.fir_b = xp.reshape(self.state.fir_b_1d, tuple(b_shape))
                # Choose method
                self.state.fir_method = "conv" if hasattr(xp, "conv_general") else "fft"
                return

            if is_fir:
                # FIR + numpy: use lfiltic with zero initial conditions
                zi = scipy.signal.lfiltic(b, a, [])
            else:
                # IIR filters...
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
        if not is_numpy_array(message.data):
            xp = get_namespace(message.data)
            zi_tiled = xp_asarray(xp, zi_tiled)
        self.state.zi = zi_tiled
        self.state.fir_method = None
        self.state.fir_b = None
        self.state.fir_b_1d = None

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
            elif self.settings.coef_type == "sos":
                if isinstance(old_coefs, np.ndarray) and isinstance(coefs, np.ndarray):
                    if old_coefs.shape != coefs.shape:
                        reset_needed = True
                else:
                    reset_needed = True

            if reset_needed:
                self.state.zi = None  # This will trigger _reset_state on the next call

    def _process(self, message: AxisArray) -> AxisArray:
        if message.data.size > 0:
            axis = message.dims[0] if self.settings.axis is None else self.settings.axis
            axis_idx = message.get_axis_idx(axis)
            if self.state.fir_method == "conv":
                xp = get_namespace(message.data)
                dat_out, self.state.zi = _fir_filt_conv(self.state.fir_b_1d, message.data, self.state.zi, axis_idx, xp)
            elif self.state.fir_method == "fft":
                xp = get_namespace(message.data)
                dat_out, self.state.zi = _fir_filt_fft(self.state.fir_b, message.data, self.state.zi, axis_idx, xp)
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
                dat_out, self.state.zi = filt_func(*coefs, message.data, axis=axis_idx, zi=self.state.zi)
                if input_xp is not None:
                    dat_out = xp_asarray(input_xp, dat_out)
                    self.state.zi = xp_asarray(input_xp, self.state.zi)
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
        # Update settings
        if new_settings is not None:
            self.settings = new_settings
        else:
            self.settings = replace(self.settings, **kwargs)

        # Set flag to trigger recalculation on next message
        if self.state.filter is not None:
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

        new_settings = FilterSettings(axis=axis, coef_type=self.settings.coef_type, coefs=coefs)
        self.state.filter = FilterTransformer(settings=new_settings)

    def _process(self, message: AxisArray) -> AxisArray:
        return self.state.filter(message)


class BaseFilterByDesignTransformerUnit(
    BaseTransformerUnit[SettingsType, AxisArray, AxisArray, FilterByDesignTransformer],
    typing.Generic[SettingsType, TransformerType],
):
    @ez.subscriber(BaseConsumerUnit.INPUT_SETTINGS)
    async def on_settings(self, msg: SettingsType) -> None:
        """
        Receive a settings message, override self.SETTINGS, and re-create the processor.
        Child classes that wish to have fine-grained control over whether the
        core processor resets on settings changes should override this method.

        Args:
            msg: a settings message.
        """
        self.apply_settings(msg)

        # Check if processor exists yet
        if hasattr(self, "processor") and self.processor is not None:
            # Update the existing processor with new settings
            self.processor.update_settings(self.SETTINGS)
        else:
            # Processor doesn't exist yet, create a new one
            self.create_processor()
