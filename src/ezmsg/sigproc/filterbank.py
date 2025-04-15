import functools
import math
import typing

import numpy as np
import scipy.signal as sps
import scipy.fft as sp_fft
from scipy.special import lambertw
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .base import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from .spectrum import OptionsEnum
from .window import WindowTransformer


class FilterbankMode(OptionsEnum):
    """The mode of operation for the filterbank."""

    CONV = "Direct Convolution"
    FFT = "FFT Convolution"
    AUTO = "Automatic"


class MinPhaseMode(OptionsEnum):
    """The mode of operation for the filterbank."""

    NONE = "No kernel modification"
    HILBERT = "Hilbert Method; designed to be used with equiripple filters (e.g., from remez) with unity or zero gain regions"
    HOMOMORPHIC = "Works best with filters with an odd number of taps, and the resulting minimum phase filter will have a magnitude response that approximates the square root of the original filter’s magnitude response using half the number of taps"
    # HOMOMORPHICFULL = "Like HOMOMORPHIC, but uses the full number of taps and same magnitude"


class FilterbankSettings(ez.Settings):
    kernels: list[npt.NDArray] | tuple[npt.NDArray, ...]

    mode: FilterbankMode = FilterbankMode.CONV
    """
    "conv", "fft", or "auto". If "auto", the mode is determined by the size of the input data.
      fft mode is more efficient for long kernels. However, fft mode uses non-overlapping windows and will
      incur a delay equal to the window length, which is larger than the largest kernel.
      conv mode is less efficient but will return data for every incoming chunk regardless of how small it is
      and thus can provide shorter latency updates.
    """

    min_phase: MinPhaseMode = MinPhaseMode.NONE
    """
    If not None, convert the kernels to minimum-phase equivalents. Valid options are
      'hilbert', 'homomorphic', and 'homomorphic-full'. Complex filters not supported.
      See `scipy.signal.minimum_phase` for details.
    """

    axis: str = "time"
    """The name of the axis to operate on. This should usually be "time"."""

    new_axis: str = "kernel"
    """The name of the new axis corresponding to the kernel index."""


@processor_state
class FilterbankState:
    tail: npt.NDArray | None = None
    template: AxisArray | None = None
    dest_arr: npt.NDArray | None = None
    prep_kerns: npt.NDArray | list[npt.NDArray] | None = None
    windower: WindowTransformer | None = None
    fft: typing.Callable | None = None
    ifft: typing.Callable | None = None
    nfft: int | None = None
    infft: int | None = None
    overlap: int | None = None
    mode: FilterbankMode | None = None


class FilterbankTransformer(
    BaseStatefulTransformer[FilterbankSettings, AxisArray, AxisArray, FilterbankState]
):
    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[0]
        gain = message.axes[axis].gain if axis in message.axes else 1.0
        targ_ax_ix = message.get_axis_idx(axis)
        in_shape = (
            message.data.shape[:targ_ax_ix] + message.data.shape[targ_ax_ix + 1 :]
        )

        return hash(
            (
                message.key,
                gain
                if self.settings.mode in [FilterbankMode.FFT, FilterbankMode.AUTO]
                else None,
                message.data.dtype.kind,
                in_shape,
            )
        )

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[0]
        gain = message.axes[axis].gain if axis in message.axes else 1.0
        targ_ax_ix = message.get_axis_idx(axis)
        in_shape = (
            message.data.shape[:targ_ax_ix] + message.data.shape[targ_ax_ix + 1 :]
        )

        kernels = self.settings.kernels
        if self.settings.min_phase != MinPhaseMode.NONE:
            method, half = {
                MinPhaseMode.HILBERT: ("hilbert", False),
                MinPhaseMode.HOMOMORPHIC: ("homomorphic", False),
                # MinPhaseMode.HOMOMORPHICFULL: ("homomorphic", True),
            }[self.settings.min_phase]
            kernels = [sps.minimum_phase(k, method=method) for k in kernels]

        # Determine if this will be operating with complex data.
        b_complex = message.data.dtype.kind == "c" or any(
            [_.dtype.kind == "c" for _ in kernels]
        )

        # Calculate window_dur, window_shift, nfft
        max_kernel_len = max([_.size for _ in kernels])
        # From sps._calc_oa_lens, where s2=max_kernel_len,:
        # fallback_nfft = n_input + max_kernel_len - 1, but n_input is unbound.
        self._state.overlap = max_kernel_len - 1

        # Prepare previous iteration's overlap tail to add to input -- all zeros.
        tail_shape = in_shape + (len(kernels), self._state.overlap)
        self._state.tail = np.zeros(
            tail_shape, dtype="complex" if b_complex else "float"
        )

        # Prepare output template -- kernels axis immediately before the target axis
        dummy_shape = in_shape + (len(kernels), 0)
        self._state.template = AxisArray(
            data=np.zeros(dummy_shape, dtype="complex" if b_complex else "float"),
            dims=message.dims[:targ_ax_ix]
            + message.dims[targ_ax_ix + 1 :]
            + [self.settings.new_axis, axis],
            axes=message.axes.copy(),
            key=message.key,
        )

        # Determine optimal mode. Assumes 100 msec chunks.
        self._state.mode = self.settings.mode
        if self._state.mode == FilterbankMode.AUTO:
            # concatenate kernels into 1 mega kernel then check what's faster.
            # Will typically return fft when combined kernel length is > 1500.
            concat_kernel = np.concatenate(kernels)
            n_dummy = max(2 * len(concat_kernel), int(0.1 / gain))
            dummy_arr = np.zeros(n_dummy)
            self._state.mode = (
                FilterbankMode.CONV
                if sps.choose_conv_method(dummy_arr, concat_kernel, mode="full")
                == "direct"
                else FilterbankMode.FFT
            )

        if self._state.mode == FilterbankMode.CONV:
            # Preallocate memory for convolution result and overlap-add
            dest_shape = in_shape + (
                len(kernels),
                self._state.overlap + message.data.shape[targ_ax_ix],
            )
            self._state.dest_arr = np.zeros(
                dest_shape, dtype="complex" if b_complex else "float"
            )
            self._state.prep_kerns = kernels
        else:  # FFT mode
            # Calculate optimal nfft and windowing size.
            opt_size = (
                -self._state.overlap
                * lambertw(-1 / (2 * math.e * self._state.overlap), k=-1).real
            )
            self._state.nfft = sp_fft.next_fast_len(math.ceil(opt_size))
            win_len = self._state.nfft - self._state.overlap
            # infft same as nfft. Keeping as separate variable because I might need it again.
            self._state.infft = win_len + self._state.overlap

            # Create windowing node.
            # Note: We could do windowing manually to avoid the overhead of the message structure,
            #  but windowing is difficult to do correctly, so we lean on the heavily-tested `windowing` generator.
            win_dur = win_len * gain
            self._state.windower = WindowTransformer(
                axis=axis,
                newaxis="win",
                window_dur=win_dur,
                window_shift=win_dur,
                zero_pad_until="none",
            )

            # Windowing output has an extra "win" dimension, so we need our tail to match.
            self._state.tail = np.expand_dims(self._state.tail, -2)

            # Prepare fft functions
            # Note: We could instead use `spectrum` but this adds overhead in creating the message structure
            #  for a rather simple calculation. We may revisit if `spectrum` gets additional features, such as
            #  more fft backends.
            if b_complex:
                self._state.fft = functools.partial(
                    sp_fft.fft, n=self._state.nfft, norm="backward"
                )
                self._state.ifft = functools.partial(
                    sp_fft.ifft, n=self._state.infft, norm="backward"
                )
            else:
                self._state.fft = functools.partial(
                    sp_fft.rfft, n=self._state.nfft, norm="backward"
                )
                self._state.ifft = functools.partial(
                    sp_fft.irfft, n=self._state.infft, norm="backward"
                )

            # Calculate fft of kernels
            self._state.prep_kerns = np.array([self._state.fft(_) for _ in kernels])
            self._state.prep_kerns = np.expand_dims(self._state.prep_kerns, -2)
            # TODO: If fft_kernels have significant stretches of zeros, convert to sparse array.

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis or message.dims[0]
        targ_ax_ix = message.get_axis_idx(axis)

        # Make sure target axis is in -1th position.
        if targ_ax_ix != (message.data.ndim - 1):
            in_dat = np.moveaxis(message.data, targ_ax_ix, -1)
            if self._state.mode == FilterbankMode.FFT:
                # Fix message.dims because we will pass it to windower
                move_dims = (
                    message.dims[:targ_ax_ix] + message.dims[targ_ax_ix + 1 :] + [axis]
                )
                message = replace(message, data=in_dat, dims=move_dims)
        else:
            in_dat = message.data

        if self._state.mode == FilterbankMode.CONV:
            n_dest = in_dat.shape[-1] + self._state.overlap
            if self._state.dest_arr.shape[-1] < n_dest:
                pad = np.zeros(
                    self._state.dest_arr.shape[:-1]
                    + (n_dest - self._state.dest_arr.shape[-1],)
                )
                self._state.dest_arr = np.concatenate(
                    [self._state.dest_arr, pad], axis=-1
                )
            self._state.dest_arr.fill(0)

            # Note: I tried several alternatives to this loop; all were slower than this.
            #  numba.jit; stride_tricks + np.einsum; threading. Latter might be better with Python 3.13.
            for k_ix, k in enumerate(self._state.prep_kerns):
                n_out = in_dat.shape[-1] + k.shape[-1] - 1
                self._state.dest_arr[..., k_ix, :n_out] = np.apply_along_axis(
                    np.convolve, -1, in_dat, k, mode="full"
                )
            self._state.dest_arr[..., : self._state.overlap] += self._state.tail
            new_tail = self._state.dest_arr[..., in_dat.shape[-1] : n_dest]
            if new_tail.size > 0:
                # COPY overlap for next iteration
                self._state.tail = new_tail.copy()
            res = self._state.dest_arr[..., : in_dat.shape[-1]].copy()
        else:  # FFT mode
            # Slice into non-overlapping windows
            win_msg = self._state.windower.send(message)
            # Calculate spectrum of each window
            spec_dat = self._state.fft(win_msg.data, axis=-1)
            # Insert axis for filters
            spec_dat = np.expand_dims(spec_dat, -3)

            # Do the FFT convolution
            # TODO: handle fft_kernels being sparse. Maybe need np.dot.
            conv_spec = spec_dat * self._state.prep_kerns
            overlapped = self._state.ifft(conv_spec, axis=-1)

            # Do the overlap-add on the `axis` axis
            # Previous iteration's tail:
            overlapped[..., :1, : self._state.overlap] += self._state.tail
            # window-to-window:
            overlapped[..., 1:, : self._state.overlap] += overlapped[
                ..., :-1, -self._state.overlap :
            ]
            # Save tail:
            new_tail = overlapped[..., -1:, -self._state.overlap :]
            if new_tail.size > 0:
                # All of the above code works if input is size-zero, but we don't want to save a zero-size tail.
                self._state.tail = new_tail
            # Concat over win axis, without overlap.
            res = overlapped[..., : -self._state.overlap].reshape(
                overlapped.shape[:-2] + (-1,)
            )

        return replace(
            self._state.template,
            data=res,
            axes={**self._state.template.axes, axis: message.axes[axis]},
        )


class Filterbank(
    BaseTransformerUnit[FilterbankSettings, AxisArray, AxisArray, FilterbankTransformer]
):
    SETTINGS = FilterbankSettings


def filterbank(
    kernels: list[npt.NDArray] | tuple[npt.NDArray, ...],
    mode: FilterbankMode = FilterbankMode.CONV,
    min_phase: MinPhaseMode = MinPhaseMode.NONE,
    axis: str = "time",
    new_axis: str = "kernel",
) -> FilterbankTransformer:
    """
    Perform multiple (direct or fft) convolutions on a signal using a bank of kernels.
     This is intended to be used during online processing, therefore both direct and fft convolutions
     use the overlap-add method.

    Returns: :obj:`FilterbankTransformer`.
    """
    return FilterbankTransformer(
        settings=FilterbankSettings(
            kernels=kernels,
            mode=mode,
            min_phase=min_phase,
            axis=axis,
            new_axis=new_axis,
        )
    )
