import functools
import math
import typing

import numpy as np
import scipy.signal as sps
import scipy.fft as sp_fft
from scipy.special import lambertw
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.util.generator import consumer

from .base import GenAxisArray
from .spectrum import OptionsEnum
from .window import windowing


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


@consumer
def filterbank(
    kernels: typing.Union[list[npt.NDArray], tuple[npt.NDArray, ...]],
    mode: FilterbankMode = FilterbankMode.CONV,
    min_phase: MinPhaseMode = MinPhaseMode.NONE,
    axis: str = "time",
    new_axis: str = "kernel",
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Perform multiple (direct or fft) convolutions on a signal using a bank of kernels.
     This is intended to be used during online processing, therefore both direct and fft convolutions
     use the overlap-add method.
    Args:
        kernels:
        mode: "conv", "fft", or "auto". If "auto", the mode is determined by the size of the input data.
          fft mode is more efficient for long kernels. However, fft mode uses non-overlapping windows and will
          incur a delay equal to the window length, which is larger than the largest kernel.
          conv mode is less efficient but will return data for every incoming chunk regardless of how small it is
          and thus can provide shorter latency updates.
        min_phase: If not None, convert the kernels to minimum-phase equivalents. Valid options are
          'hilbert', 'homomorphic', and 'homomorphic-full'. Complex filters not supported.
          See `scipy.signal.minimum_phase` for details.
        axis: The name of the axis to operate on. This should usually be "time".
        new_axis: The name of the new axis corresponding to the kernel index.

    Returns: A primed generator that, when passed an input message via `.send(msg)`, yields an :obj:`AxisArray`
     with the data payload containing the absolute value of the input :obj:`AxisArray` data.

    """
    msg_out: typing.Optional[AxisArray] = None

    # State variables
    template: typing.Optional[AxisArray] = None

    # Reset if these change
    check_input = {
        "key": None,
        "template": None,
        "gain": None,
        "kind": None,
        "shape": None,
    }

    while True:
        msg_in: AxisArray = yield msg_out

        axis = axis or msg_in.dims[0]
        gain = msg_in.axes[axis].gain if axis in msg_in.axes else 1.0
        targ_ax_ix = msg_in.get_axis_idx(axis)
        in_shape = msg_in.data.shape[:targ_ax_ix] + msg_in.data.shape[targ_ax_ix + 1 :]

        b_reset = msg_in.key != check_input["key"]
        b_reset = b_reset or (
            gain != check_input["gain"]
            and mode in [FilterbankMode.FFT, FilterbankMode.AUTO]
        )
        b_reset = b_reset or msg_in.data.dtype.kind != check_input["kind"]
        b_reset = b_reset or in_shape != check_input["shape"]
        if b_reset:
            check_input["key"] = msg_in.key
            check_input["gain"] = gain
            check_input["kind"] = msg_in.data.dtype.kind
            check_input["shape"] = in_shape

            if min_phase != MinPhaseMode.NONE:
                method, half = {
                    MinPhaseMode.HILBERT: ("hilbert", False),
                    MinPhaseMode.HOMOMORPHIC: ("homomorphic", False),
                    # MinPhaseMode.HOMOMORPHICFULL: ("homomorphic", True),
                }[min_phase]
                kernels = [
                    sps.minimum_phase(
                        k, method=method
                    )  # , half=half)  -- half requires later scipy >= 1.14
                    for k in kernels
                ]

            # Determine if this will be operating with complex data.
            b_complex = msg_in.data.dtype.kind == "c" or any(
                [_.dtype.kind == "c" for _ in kernels]
            )

            # Calculate window_dur, window_shift, nfft
            max_kernel_len = max([_.size for _ in kernels])
            # From sps._calc_oa_lens, where s2=max_kernel_len,:
            # fallback_nfft = n_input + max_kernel_len - 1, but n_input is unbound.
            overlap = max_kernel_len - 1

            # Prepare previous iteration's overlap tail to add to input -- all zeros.
            tail_shape = in_shape + (len(kernels), overlap)
            tail = np.zeros(tail_shape, dtype="complex" if b_complex else "float")

            # Prepare output template -- kernels axis immediately before the target axis
            dummy_shape = in_shape + (len(kernels), 0)
            template = AxisArray(
                data=np.zeros(dummy_shape, dtype="complex" if b_complex else "float"),
                dims=msg_in.dims[:targ_ax_ix]
                + msg_in.dims[targ_ax_ix + 1 :]
                + [new_axis, axis],
                axes=msg_in.axes.copy(),  # We do not have info for kernel/filter axis :(.
                key=msg_in.key,
            )

            # Determine optimal mode. Assumes 100 msec chunks.
            if mode == FilterbankMode.AUTO:
                # concatenate kernels into 1 mega kernel then check what's faster.
                # Will typically return fft when combined kernel length is > 1500.
                concat_kernel = np.concatenate(kernels)
                n_dummy = max(2 * len(concat_kernel), int(0.1 / gain))
                dummy_arr = np.zeros(n_dummy)
                mode = sps.choose_conv_method(dummy_arr, concat_kernel, mode="full")
                mode = FilterbankMode.CONV if mode == "direct" else FilterbankMode.FFT

            if mode == FilterbankMode.CONV:
                # Preallocate memory for convolution result and overlap-add
                dest_shape = in_shape + (
                    len(kernels),
                    overlap + msg_in.data.shape[targ_ax_ix],
                )
                dest_arr = np.zeros(
                    dest_shape, dtype="complex" if b_complex else "float"
                )

            elif mode == FilterbankMode.FFT:
                # Calculate optimal nfft and windowing size.
                opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
                nfft = sp_fft.next_fast_len(math.ceil(opt_size))
                win_len = nfft - overlap
                # infft same as nfft. Keeping as separate variable because I might need it again.
                infft = win_len + overlap

                # Create windowing node.
                # Note: We could do windowing manually to avoid the overhead of the message structure,
                #  but windowing is difficult to do correctly, so we lean on the heavily-tested `windowing` generator.
                win_dur = win_len * gain
                wingen = windowing(
                    axis=axis,
                    newaxis="win",  # Big data chunks might yield more than 1 window.
                    window_dur=win_dur,
                    window_shift=win_dur,  # Tumbling (not sliding) windows expected!
                    zero_pad_until="none",
                )

                # Windowing output has an extra "win" dimension, so we need our tail to match.
                tail = np.expand_dims(tail, -2)

                # Prepare fft functions
                # Note: We could instead use `spectrum` but this adds overhead in creating the message structure
                #  for a rather simple calculation. We may revisit if `spectrum` gets additional features, such as
                #  more fft backends.
                if b_complex:
                    fft = functools.partial(sp_fft.fft, n=nfft, norm="backward")
                    ifft = functools.partial(sp_fft.ifft, n=infft, norm="backward")
                else:
                    fft = functools.partial(sp_fft.rfft, n=nfft, norm="backward")
                    ifft = functools.partial(sp_fft.irfft, n=infft, norm="backward")

                # Calculate fft of kernels
                prep_kerns = np.array([fft(_) for _ in kernels])
                prep_kerns = np.expand_dims(prep_kerns, -2)
                # TODO: If fft_kernels have significant stretches of zeros, convert to sparse array.

        # Make sure target axis is in -1th position.
        if targ_ax_ix != (msg_in.data.ndim - 1):
            in_dat = np.moveaxis(msg_in.data, targ_ax_ix, -1)
            if mode == FilterbankMode.FFT:
                # Fix msg_in .dims because we will pass it to wingen
                move_dims = (
                    msg_in.dims[:targ_ax_ix] + msg_in.dims[targ_ax_ix + 1 :] + [axis]
                )
                msg_in = replace(msg_in, data=in_dat, dims=move_dims)
        else:
            in_dat = msg_in.data

        if mode == FilterbankMode.CONV:
            n_dest = in_dat.shape[-1] + overlap
            if dest_arr.shape[-1] < n_dest:
                pad = np.zeros(dest_arr.shape[:-1] + (n_dest - dest_arr.shape[-1],))
                dest_arr = np.concatenate(dest_arr, pad, axis=-1)
            dest_arr.fill(0)
            # TODO: Parallelize this loop.
            for k_ix, k in enumerate(kernels):
                n_out = in_dat.shape[-1] + k.shape[-1] - 1
                dest_arr[..., k_ix, :n_out] = np.apply_along_axis(
                    np.convolve, -1, in_dat, k, mode="full"
                )
            dest_arr[..., :overlap] += tail  # Add previous overlap
            new_tail = dest_arr[..., in_dat.shape[-1] : n_dest]
            if new_tail.size > 0:
                # COPY overlap for next iteration
                tail = new_tail.copy()
            res = dest_arr[..., : in_dat.shape[-1]].copy()
        elif mode == FilterbankMode.FFT:
            # Slice into non-overlapping windows
            win_msg = wingen.send(msg_in)
            # Calculate spectra of each window
            spec_dat = fft(win_msg.data, axis=-1)
            # Insert axis for filters
            spec_dat = np.expand_dims(spec_dat, -3)

            # Do the FFT convolution
            # TODO: handle fft_kernels being sparse. Maybe need np.dot.
            conv_spec = spec_dat * prep_kerns
            overlapped = ifft(conv_spec, axis=-1)

            # Do the overlap-add on the `axis` axis
            # Previous iteration's tail:
            overlapped[..., :1, :overlap] += tail
            # window-to-window:
            overlapped[..., 1:, :overlap] += overlapped[..., :-1, -overlap:]
            # Save tail:
            new_tail = overlapped[..., -1:, -overlap:]
            if new_tail.size > 0:
                # All of the above code works if input is size-zero, but we don't want to save a zero-size tail.
                tail = new_tail  # Save the tail for the next iteration.
            # Concat over win axis, without overlap.
            res = overlapped[..., :-overlap].reshape(overlapped.shape[:-2] + (-1,))

        msg_out = replace(
            template, data=res, axes={**template.axes, axis: msg_in.axes[axis]}
        )


class FilterbankSettings(ez.Settings):
    kernels: typing.Union[list[npt.NDArray], tuple[npt.NDArray, ...]]
    mode: FilterbankMode = FilterbankMode.CONV
    min_phase: MinPhaseMode = MinPhaseMode.NONE
    axis: str = "time"


class Filterbank(GenAxisArray):
    """Unit for :obj:`spectrum`"""

    SETTINGS = FilterbankSettings

    INPUT_SETTINGS = ez.InputStream(FilterbankSettings)

    def construct_generator(self):
        self.STATE.gen = filterbank(
            kernels=self.SETTINGS.kernels,
            mode=self.SETTINGS.mode,
            min_phase=self.SETTINGS.min_phase,
            axis=self.SETTINGS.axis,
        )
