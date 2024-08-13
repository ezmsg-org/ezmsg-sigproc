from dataclasses import replace
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
from ezmsg.util.generator import consumer
from ezmsg.sigproc.filter import filtergen
from ezmsg.sigproc.window import windowing

from .base import GenAxisArray
from .spectrum import OptionsEnum


class FilterbankMode(OptionsEnum):
    """The mode of operation for the filterbank."""
    CONV = "Direct Convolution"
    FFT = "FFT Convolution"
    AUTO = "Automatic"


@consumer
def filterbank(
    kernels: typing.Union[list[npt.NDArray], tuple[npt.NDArray, ...]],
    mode: FilterbankMode = FilterbankMode.CONV,
    axis: str = "time",
    new_axis: str = "kernel",
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Returns a generator that perform multiple (direct or fft) convolutions on a signal using a bank of kernels.
    This generator is intended to be used during online processing, therefore both direct and fft convolutions
    use the overlap-add method.
    Args:
        kernels:
        mode: "conv", "fft", or "auto". If "auto", the mode is determined by the size of the input data.
          fft mode is more efficient for long kernels. However, fft mode uses non-overlapping windows and will
          incur a delay equal to the window length, which is larger than the largest kernel.
          conv mode is less efficient but will return data for every incoming chunk regardless of how small it is
          and thus can provide shorter latency updates.
        axis: The name of the axis to operate on. This should usually be "time".
        new_axis: The name of the new axis corresponding to the kernel index.

    Returns:

    """

    template: typing.Optional[AxisArray] = None
    msg_out: typing.Optional[AxisArray] = None

    while True:
        msg_in: AxisArray = yield msg_out

        if template is None or mode == FilterbankMode.AUTO:
            fs = 1 / msg_in.axes["time"].gain if "time" in msg_in.axes else 1.0
            ax_ix = msg_in.get_axis_idx(axis)

            # Determine if this will be operating with complex data.
            b_complex = msg_in.data.dtype.kind == "c" or any([_.dtype.kind == "c" for _ in kernels])

            # Calculate window_dur, window_shift, nfft
            max_kernel_len = max([_.size for _ in kernels])
            # From sps._calc_oa_lens, where s2=max_kernel_len,:
            # fallback_nfft = n_input + max_kernel_len - 1, but n_input is unbound.
            overlap = max_kernel_len - 1

            # Prepare previous iteration's overlap tail to add to input -- all zeros.
            tail_shape = msg_in.data.shape[:ax_ix] + msg_in.data.shape[ax_ix + 1:] + (len(kernels), overlap)
            tail = np.zeros(tail_shape, dtype="complex" if b_complex else "float")

            # Prepare output template
            dummy_shape = msg_in.data.shape[:ax_ix] + msg_in.data.shape[ax_ix + 1:] + (len(kernels), 0)
            template = AxisArray(
                data=np.zeros(dummy_shape, dtype="complex" if b_complex else "float"),
                dims=msg_in.dims[:ax_ix] + msg_in.dims[ax_ix + 1:] + [new_axis, axis],
                axes=msg_in.axes.copy()  # No idea what to use to fill 'filter' axis.
            )

            # Determine optimal mode. Assumes 100 msec chunks.
            if mode == FilterbankMode.AUTO:
                # concatenate kernels into 1 mega kernel then check what's faster.
                # Will typically return fft when combined kernel length is > 1500.
                concat_kernel = np.concatenate(kernels)
                n_dummy = max(2 * len(concat_kernel), int(0.1 * fs))
                dummy_arr = np.zeros(n_dummy)
                mode = sps.choose_conv_method(dummy_arr, concat_kernel, mode="full")
                mode = FilterbankMode.CONV if mode == "direct" else FilterbankMode.FFT

            if mode == FilterbankMode.CONV:
                # Prepare kernels
                prep_kerns = np.array([k[..., ::-1].conj() for k in kernels])
                prep_kerns = np.expand_dims(prep_kerns, -2)
            elif mode == FilterbankMode.FFT:
                # Calculate optimal nfft and windowing size.
                opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
                nfft = sp_fft.next_fast_len(math.ceil(opt_size))
                win_len = nfft - overlap
                infft = win_len + overlap  # Same as nfft. Keeping as separate variable because I might need it again.

                # Create windowing node.
                # Note: We could do windowing manually to avoid the overhead of the message structure,
                #  but windowing is difficult to do correctly, so we lean on the heavily-tested `windowing` generator.
                wingen = windowing(
                    axis=axis,
                    newaxis="win",  # Big data chunks might yield more than 1 window.
                    window_dur=win_len / fs,
                    window_shift=win_len / fs,  # Tumbling (not sliding) windows expected!
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
        targ_ax_ix = msg_in.get_axis_idx(axis)
        if targ_ax_ix != (msg_in.data.ndim - 1):
            in_dat = np.moveaxis(msg_in.data, targ_ax_ix, -1)
            if mode == FilterbankMode.FFT:
                # Need to rebuild msg for wingen
                move_dims = msg_in.dims[:targ_ax_ix] + msg_in.dims[targ_ax_ix + 1:] + [axis]
                msg_in = replace(msg_in, data=in_dat, dims=move_dims)
        else:
            in_dat = msg_in.data

        if mode == FilterbankMode.CONV:
            # TODO: Next line can be parallelized.
            overlapped = [sps.correlate(in_dat, k, "full", "direct") for k in prep_kerns]
            # overlapped = np.apply_along_axis(lambda y: np.convolve(k, y), -1, in_dat)
            overlapped = np.stack(overlapped, axis=-2)
            overlapped[..., :overlap] += tail  # Previous iteration's tail
            new_tail = overlapped[..., -overlap:]
            if new_tail.size > 0:
                tail = new_tail
            res = overlapped[..., :-overlap]
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
            overlapped[..., :1, :overlap] += tail  # Previous iteration's tail
            overlapped[..., 1:, :overlap] += overlapped[..., :-1, -overlap:]  # window-to-window
            new_tail = overlapped[..., -1:, -overlap:]  # Save tail
            if new_tail.size > 0:
                # All of the above code works if input is size-zero, but we don't want to save a zero-size tail.
                tail = new_tail  # Save the tail for the next iteration.
            # Concat over win axis, without overlap.
            res = overlapped[..., :-overlap].reshape(overlapped.shape[:-2] + (-1,))

        msg_out = replace(
            template,
            data=res,
            axes={**template.axes, axis: msg_in.axes[axis]}
        )


class FilterbankSettings(ez.Settings):
    kernels: typing.Union[list[npt.NDArray], tuple[npt.NDArray, ...]]
    mode: FilterbankMode = FilterbankMode.CONV
    axis: str = "time"


class Filterbank(GenAxisArray):
    """Unit for :obj:`spectrum`"""
    SETTINGS: FilterbankSettings

    INPUT_SETTINGS = ez.InputStream(FilterbankSettings)

    def construct_generator(self):
        self.STATE.gen = filterbank(
            kernels=self.SETTINGS.kernels, mode=self.SETTINGS.mode, axis=self.SETTINGS.axis
        )
