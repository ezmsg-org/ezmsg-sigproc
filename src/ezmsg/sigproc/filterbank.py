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

    Returns:

    """

    template: typing.Optional[AxisArray] = None
    msg_out: typing.Optional[AxisArray] = None

    while True:
        msg_in: AxisArray = yield msg_out

        if template is None or mode == FilterbankMode.AUTO:
            fs = 1 / msg_in.axes["time"].gain if "time" in msg_in.axes else 1.0
            ax_ix = msg_in.get_axis_idx(axis)
            if mode == FilterbankMode.AUTO:
                # concatenate kernels into 1 mega kernel then check what's faster.
                # Will typically return fft when combined kernel length is > 1500.
                concat_kernel = np.concatenate(kernels)
                n_dummy = max(2 * len(concat_kernel), int(0.1 * fs))
                dummy_arr = np.zeros(n_dummy)
                mode = sps.choose_conv_method(dummy_arr, concat_kernel, mode="full")
                mode = FilterbankMode.CONV if mode == "direct" else FilterbankMode.FFT
            if mode == FilterbankMode.CONV:
                # Note: Instead of filtergen, we could use np.convolve directly and manually manage overlap-add.
                #  We should reconsider this after we finish managing overlap add for fft filtering.
                #  out_full = np.apply_along_axis(lambda y: np.convolve(b, y), axis, x)
                # TODO: Parallelize!
                filtergens = [filtergen(axis="time", coefs=(_, 1.0), coef_type="ba") for _ in kernels]
                # Prepare output template
                template = AxisArray(
                    data=np.zeros(msg_in.data.shape[:ax_ix] + msg_in.data.shape[ax_ix + 1:] + (len(kernels), 0)),
                    dims=msg_in.dims[:ax_ix] + msg_in.dims[ax_ix + 1:] + ["filter", axis],
                    axes=msg_in.axes.copy()  # No idea what to use to fill 'filter' axis.
                )
            elif mode == FilterbankMode.FFT:
                # Note: We should not pass this off to multiple (non-existing) fftconvolve nodes, because here we
                #  can calculate the FFT on the data only once which is then shared across kernels.

                # Determine if this will be operating with complex data.
                b_complex = msg_in.data.dtype.kind == "c" or any([_.dtype.kind == "c" for _ in kernels])

                # Calculate window_dur, window_shift, nfft
                max_kernel_len = max([_.size for _ in kernels])
                # From sps._calc_oa_lens, where s2=max_kernel_len,:
                # fallback_nfft = n_input + max_kernel_len - 1, but n_input is unbound.
                overlap = max_kernel_len - 1
                opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
                nfft = sp_fft.next_fast_len(math.ceil(opt_size))
                win_len = nfft - overlap
                infft = win_len + overlap  # Same as nfft. Keeping additional variable because I might need it again.

                # Create windowing node
                wingen = windowing(
                    axis=axis,
                    newaxis="win",  # Big data chunks might yield more than 1 window.
                    window_dur=win_len / fs,
                    window_shift=win_len / fs,  # Tumbling (not sliding) windows expected!
                    zero_pad_until="none",
                )

                # Note: Instead of calculating fft in this node, we could use the spectrum node:
                # specgen = spectrum(
                #     axis=axis,
                #     window=WindowFunction.NONE,
                #     transform=SpectralTransform.RAW_COMPLEX if b_complex else SpectralTransform.REAL,
                #     output=SpectralOutput.FULL if b_complex else SpectralOutput.POSITIVE,
                #     norm="backward",
                #     do_fftshift=False,
                #     nfft=nfft,
                # )
                # However, this adds unnecessary overhead in creating the message structure and the calls to fft
                #  are straightforward. We can revisit this if we add more fft optimization to spectrum.
                # We accept that overhead for `windowing` because that is very difficult to implement correctly.

                # Prepare fft functions
                if b_complex:
                    fft = functools.partial(sp_fft.fft, n=nfft, norm="backward")
                    ifft = functools.partial(sp_fft.ifft, n=infft, norm="backward")
                else:
                    fft = functools.partial(sp_fft.rfft, n=nfft, norm="backward")
                    ifft = functools.partial(sp_fft.irfft, n=infft, norm="backward")

                # Calculate fft of kernels
                fft_kernels = np.array([fft(_) for _ in kernels])
                fft_kernels = np.expand_dims(fft_kernels, -2)
                # TODO: If fft_kernels have significant stretches of zeros, convert to sparse array.

                # Prepare previous iteration's overlap tail to add to input -- all zeros.
                tail_shape = msg_in.data.shape[:ax_ix] + msg_in.data.shape[ax_ix+1:] + (len(kernels), 1, overlap)
                tail = np.zeros(tail_shape, dtype="complex" if b_complex else "float")

                # Prepare output template
                template = AxisArray(
                    data=np.zeros(tail_shape[:-2] + (0,), dtype="complex" if b_complex else "float"),
                    dims=msg_in.dims[:ax_ix] + msg_in.dims[ax_ix+1:] + ["filter", axis],
                    axes=msg_in.axes.copy()  # No idea what to use to fill 'filter' axis.
                )

        if mode == FilterbankMode.CONV:
            # for each filtergen, send the message, accumulate outputs, stack along "filter" axis.
            filtres = [fg.send(msg_in).data for fg in filtergens]  # TODO: parallelize!?
            msg_out = replace(
                template,
                data=np.stack(filtres, axis=-2),
                axes={**template.axes, axis: msg_in.axes[axis]}
            )
        elif mode == FilterbankMode.FFT:
            # Make sure target axis is in -1th position.
            targ_ax_ix = msg_in.get_axis_idx(axis)
            if targ_ax_ix != (msg_in.data.ndim - 1):
                in_dat = np.moveaxis(msg_in.data, targ_ax_ix, -1)
                move_dims = msg_in.dims[:targ_ax_ix] + msg_in.dims[targ_ax_ix+1:] + [axis]
                msg_in = replace(msg_in, data=in_dat, dims=move_dims)
            # Slice into non-overlapping windows
            win_msg = wingen.send(msg_in)
            # Calculate spectra of each window
            spec_dat = fft(win_msg.data, axis=-1)
            # Insert axis for filters
            spec_dat = np.expand_dims(spec_dat, -3)
            # Do the FFT convolution
            # TODO: handle fft_kernels being sparse. Maybe need np.dot.
            conv_spec = spec_dat * fft_kernels
            overlapped = ifft(conv_spec, axis=-1)
            # Do the overlap-add on the `axis` axis, across rows in the "win" axis.
            overlapped[..., :1, :overlap] += tail  # Previous iteration's tail
            overlapped[..., 1:, :overlap] += overlapped[..., :-1, -overlap:]
            new_tail = overlapped[..., -1:, -overlap:]
            if new_tail.size > 0:
                # All of the above code works if input is size-zero, but we don't want to save a zero-size tail.
                tail = overlapped[..., -1:, -overlap:]  # Save the tail for the next iteration.
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
