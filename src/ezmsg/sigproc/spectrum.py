from dataclasses import replace
import enum
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.generator import consumer

from .base import GenAxisArray


class OptionsEnum(enum.Enum):
    @classmethod
    def options(cls):
        return list(map(lambda c: c.value, cls))


class WindowFunction(OptionsEnum):
    """Windowing function prior to calculating spectrum. """
    NONE = "None (Rectangular)"
    """None."""

    HAMMING = "Hamming"
    """:obj:`numpy.hamming`"""

    HANNING = "Hanning"
    """:obj:`numpy.hanning`"""

    BARTLETT = "Bartlett"
    """:obj:`numpy.bartlett`"""

    BLACKMAN = "Blackman"
    """:obj:`numpy.blackman`"""


WINDOWS = {
    WindowFunction.NONE: np.ones,
    WindowFunction.HAMMING: np.hamming,
    WindowFunction.HANNING: np.hanning,
    WindowFunction.BARTLETT: np.bartlett,
    WindowFunction.BLACKMAN: np.blackman,
}


class SpectralTransform(OptionsEnum):
    """Additional transformation functions to apply to the spectral result."""
    RAW_COMPLEX = "Complex FFT Output"
    REAL = "Real Component of FFT"
    IMAG = "Imaginary Component of FFT"
    REL_POWER = "Relative Power"
    REL_DB = "Log Power (Relative dB)"


class SpectralOutput(OptionsEnum):
    """The expected spectral contents."""
    FULL = "Full Spectrum"
    POSITIVE = "Positive Frequencies"
    NEGATIVE = "Negative Frequencies"


@consumer
def spectrum(
    axis: typing.Optional[str] = None,
    out_axis: typing.Optional[str] = "freq",
    window: WindowFunction = WindowFunction.HANNING,
    transform: SpectralTransform = SpectralTransform.REL_DB,
    output: SpectralOutput = SpectralOutput.POSITIVE,
    norm: typing.Optional[str] = "forward",
    do_fftshift: bool = True,
    nfft: typing.Optional[int] = None,
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Calculate a spectrum on a data slice.

    Args:
        axis: The name of the axis on which to calculate the spectrum.
        out_axis: The name of the new axis. Defaults to "freq".
        window: The :obj:`WindowFunction` to apply to the data slice prior to calculating the spectrum.
        transform: The :obj:`SpectralTransform` to apply to the spectral magnitude.
        output: The :obj:`SpectralOutput` format.
        norm: Normalization mode. Default "forward" is best used when the inverse transform is not needed,
          for example when the goal is to get spectral power. Use "backward" (equivalent to None) to not
          scale the spectrum which is useful when the spectra will be manipulated and possibly inverse-transformed.
          See numpy.fft.fft for details.
        do_fftshift: Whether to apply fftshift to the output. Default is True. This value is ignored unless
          output is SpectralOutput.FULL.
        nfft: The number of points to use for the FFT. If None, the length of the input data is used.

    Returns:
        A primed generator object that expects `.send(axis_array)` of continuous data
        and yields an AxisArray of spectral magnitudes or powers.
    """

    # State variables
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])

    axis_name = axis
    axis_idx = None
    n_time = None
    apply_window = window != WindowFunction.NONE
    b_shift = do_fftshift or output != SpectralOutput.FULL

    while True:
        axis_arr_in = yield axis_arr_out

        if axis_name is None:
            axis_name = axis_arr_in.dims[0]

        # Initial setup
        if n_time is None or axis_idx is None or axis_arr_in.data.shape[axis_idx] != n_time:
            axis_idx = axis_arr_in.get_axis_idx(axis_name)
            _axis = axis_arr_in.get_axis(axis_name)
            n_time = axis_arr_in.data.shape[axis_idx]
            nfft = nfft or n_time
            freqs = np.fft.fftfreq(nfft, d=_axis.gain * n_time / nfft)
            if b_shift:
                freqs = np.fft.fftshift(freqs, axes=-1)
            window = WINDOWS[window](n_time)
            window = window.reshape([1] * axis_idx + [len(window),] + [1] * (axis_arr_in.data.ndim - 1 - axis_idx))
            if (transform != SpectralTransform.RAW_COMPLEX and
                    not (transform == SpectralTransform.REAL or transform == SpectralTransform.IMAG)):
                scale = np.sum(window ** 2.0) * _axis.gain
            axis_offset = freqs[0]
            if output == SpectralOutput.POSITIVE:
                axis_offset = freqs[nfft // 2]
            freq_axis = AxisArray.Axis(
                unit="Hz", gain=1.0 / (_axis.gain * nfft), offset=axis_offset
            )
            if out_axis is None:
                out_axis = axis_name
            new_dims = axis_arr_in.dims[:axis_idx] + [out_axis, ] + axis_arr_in.dims[axis_idx + 1:]

            f_transform = lambda x: x
            if transform != SpectralTransform.RAW_COMPLEX:
                if transform == SpectralTransform.REAL:
                    f_transform = lambda x: x.real
                elif transform == SpectralTransform.IMAG:
                    f_transform = lambda x: x.imag
                else:
                    f1 = lambda x: (2.0 * (np.abs(x) ** 2.0)) / scale
                    if transform == SpectralTransform.REL_DB:
                        f_transform = lambda x: 10 * np.log10(f1(x))
                    else:
                        f_transform = f1

        new_axes = {k: v for k, v in axis_arr_in.axes.items() if k not in [out_axis, axis_name]}
        new_axes[out_axis] = freq_axis

        if apply_window:
            win_dat = axis_arr_in.data * window
        else:
            win_dat = axis_arr_in.data
        spec = np.fft.fft(win_dat, n=nfft, axis=axis_idx, norm=norm)  # norm="forward" equivalent to `/ nfft`
        if b_shift:
            spec = np.fft.fftshift(spec, axes=axis_idx)
        spec = f_transform(spec)

        if output == SpectralOutput.POSITIVE:
            spec = slice_along_axis(spec, slice(nfft // 2, None), axis_idx)

        elif output == SpectralOutput.NEGATIVE:
            spec = slice_along_axis(spec, slice(None, nfft // 2), axis_idx)

        axis_arr_out = replace(axis_arr_in, data=spec, dims=new_dims, axes=new_axes)


class SpectrumSettings(ez.Settings):
    """
    Settings for :obj:`Spectrum.
    See :obj:`spectrum` for a description of the parameters.
    """
    axis: typing.Optional[str] = None
    # n: typing.Optional[int] = None # n parameter for fft
    out_axis: typing.Optional[str] = "freq"  # If none; don't change dim name
    window: WindowFunction = WindowFunction.HAMMING
    transform: SpectralTransform = SpectralTransform.REL_DB
    output: SpectralOutput = SpectralOutput.POSITIVE


class Spectrum(GenAxisArray):
    """Unit for :obj:`spectrum`"""
    SETTINGS = SpectrumSettings

    INPUT_SETTINGS = ez.InputStream(SpectrumSettings)

    def construct_generator(self):
        self.STATE.gen = spectrum(
            axis=self.SETTINGS.axis,
            out_axis=self.SETTINGS.out_axis,
            window=self.SETTINGS.window,
            transform=self.SETTINGS.transform,
            output=self.SETTINGS.output
        )
