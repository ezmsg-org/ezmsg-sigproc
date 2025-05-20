import enum
from functools import partial
import typing

import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
    replace,
)

from .base import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)


class OptionsEnum(enum.Enum):
    @classmethod
    def options(cls):
        return list(map(lambda c: c.value, cls))


class WindowFunction(OptionsEnum):
    """Windowing function prior to calculating spectrum."""

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


class SpectrumSettings(ez.Settings):
    """
    Settings for :obj:`Spectrum.
    See :obj:`spectrum` for a description of the parameters.
    """

    axis: str | None = None
    """
    The name of the axis on which to calculate the spectrum.
      Note: The axis must have an .axes entry of type LinearAxis, not CoordinateAxis.
    """

    # n: int | None = None # n parameter for fft

    out_axis: str | None = "freq"
    """The name of the new axis. Defaults to "freq". If none; don't change dim name"""

    window: WindowFunction = WindowFunction.HAMMING
    """The :obj:`WindowFunction` to apply to the data slice prior to calculating the spectrum."""

    transform: SpectralTransform = SpectralTransform.REL_DB
    """The :obj:`SpectralTransform` to apply to the spectral magnitude."""

    output: SpectralOutput = SpectralOutput.POSITIVE
    """The :obj:`SpectralOutput` format."""

    norm: str | None = "forward"
    """
    Normalization mode. Default "forward" is best used when the inverse transform is not needed,
      for example when the goal is to get spectral power. Use "backward" (equivalent to None) to not
      scale the spectrum which is useful when the spectra will be manipulated and possibly inverse-transformed.
      See numpy.fft.fft for details.
    """

    do_fftshift: bool = True
    """
    Whether to apply fftshift to the output. Default is True.
      This value is ignored unless output is SpectralOutput.FULL.
    """

    nfft: int | None = None
    """
    The number of points to use for the FFT. If None, the length of the input data is used.
    """


@processor_state
class SpectrumState:
    f_sl: slice | None = None
    # I would prefer `slice(None)` as f_sl default but this fails because it is mutable.
    freq_axis: AxisArray.LinearAxis | None = None
    fftfun: typing.Callable | None = None
    f_transform: typing.Callable | None = None
    new_dims: list[str] | None = None
    window: npt.NDArray | None = None


class SpectrumTransformer(
    BaseStatefulTransformer[SpectrumSettings, AxisArray, AxisArray, SpectrumState]
):
    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[0]
        ax_idx = message.get_axis_idx(axis)
        ax_info = message.axes[axis]
        targ_len = message.data.shape[ax_idx]
        return hash(
            (targ_len, message.data.ndim, message.data.dtype.kind, ax_idx, ax_info.gain)
        )

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[0]
        ax_idx = message.get_axis_idx(axis)
        ax_info = message.axes[axis]
        targ_len = message.data.shape[ax_idx]
        nfft = self.settings.nfft or targ_len

        # Pre-calculate windowing
        window = WINDOWS[self.settings.window](targ_len)
        window = window.reshape(
            [1] * ax_idx
            + [
                len(window),
            ]
            + [1] * (message.data.ndim - 1 - ax_idx)
        )
        if self.settings.transform != SpectralTransform.RAW_COMPLEX and not (
            self.settings.transform == SpectralTransform.REAL
            or self.settings.transform == SpectralTransform.IMAG
        ):
            scale = np.sum(window**2.0) * ax_info.gain

        if self.settings.window != WindowFunction.NONE:
            self.state.window = window

        # Pre-calculate frequencies and select our fft function.
        b_complex = message.data.dtype.kind == "c"
        self.state.f_sl = slice(None)
        if (not b_complex) and self.settings.output == SpectralOutput.POSITIVE:
            # If input is not complex and desired output is SpectralOutput.POSITIVE, we can save some computation
            #  by using rfft and rfftfreq.
            self.state.fftfun = partial(
                np.fft.rfft, n=nfft, axis=ax_idx, norm=self.settings.norm
            )
            freqs = np.fft.rfftfreq(nfft, d=ax_info.gain * targ_len / nfft)
        else:
            self.state.fftfun = partial(
                np.fft.fft, n=nfft, axis=ax_idx, norm=self.settings.norm
            )
            freqs = np.fft.fftfreq(nfft, d=ax_info.gain * targ_len / nfft)
            if self.settings.output == SpectralOutput.POSITIVE:
                self.state.f_sl = slice(None, nfft // 2 + 1 - (nfft % 2))
            elif self.settings.output == SpectralOutput.NEGATIVE:
                freqs = np.fft.fftshift(freqs, axes=-1)
                self.state.f_sl = slice(None, nfft // 2 + 1)
            elif (
                self.settings.do_fftshift
                and self.settings.output == SpectralOutput.FULL
            ):
                freqs = np.fft.fftshift(freqs, axes=-1)
            freqs = freqs[self.state.f_sl]
        freqs = freqs.tolist()  # To please type checking
        self.state.freq_axis = AxisArray.LinearAxis(
            unit="Hz", gain=freqs[1] - freqs[0], offset=freqs[0]
        )
        self.state.new_dims = (
            message.dims[:ax_idx]
            + [
                self.settings.out_axis or axis,
            ]
            + message.dims[ax_idx + 1 :]
        )

        def f_transform(x):
            return x

        if self.settings.transform != SpectralTransform.RAW_COMPLEX:
            if self.settings.transform == SpectralTransform.REAL:

                def f_transform(x):
                    return x.real
            elif self.settings.transform == SpectralTransform.IMAG:

                def f_transform(x):
                    return x.imag
            else:

                def f1(x):
                    return (np.abs(x) ** 2.0) / scale

                if self.settings.transform == SpectralTransform.REL_DB:

                    def f_transform(x):
                        return 10 * np.log10(f1(x))
                else:
                    f_transform = f1
        self.state.f_transform = f_transform

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis or message.dims[0]
        ax_idx = message.get_axis_idx(axis)
        targ_len = message.data.shape[ax_idx]

        new_axes = {
            k: v
            for k, v in message.axes.items()
            if k not in [self.settings.out_axis, axis]
        }
        new_axes[self.settings.out_axis or axis] = self.state.freq_axis

        if self.state.window is not None:
            win_dat = message.data * self.state.window
        else:
            win_dat = message.data
        spec = self.state.fftfun(
            win_dat,
            n=self.settings.nfft or targ_len,
            axis=ax_idx,
            norm=self.settings.norm,
        )
        # Note: norm="forward" equivalent to `/ nfft`
        if (
            self.settings.do_fftshift and self.settings.output == SpectralOutput.FULL
        ) or self.settings.output == SpectralOutput.NEGATIVE:
            spec = np.fft.fftshift(spec, axes=ax_idx)
        spec = self.state.f_transform(spec)
        spec = slice_along_axis(spec, self.state.f_sl, ax_idx)

        msg_out = replace(message, data=spec, dims=self.state.new_dims, axes=new_axes)
        return msg_out


class Spectrum(
    BaseTransformerUnit[SpectrumSettings, AxisArray, AxisArray, SpectrumTransformer]
):
    SETTINGS = SpectrumSettings


def spectrum(
    axis: str | None = None,
    out_axis: str | None = "freq",
    window: WindowFunction = WindowFunction.HANNING,
    transform: SpectralTransform = SpectralTransform.REL_DB,
    output: SpectralOutput = SpectralOutput.POSITIVE,
    norm: str | None = "forward",
    do_fftshift: bool = True,
    nfft: int | None = None,
) -> SpectrumTransformer:
    """
    Calculate a spectrum on a data slice.

    Returns:
        A :obj:`SpectrumTransformer` object that expects an :obj:`AxisArray` via `.(axis_array)` (__call__)
        containing continuous data and returns an :obj:`AxisArray` with data of spectral magnitudes or powers.
    """
    return SpectrumTransformer(
        SpectrumSettings(
            axis=axis,
            out_axis=out_axis,
            window=window,
            transform=transform,
            output=output,
            norm=norm,
            do_fftshift=do_fftshift,
            nfft=nfft,
        )
    )
