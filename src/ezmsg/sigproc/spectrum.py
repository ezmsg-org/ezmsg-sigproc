import enum
import math
import typing
from functools import partial

import ezmsg.core as ez
import numpy as np
from array_api_compat import get_namespace
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import (
    AxisArray,
    replace,
    slice_along_axis,
)

from .util.array import is_complex_dtype


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
    fftshift: typing.Callable | None = None
    f_transform: typing.Callable | None = None
    new_dims: list[str] | None = None
    window: typing.Any = None


class SpectrumTransformer(BaseStatefulTransformer[SpectrumSettings, AxisArray, AxisArray, SpectrumState]):
    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[0]
        ax_idx = message.get_axis_idx(axis)
        ax_info = message.axes[axis]
        targ_len = message.data.shape[ax_idx]
        return hash((targ_len, message.data.ndim, is_complex_dtype(message.data.dtype), ax_idx, ax_info.gain))

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[0]
        ax_idx = message.get_axis_idx(axis)
        ax_info = message.axes[axis]
        targ_len = message.data.shape[ax_idx]
        nfft = self.settings.nfft or targ_len
        xp = get_namespace(message.data)

        # Pre-calculate windowing (always compute with numpy, then convert to backend)
        window_np = WINDOWS[self.settings.window](targ_len)
        shape = [1] * ax_idx + [len(window_np)] + [1] * (message.data.ndim - 1 - ax_idx)
        window = xp.asarray(window_np).reshape(shape)
        if self.settings.transform != SpectralTransform.RAW_COMPLEX and not (
            self.settings.transform == SpectralTransform.REAL or self.settings.transform == SpectralTransform.IMAG
        ):
            scale = float(xp.sum(window**2.0)) * ax_info.gain

        if self.settings.window != WindowFunction.NONE:
            self.state.window = window

        # Build FFT closure with manual norm fallback for backends that don't support norm=
        norm = self.settings.norm
        if norm == "forward":
            norm_factor = 1.0 / nfft
        elif norm == "ortho":
            norm_factor = 1.0 / math.sqrt(nfft)
        else:
            norm_factor = None  # backward / None — no scaling

        def _make_fft_closure(raw_fft):
            """Build a closure that calls *raw_fft* and applies norm manually if needed."""

            def fftfun(x):
                try:
                    return raw_fft(x, n=nfft, axis=ax_idx, norm=norm)
                except TypeError:
                    result = raw_fft(x, n=nfft, axis=ax_idx)
                    if norm_factor is not None:
                        result = result * norm_factor
                    return result

            return fftfun

        # Pre-calculate frequencies and select our fft function.
        b_complex = is_complex_dtype(message.data.dtype)
        self.state.f_sl = slice(None)
        self.state.fftshift = None
        if (not b_complex) and self.settings.output == SpectralOutput.POSITIVE:
            # If input is not complex and desired output is SpectralOutput.POSITIVE, we can save some computation
            #  by using rfft and rfftfreq.
            self.state.fftfun = _make_fft_closure(xp.fft.rfft)
            freqs = np.fft.rfftfreq(nfft, d=ax_info.gain * targ_len / nfft)
        else:
            self.state.fftfun = _make_fft_closure(xp.fft.fft)
            freqs = np.fft.fftfreq(nfft, d=ax_info.gain * targ_len / nfft)
            if self.settings.output == SpectralOutput.POSITIVE:
                self.state.f_sl = slice(None, nfft // 2 + 1 - (nfft % 2))
            elif self.settings.output == SpectralOutput.NEGATIVE:
                freqs = np.fft.fftshift(freqs, axes=-1)
                self.state.f_sl = slice(None, nfft // 2 + 1)
            elif self.settings.do_fftshift and self.settings.output == SpectralOutput.FULL:
                freqs = np.fft.fftshift(freqs, axes=-1)
            freqs = freqs[self.state.f_sl]

        # Store fftshift closure if shifting is needed (use tuple for axes — MLX requirement)
        if (
            self.settings.do_fftshift and self.settings.output == SpectralOutput.FULL
        ) or self.settings.output == SpectralOutput.NEGATIVE:
            self.state.fftshift = partial(xp.fft.fftshift, axes=(ax_idx,))

        freqs = freqs.tolist()  # To please type checking
        self.state.freq_axis = AxisArray.LinearAxis(unit="Hz", gain=freqs[1] - freqs[0], offset=freqs[0])
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
                    return (xp.abs(x) ** 2.0) / scale

                if self.settings.transform == SpectralTransform.REL_DB:

                    def f_transform(x):
                        return 10 * xp.log10(f1(x))
                else:
                    f_transform = f1
        self.state.f_transform = f_transform

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis or message.dims[0]

        new_axes = {k: v for k, v in message.axes.items() if k not in [self.settings.out_axis, axis]}
        new_axes[self.settings.out_axis or axis] = self.state.freq_axis

        if self.state.window is not None:
            win_dat = message.data * self.state.window
        else:
            win_dat = message.data
        spec = self.state.fftfun(win_dat)
        if self.state.fftshift is not None:
            spec = self.state.fftshift(spec)
        spec = self.state.f_transform(spec)
        spec = slice_along_axis(spec, self.state.f_sl, message.get_axis_idx(axis))

        msg_out = replace(message, data=spec, dims=self.state.new_dims, axes=new_axes)
        return msg_out


class Spectrum(BaseTransformerUnit[SpectrumSettings, AxisArray, AxisArray, SpectrumTransformer]):
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
