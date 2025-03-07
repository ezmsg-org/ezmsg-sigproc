import typing

import numpy as np
import numpy.typing as npt
import pywt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .base import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from .filterbank import filterbank, FilterbankMode, MinPhaseMode


class CWTSettings(ez.Settings):
    """
    Settings for :obj:`CWT`
    See :obj:`cwt` for argument details.
    """

    frequencies: list | tuple | npt.NDArray | None
    wavelet: str | pywt.ContinuousWavelet | pywt.Wavelet
    min_phase: MinPhaseMode = MinPhaseMode.NONE
    axis: str = "time"
    scales: list | tuple | npt.NDArray | None = None


@processor_state
class CWTState:
    neg_rt_scales: npt.NDArray | None = None
    int_psi_scales: list[npt.NDArray] | None = None
    template: AxisArray | None = None
    fbgen: typing.Generator[AxisArray, AxisArray, None] | None = None
    last_conv_samp: npt.NDArray | None = None


class CWTTransformer(
    BaseStatefulTransformer[CWTSettings, AxisArray, AxisArray, CWTState]
):
    def _hash_message(self, message: AxisArray) -> int:
        ax_idx = message.get_axis_idx(self.settings.axis)
        in_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        return hash(
            (
                message.data.dtype.kind,
                message.axes[self.settings.axis].gain,
                in_shape,
                message.key,
            )
        )

    def _reset_state(self, message: AxisArray) -> None:
        precision = 10

        # Process wavelet
        wavelet = (
            self.settings.wavelet
            if isinstance(self.settings.wavelet, (pywt.ContinuousWavelet, pywt.Wavelet))
            else pywt.DiscreteContinuousWavelet(self.settings.wavelet)
        )
        # Process wavelet integration
        int_psi, wave_xvec = pywt.integrate_wavelet(wavelet, precision=precision)
        int_psi = np.conj(int_psi) if wavelet.complex_cwt else int_psi

        # Calculate scales and frequencies
        if self.settings.frequencies is not None:
            frequencies = np.sort(np.array(self.settings.frequencies))
            scales = pywt.frequency2scale(
                wavelet,
                frequencies * message.axes[self.settings.axis].gain,
                precision=precision,
            )
        else:
            scales = np.sort(self.settings.scales)[::-1]

        self._state.neg_rt_scales = -np.sqrt(scales)[:, None]

        # Convert to appropriate dtype
        dt_data = message.data.dtype
        dt_cplx = np.result_type(dt_data, np.complex64)
        dt_psi = dt_cplx if int_psi.dtype.kind == "c" else dt_data
        int_psi = np.asarray(int_psi, dtype=dt_psi)
        # Note: Currently int_psi cannot be made non-complex once it is complex.

        # Calculate waves for each scale
        wave_xvec = np.asarray(wave_xvec, dtype=message.data.real.dtype)
        wave_range = wave_xvec[-1] - wave_xvec[0]
        step = wave_xvec[1] - wave_xvec[0]
        self._state.int_psi_scales = []
        for scale in scales:
            reix = (np.arange(scale * wave_range + 1) / (scale * step)).astype(int)
            if reix[-1] >= int_psi.size:
                reix = np.extract(reix < int_psi.size, reix)
            self._state.int_psi_scales.append(int_psi[reix][::-1])

        # Setup filterbank generator
        self._state.fbgen = filterbank(
            self._state.int_psi_scales,
            mode=FilterbankMode.CONV,
            min_phase=self.settings.min_phase,
            axis=self.settings.axis,
        )

        # Create output template
        ax_idx = message.get_axis_idx(self.settings.axis)
        in_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        freqs = (
            pywt.scale2frequency(wavelet, scales, precision)
            / message.axes[self.settings.axis].gain
        )
        dummy_shape = in_shape + (len(scales), 0)
        self._state.template = AxisArray(
            np.zeros(dummy_shape, dtype=dt_cplx if wavelet.complex_cwt else dt_data),
            dims=message.dims[:ax_idx]
            + message.dims[ax_idx + 1 :]
            + ["freq", self.settings.axis],
            axes={
                **message.axes,
                "freq": AxisArray.CoordinateAxis(unit="Hz", data=freqs, dims=["freq"]),
            },
            key=message.key,
        )
        self._state.last_conv_samp = np.zeros(
            dummy_shape[:-1] + (1,), dtype=self._state.template.data.dtype
        )

    def _process(self, message: AxisArray) -> AxisArray:
        conv_msg = self._state.fbgen.send(message)

        # Prepend with last_conv_samp before doing diff
        dat = np.concatenate((self._state.last_conv_samp, conv_msg.data), axis=-1)
        coef = self._state.neg_rt_scales * np.diff(dat, axis=-1)
        # Store last_conv_samp for next iteration
        self._state.last_conv_samp = conv_msg.data[..., -1:]

        if self._state.template.data.dtype.kind != "c":
            coef = coef.real

        # pywt.cwt slices off the beginning and end of the result where the convolution overran. We don't have
        #  that luxury when streaming.
        # d = (coef.shape[-1] - msg_in.data.shape[ax_idx]) / 2.
        # coef = coef[..., math.floor(d):-math.ceil(d)]
        return replace(
            self._state.template,
            data=coef,
            axes={
                **self._state.template.axes,
                self.settings.axis: message.axes[self.settings.axis],
            },
        )


class CWT(BaseTransformerUnit[CWTSettings, AxisArray, AxisArray, CWTTransformer]):
    SETTINGS = CWTSettings


def cwt(
    frequencies: list | tuple | npt.NDArray | None,
    wavelet: str | pywt.ContinuousWavelet | pywt.Wavelet,
    min_phase: MinPhaseMode = MinPhaseMode.NONE,
    axis: str = "time",
    scales: list | tuple | npt.NDArray | None = None,
) -> CWTTransformer:
    """
    Perform a continuous wavelet transform.
    The function is equivalent to the :obj:`pywt.cwt` function, but is designed to work with streaming data.

    Args:
        frequencies: The wavelet frequencies to use in Hz. If `None` provided then the scales will be used.
          Note: frequencies will be sorted from smallest to largest.
        wavelet: Wavelet object or name of wavelet to use.
        min_phase: See filterbank MinPhaseMode for details.
        axis: The target axis for operation. Note that this will be moved to the -1th dimension
          because fft and matrix multiplication is much faster on the last axis.
          This axis must be in the msg.axes and it must be of type AxisArray.LinearAxis.
        scales: The scales to use. If None, the scales will be calculated from the frequencies.
          Note: Scales will be sorted from largest to smallest.
          Note: Use of scales is deprecated in favor of frequencies. Convert scales to frequencies using
            `pywt.scale2frequency(wavelet, scales, precision=10) * fs` where fs is the sampling frequency.

    Returns:
        A primed Generator object that expects an :obj:`AxisArray` via `.send(axis_array)` of continuous data
        and yields an :obj:`AxisArray` with a continuous wavelet transform in its data.
    """
    return CWTTransformer(
        CWTSettings(
            frequencies=frequencies,
            wavelet=wavelet,
            min_phase=min_phase,
            axis=axis,
            scales=scales,
        )
    )
