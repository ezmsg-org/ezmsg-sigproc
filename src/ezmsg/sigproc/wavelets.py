from dataclasses import replace
import typing

import numpy as np
import numpy.typing as npt
import pywt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer

from ezmsg.sigproc.base import GenAxisArray
from ezmsg.sigproc.filterbank import filterbank, FilterbankMode, MinPhaseMode


@consumer
def cwt(
    scales: typing.Union[list, tuple, npt.NDArray],
    wavelet: typing.Union[str, pywt.ContinuousWavelet, pywt.Wavelet],
    min_phase: MinPhaseMode = MinPhaseMode.NONE,
    axis: str = "time",
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Build a generator to perform a continuous wavelet transform on sent AxisArray messages.
    The function is equivalent to the `pywt.cwt` function, but is designed to work with streaming data.

    Args:
        scales: The wavelet scales to use.
        wavelet: Wavelet object or name of wavelet to use.
        min_phase: See filterbank MinPhaseMode for details.
        axis: The target axis for operation. Note that this will be moved to the -1th dimension
          because fft and matrix multiplication is much faster on the last axis.

    Returns:
        A Generator object that expects `.send(axis_array)` of continuous data
    """
    scales = np.array(scales)
    assert np.all(scales > 0), "Scales must be positive."
    assert scales.ndim == 1, "Scales must be a 1D list, tuple, or array."
    neg_rt_scales = -np.sqrt(scales)[:, None]

    if not isinstance(wavelet, (pywt.ContinuousWavelet, pywt.Wavelet)):
        wavelet = pywt.DiscreteContinuousWavelet(wavelet)
    precision = 10
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
    int_psi = np.conj(int_psi) if wavelet.complex_cwt else int_psi

    msg_out: typing.Optional[AxisArray] = None
    template: typing.Optional[AxisArray] = None

    while True:
        msg_in: AxisArray = yield msg_out
        ax_idx = msg_in.get_axis_idx(axis)

        if msg_in.data.size and template is None:
            # convert int_psi, x to the same precision as the data
            dt_data = msg_in.data.dtype  # _check_dtype(msg_in.data)
            dt_cplx = np.result_type(dt_data, np.complex64)
            dt_psi = dt_cplx if int_psi.dtype.kind == 'c' else dt_data
            int_psi = np.asarray(int_psi, dtype=dt_psi)
            x = np.asarray(x, dtype=msg_in.data.real.dtype)
            wave_range = x[-1] - x[0]
            step = x[1] - x[0]
            int_psi_scales = []
            for scale in scales:
                reix = (np.arange(scale * wave_range + 1) / (scale * step)).astype(int)
                if reix[-1] >= int_psi.size:
                    reix = np.extract(reix < int_psi.size, reix)
                int_psi_scales.append(int_psi[reix][::-1])

            # CONV is probably best because we often get huge kernels.
            fbgen = filterbank(int_psi_scales, mode=FilterbankMode.CONV, min_phase=min_phase, axis=axis)

            freqs = pywt.scale2frequency(wavelet, scales, precision) / msg_in.axes[axis].gain
            fstep = (freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
            # Create output template
            dummy_shape = msg_in.data.shape[:ax_idx] + msg_in.data.shape[ax_idx + 1:] + (len(scales), 0)
            template = AxisArray(
                np.zeros(dummy_shape, dtype=dt_cplx if wavelet.complex_cwt else dt_data),
                dims=msg_in.dims[:ax_idx] + msg_in.dims[ax_idx + 1:] + ["freq", axis],
                axes={
                    **msg_in.axes,
                    "freq": AxisArray.Axis("Hz", offset=freqs[0], gain=fstep)
                },
            )
            last_conv_samp = np.zeros(dummy_shape[:-1] + (1,), dtype=template.data.dtype)

        conv_msg = fbgen.send(msg_in)

        # Prepend with last_conv_samp before doing diff
        dat = np.concatenate((last_conv_samp, conv_msg.data), axis=-1)
        coef = neg_rt_scales * np.diff(dat, axis=-1)
        # Store last_conv_samp for next iteration.
        last_conv_samp = conv_msg.data[..., -1:]

        if template.data.dtype.kind != 'c':
            coef = coef.real

        # pywt.cwt slices off the beginning and end of the result where the convolution overran. We don't have
        #  that luxury when streaming.
        # d = (coef.shape[-1] - msg_in.data.shape[ax_idx]) / 2.
        # coef = coef[..., math.floor(d):-math.ceil(d)]
        msg_out = replace(
            template,
            data=coef,
            axes={**template.axes, axis: msg_in.axes[axis]}
        )


class CWTSettings(ez.Settings):
    """
    Settings for :obj:`CWT`
    See :obj:`cwt` for argument details.
    """
    scales: typing.Union[list, tuple, npt.NDArray]
    wavelet: typing.Union[str, pywt.ContinuousWavelet, pywt.Wavelet]
    min_phase: MinPhaseMode = MinPhaseMode.NONE
    axis: str = "time"


class CWT(GenAxisArray):
    """
    :obj:`Unit` for :obj:`common_rereference`.
    """
    SETTINGS = CWTSettings

    def construct_generator(self):
        self.STATE.gen = cwt(
            scales=self.SETTINGS.scales,
            wavelet=self.SETTINGS.wavelet,
            min_phase=self.SETTINGS.min_phase,
            axis=self.SETTINGS.axis
        )
