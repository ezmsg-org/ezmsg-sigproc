import typing

import numpy as np
import numpy.typing as npt
import pywt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.util.generator import consumer

from .base import GenAxisArray
from .filterbank import filterbank, FilterbankMode, MinPhaseMode


@consumer
def cwt(
    scales: typing.Union[list, tuple, npt.NDArray],
    wavelet: typing.Union[str, pywt.ContinuousWavelet, pywt.Wavelet],
    min_phase: MinPhaseMode = MinPhaseMode.NONE,
    axis: str = "time",
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Perform a continuous wavelet transform.
    The function is equivalent to the :obj:`pywt.cwt` function, but is designed to work with streaming data.

    Args:
        scales: The wavelet scales to use. Note: Scales will be sorted from largest to smallest.
        wavelet: Wavelet object or name of wavelet to use.
        min_phase: See filterbank MinPhaseMode for details.
        axis: The target axis for operation. Note that this will be moved to the -1th dimension
          because fft and matrix multiplication is much faster on the last axis.
          This axis must be in the msg.axes and it must be of type AxisArray.LinearAxis.

    Returns:
        A primed Generator object that expects an :obj:`AxisArray` via `.send(axis_array)` of continuous data
        and yields an :obj:`AxisArray` with a continuous wavelet transform in its data.
    """
    msg_out: typing.Optional[AxisArray] = None

    # Check parameters
    scales = np.sort(scales)[::-1]
    assert np.all(scales > 0), "Scales must be positive."
    assert scales.ndim == 1, "Scales must be a 1D list, tuple, or array."
    if not isinstance(wavelet, (pywt.ContinuousWavelet, pywt.Wavelet)):
        wavelet = pywt.DiscreteContinuousWavelet(wavelet)
    precision = 10

    # State variables
    neg_rt_scales = -np.sqrt(scales)[:, None]
    int_psi, wave_xvec = pywt.integrate_wavelet(wavelet, precision=precision)
    int_psi = np.conj(int_psi) if wavelet.complex_cwt else int_psi
    template: typing.Optional[AxisArray] = None
    fbgen: typing.Optional[typing.Generator[AxisArray, AxisArray, None]] = None
    last_conv_samp: typing.Optional[npt.NDArray] = None

    # Reset if input changed
    check_input = {
        "kind": None,  # Need to recalc kernels at same complexity as input
        "gain": None,  # Need to recalc freqs
        "shape": None,  # Need to recalc template and buffer
        "key": None,  # Buffer obsolete
    }

    while True:
        msg_in: AxisArray = yield msg_out
        ax_idx = msg_in.get_axis_idx(axis)
        in_shape = msg_in.data.shape[:ax_idx] + msg_in.data.shape[ax_idx + 1 :]

        b_reset = msg_in.data.dtype.kind != check_input["kind"]
        b_reset = b_reset or msg_in.axes[axis].gain != check_input["gain"]
        b_reset = b_reset or in_shape != check_input["shape"]
        b_reset = b_reset or msg_in.key != check_input["key"]
        b_reset = b_reset and msg_in.data.size > 0
        if b_reset:
            check_input["kind"] = msg_in.data.dtype.kind
            check_input["gain"] = msg_in.axes[axis].gain
            check_input["shape"] = in_shape
            check_input["key"] = msg_in.key

            # convert int_psi, wave_xvec to the same precision as the data
            dt_data = msg_in.data.dtype  # _check_dtype(msg_in.data)
            dt_cplx = np.result_type(dt_data, np.complex64)
            dt_psi = dt_cplx if int_psi.dtype.kind == "c" else dt_data
            int_psi = np.asarray(int_psi, dtype=dt_psi)
            # TODO: Currently int_psi cannot be made non-complex once it is complex.

            # Calculate waves for each scale
            wave_xvec = np.asarray(wave_xvec, dtype=msg_in.data.real.dtype)
            wave_range = wave_xvec[-1] - wave_xvec[0]
            step = wave_xvec[1] - wave_xvec[0]
            int_psi_scales = []
            for scale in scales:
                reix = (np.arange(scale * wave_range + 1) / (scale * step)).astype(int)
                if reix[-1] >= int_psi.size:
                    reix = np.extract(reix < int_psi.size, reix)
                int_psi_scales.append(int_psi[reix][::-1])

            # CONV is probably best because we often get huge kernels.
            fbgen = filterbank(
                int_psi_scales, mode=FilterbankMode.CONV, min_phase=min_phase, axis=axis
            )

            freqs = (
                pywt.scale2frequency(wavelet, scales, precision)
                / msg_in.axes[axis].gain
            )
            # Create output template
            dummy_shape = in_shape + (len(scales), 0)
            template = AxisArray(
                np.zeros(
                    dummy_shape, dtype=dt_cplx if wavelet.complex_cwt else dt_data
                ),
                dims=msg_in.dims[:ax_idx] + msg_in.dims[ax_idx + 1 :] + ["freq", axis],
                axes={
                    **msg_in.axes,
                    "freq": AxisArray.CoordinateAxis(
                        unit="Hz", data=freqs, dims=["freq"]
                    ),
                },
                key=msg_in.key,
            )
            last_conv_samp = np.zeros(
                dummy_shape[:-1] + (1,), dtype=template.data.dtype
            )

        conv_msg = fbgen.send(msg_in)

        # Prepend with last_conv_samp before doing diff
        dat = np.concatenate((last_conv_samp, conv_msg.data), axis=-1)
        coef = neg_rt_scales * np.diff(dat, axis=-1)
        # Store last_conv_samp for next iteration.
        last_conv_samp = conv_msg.data[..., -1:]

        if template.data.dtype.kind != "c":
            coef = coef.real

        # pywt.cwt slices off the beginning and end of the result where the convolution overran. We don't have
        #  that luxury when streaming.
        # d = (coef.shape[-1] - msg_in.data.shape[ax_idx]) / 2.
        # coef = coef[..., math.floor(d):-math.ceil(d)]
        msg_out = replace(
            template, data=coef, axes={**template.axes, axis: msg_in.axes[axis]}
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
            axis=self.SETTINGS.axis,
        )
