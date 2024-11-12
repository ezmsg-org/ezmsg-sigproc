import typing

import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.util.generator import consumer

from .base import GenAxisArray


def _tau_from_alpha(alpha: float, dt: float) -> float:
    """
    Inverse of _alpha_from_tau. See that function for explanation.
    """
    return -dt / np.log(1 - alpha)


def _alpha_from_tau(tau: float, dt: float) -> float:
    """
    # https://en.wikipedia.org/wiki/Exponential_smoothing#Time_constant
    :param tau: The amount of time for the smoothed response of a unit step function to reach
        1 - 1/e approx-eq 63.2%.
    :param dt: sampling period, or 1 / sampling_rate.
    :return: alpha, the "fading factor" in exponential smoothing.
    """
    return 1 - np.exp(-dt / tau)


@consumer
def scaler(
    time_constant: float = 1.0, axis: typing.Optional[str] = None
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Apply the adaptive standard scaler from https://riverml.xyz/latest/api/preprocessing/AdaptiveStandardScaler/
    This is faster than :obj:`scaler_np` for single-channel data.

    Args:
        time_constant: Decay constant `tau` in seconds.
        axis: The name of the axis to accumulate statistics over.

    Returns:
        A primed generator object that expects to be sent a :obj:`AxisArray` via `.send(axis_array)`
         and yields an :obj:`AxisArray` with its data being a standardized, or "Z-scored" version of the input data.
    """
    from river import preprocessing

    msg_out = AxisArray(np.array([]), dims=[""])
    _scaler = None
    while True:
        msg_in: AxisArray = yield msg_out
        data = msg_in.data
        if axis is None:
            axis = msg_in.dims[0]
            axis_idx = 0
        else:
            axis_idx = msg_in.get_axis_idx(axis)
            if axis_idx != 0:
                data = np.moveaxis(data, axis_idx, 0)

        if _scaler is None:
            alpha = _alpha_from_tau(time_constant, msg_in.axes[axis].gain)
            _scaler = preprocessing.AdaptiveStandardScaler(fading_factor=alpha)

        result = []
        for sample in data:
            x = {k: v for k, v in enumerate(sample.flatten().tolist())}
            _scaler.learn_one(x)
            y = _scaler.transform_one(x)
            k = sorted(y.keys())
            result.append(np.array([y[_] for _ in k]).reshape(sample.shape))

        result = np.stack(result)
        result = np.moveaxis(result, 0, axis_idx)
        msg_out = replace(msg_in, data=result)


@consumer
def scaler_np(
    time_constant: float = 1.0, axis: typing.Optional[str] = None
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Create a generator function that applies an adaptive standard scaler.
    This is faster than :obj:`scaler` for multichannel data.

    Args:
        time_constant: Decay constant `tau` in seconds.
        axis: The name of the axis to accumulate statistics over.
            Note: The axis must be in the msg.axes and be of type AxisArray.LinearAxis.

    Returns:
        A primed generator object that expects to be sent a :obj:`AxisArray` via `.send(axis_array)`
         and yields an :obj:`AxisArray` with its data being a standardized, or "Z-scored" version of the input data.
    """
    msg_out = AxisArray(np.array([]), dims=[""])

    # State variables
    alpha: float = 0.0
    means: typing.Optional[npt.NDArray] = None
    vars_means: typing.Optional[npt.NDArray] = None
    vars_sq_means: typing.Optional[npt.NDArray] = None

    # Reset if input changes
    check_input = {
        "gain": None,  # Resets alpha
        "shape": None,
        "key": None,  # Key change implies buffered means/vars are invalid.
    }

    def _ew_update(arr, prev, _alpha):
        if np.all(prev == 0):
            return arr
        # return _alpha * arr + (1 - _alpha) * prev
        # Micro-optimization: sub, mult, add (below) is faster than sub, mult, mult, add (above)
        return prev + _alpha * (arr - prev)

    while True:
        msg_in: AxisArray = yield msg_out

        axis = axis or msg_in.dims[0]
        axis_idx = msg_in.get_axis_idx(axis)

        if msg_in.axes[axis].gain != check_input["gain"]:
            alpha = _alpha_from_tau(time_constant, msg_in.axes[axis].gain)
            check_input["gain"] = msg_in.axes[axis].gain

        data: npt.NDArray = np.moveaxis(msg_in.data, axis_idx, 0)
        b_reset = data.shape[1:] != check_input["shape"]
        b_reset |= msg_in.key != check_input["key"]
        if b_reset:
            check_input["shape"] = data.shape[1:]
            check_input["key"] = msg_in.key
            vars_sq_means = np.zeros_like(data[0], dtype=float)
            vars_means = np.zeros_like(data[0], dtype=float)
            means = np.zeros_like(data[0], dtype=float)

        result = np.zeros_like(data)
        for sample_ix in range(data.shape[0]):
            sample = data[sample_ix]
            # Update step
            vars_means = _ew_update(sample, vars_means, alpha)
            vars_sq_means = _ew_update(sample**2, vars_sq_means, alpha)
            means = _ew_update(sample, means, alpha)
            # Get step
            varis = vars_sq_means - vars_means**2
            y = (sample - means) / (varis**0.5)
            result[sample_ix] = y

        result[np.isnan(result)] = 0.0
        result = np.moveaxis(result, 0, axis_idx)
        msg_out = replace(msg_in, data=result)


class AdaptiveStandardScalerSettings(ez.Settings):
    """
    Settings for :obj:`AdaptiveStandardScaler`.
    See :obj:`scaler_np` for a description of the parameters.
    """

    time_constant: float = 1.0
    axis: typing.Optional[str] = None


class AdaptiveStandardScaler(GenAxisArray):
    """Unit for :obj:`scaler_np`"""

    SETTINGS = AdaptiveStandardScalerSettings

    def construct_generator(self):
        self.STATE.gen = scaler_np(
            time_constant=self.SETTINGS.time_constant, axis=self.SETTINGS.axis
        )
