import functools
import typing

import numpy as np
import numpy.typing as npt
import scipy.signal
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from ezmsg.util.generator import consumer

from .base import ProcessorState, BaseStatefulTransformer, BaseTransformerUnit


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


def ewma_step(
    sample: npt.NDArray, zi: npt.NDArray, alpha: float, beta: float | None = None
):
    """
    Do an exponentially weighted moving average step.

    Args:
        sample: The new sample.
        zi: The output of the previous step.
        alpha: Fading factor.
        beta: Persisting factor. If None, it is calculated as 1-alpha.

    Returns:
        alpha * sample + beta * zi

    """
    # Potential micro-optimization:
    #  Current: scalar-arr multiplication, scalar-arr multiplication, arr-arr addition
    #  Alternative: arr-arr subtraction, arr-arr multiplication, arr-arr addition
    # return zi + alpha * (new_sample - zi)
    beta = beta or (1 - alpha)
    return alpha * sample + beta * zi


class EWMA:
    def __init__(self, alpha: float):
        self.beta = 1 - alpha
        self._filt_func = functools.partial(
            scipy.signal.lfilter, [alpha], [1.0, alpha - 1.0], axis=0
        )
        self.prev = None

    def compute(self, arr: npt.NDArray) -> npt.NDArray:
        if self.prev is None:
            self.prev = self.beta * arr[:1]
        expected, self.prev = self._filt_func(arr, zi=self.prev)
        return expected


class EWMA_Deprecated:
    """
    Grabbed these methods from https://stackoverflow.com/a/70998068 and other answers in that topic,
    but they ended up being slower than the scipy.signal.lfilter method.
    Additionally, `compute` and `compute2` suffer from potential errors as the vector length increases
    and beta**n approaches zero.
    """

    def __init__(self, alpha: float, max_len: int):
        self.alpha = alpha
        self.beta = 1 - alpha
        self.prev: npt.NDArray | None = None
        self.weights = np.empty((max_len + 1,), float)
        self._precalc_weights(max_len)
        self._step_func = functools.partial(ewma_step, alpha=self.alpha, beta=self.beta)

    def _precalc_weights(self, n: int):
        #   (1-α)^0, (1-α)^1, (1-α)^2, ..., (1-α)^n
        np.power(self.beta, np.arange(n + 1), out=self.weights)

    def compute(self, arr: npt.NDArray, out: npt.NDArray | None = None) -> npt.NDArray:
        if out is None:
            out = np.empty(arr.shape, arr.dtype)

        n = arr.shape[0]
        weights = self.weights[:n]
        weights = np.expand_dims(weights, list(range(1, arr.ndim)))

        #   α*P0, α*P1, α*P2, ..., α*Pn
        np.multiply(self.alpha, arr, out)

        #   α*P0/(1-α)^0, α*P1/(1-α)^1, α*P2/(1-α)^2, ..., α*Pn/(1-α)^n
        np.divide(out, weights, out)

        #   α*P0/(1-α)^0, α*P0/(1-α)^0 + α*P1/(1-α)^1, ...
        np.cumsum(out, axis=0, out=out)

        #   (α*P0/(1-α)^0)*(1-α)^0, (α*P0/(1-α)^0 + α*P1/(1-α)^1)*(1-α)^1, ...
        np.multiply(out, weights, out)

        # Add the previous output
        if self.prev is None:
            self.prev = arr[:1]

        out += self.prev * np.expand_dims(
            self.weights[1 : n + 1], list(range(1, arr.ndim))
        )

        self.prev = out[-1:]

        return out

    def compute2(self, arr: npt.NDArray) -> npt.NDArray:
        """
        Compute the Exponentially Weighted Moving Average (EWMA) of the input array.

        Args:
            arr: The input array to be smoothed.

        Returns:
            The smoothed array.
        """
        n = arr.shape[0]
        if n > len(self.weights):
            self._precalc_weights(n)
        weights = self.weights[:n][::-1]
        weights = np.expand_dims(weights, list(range(1, arr.ndim)))

        result = np.cumsum(self.alpha * weights * arr, axis=0)
        result = result / weights

        # Handle the first call when prev is unset
        if self.prev is None:
            self.prev = arr[:1]

        result += self.prev * np.expand_dims(
            self.weights[1 : n + 1], list(range(1, arr.ndim))
        )

        # Store the result back into prev
        self.prev = result[-1]

        return result

    def compute_sample(self, new_sample: npt.NDArray) -> npt.NDArray:
        if self.prev is None:
            self.prev = new_sample
        self.prev = self._step_func(new_sample, self.prev)
        return self.prev


@consumer
def scaler(
    time_constant: float = 1.0, axis: str | None = None
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


class AdaptiveStandardScalerSettings(ez.Settings):
    """
    Settings for :obj:`AdaptiveStandardScaler`.
    See :obj:`scaler_np` for a description of the parameters.
    """

    time_constant: float = 1.0
    axis: str | None = None


class AdaptiveStandardScalerState(ProcessorState):
    samps_ewma: EWMA | None = None
    vars_sq_ewma: EWMA | None = None
    alpha: float | None = None


class AdaptiveStandardScalerTransformer(
    BaseStatefulTransformer[
        AdaptiveStandardScalerSettings,
        AxisArray,
        AxisArray,
        AdaptiveStandardScalerState,
    ]
):
    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        data: npt.NDArray = np.moveaxis(message.data, axis_idx, 0)
        return hash((data.shape[1:], message.axes[axis].gain, message.key))

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[0]
        self._state.alpha = _alpha_from_tau(
            self.settings.time_constant, message.axes[axis].gain
        )
        self._state.samps_ewma = EWMA(alpha=self._state.alpha)
        self._state.vars_sq_ewma = EWMA(alpha=self._state.alpha)

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        data = np.moveaxis(message.data, axis_idx, 0)

        # Update step
        means = self._state.samps_ewma.compute(data)
        vars_sq_means = self._state.vars_sq_ewma.compute(data**2)

        # Get step
        varis = vars_sq_means - means**2
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (data - means) / (varis**0.5)
        result[np.isnan(result)] = 0.0
        result = np.moveaxis(result, 0, axis_idx)
        return replace(message, data=result)


class AdaptiveStandardScaler(
    BaseTransformerUnit[
        AdaptiveStandardScalerSettings,
        AxisArray,
        AxisArray,
        AdaptiveStandardScalerTransformer,
    ]
):
    SETTINGS = AdaptiveStandardScalerSettings


# Backwards compatibility...
def scaler_np(
    time_constant: float = 1.0, axis: str | None = None
) -> AdaptiveStandardScalerTransformer:
    return AdaptiveStandardScalerTransformer(
        settings=AdaptiveStandardScalerSettings(time_constant=time_constant, axis=axis)
    )
