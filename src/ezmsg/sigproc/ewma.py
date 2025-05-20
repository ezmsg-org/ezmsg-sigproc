from dataclasses import field
import functools

import numpy as np
import numpy.typing as npt
import scipy.signal as sps
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.messages.util import replace

from .base import BaseStatefulTransformer, processor_state, BaseTransformerUnit


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


class EWMASettings(ez.Settings):
    time_constant: float = 1.0
    axis: str | None = None


@processor_state
class EWMAState:
    alpha: float = field(default_factory=lambda: _alpha_from_tau(1.0, 1000.0))
    zi: npt.NDArray | None = None


class EWMATransformer(
    BaseStatefulTransformer[EWMASettings, AxisArray, AxisArray, EWMAState]
):
    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        sample_shape = (
            message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        )
        return hash((sample_shape, message.axes[axis].gain, message.key))

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[0]
        self._state.alpha = _alpha_from_tau(
            self.settings.time_constant, message.axes[axis].gain
        )
        sub_dat = slice_along_axis(
            message.data, slice(None, 1, None), axis=message.get_axis_idx(axis)
        )
        self._state.zi = (1 - self._state.alpha) * sub_dat

    def _process(self, message: AxisArray) -> AxisArray:
        if np.prod(message.data.shape) == 0:
            return message
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        expected, self._state.zi = sps.lfilter(
            [self._state.alpha],
            [1.0, self._state.alpha - 1.0],
            message.data,
            axis=axis_idx,
            zi=self._state.zi,
        )
        return replace(message, data=expected)


class EWMAUnit(
    BaseTransformerUnit[EWMASettings, AxisArray, AxisArray, EWMATransformer]
):
    SETTINGS = EWMASettings
