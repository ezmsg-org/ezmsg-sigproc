import typing

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from ezmsg.util.generator import consumer

from .base import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from .ewma import EWMATransformer, EWMASettings, _alpha_from_tau

# Imports for backwards compatibility with previous module location
from .ewma import EWMA_Deprecated as EWMA_Deprecated
from .ewma import ewma_step as ewma_step
from .ewma import _tau_from_alpha as _tau_from_alpha


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


class AdaptiveStandardScalerSettings(EWMASettings): ...


@processor_state
class AdaptiveStandardScalerState:
    samps_ewma: EWMATransformer | None = None
    vars_sq_ewma: EWMATransformer | None = None
    alpha: float | None = None


class AdaptiveStandardScalerTransformer(
    BaseStatefulTransformer[
        AdaptiveStandardScalerSettings,
        AxisArray,
        AxisArray,
        AdaptiveStandardScalerState,
    ]
):
    def _reset_state(self, message: AxisArray) -> None:
        self._state.samps_ewma = EWMATransformer(
            time_constant=self.settings.time_constant, axis=self.settings.axis
        )
        self._state.vars_sq_ewma = EWMATransformer(
            time_constant=self.settings.time_constant, axis=self.settings.axis
        )

    def _process(self, message: AxisArray) -> AxisArray:
        # Update step
        mean_message = self._state.samps_ewma(message)
        var_sq_message = self._state.vars_sq_ewma(
            replace(message, data=message.data**2)
        )

        # Get step
        varis = var_sq_message.data - mean_message.data**2
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (message.data - mean_message.data) / (varis**0.5)
        result[np.isnan(result)] = 0.0
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
