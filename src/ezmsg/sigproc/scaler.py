import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

# Imports for backwards compatibility with previous module location
from .ewma import EWMA_Deprecated as EWMA_Deprecated
from .ewma import EWMASettings, EWMATransformer, _alpha_from_tau
from .ewma import _tau_from_alpha as _tau_from_alpha
from .ewma import ewma_step as ewma_step


class RiverAdaptiveStandardScalerSettings(ez.Settings):
    time_constant: float = 1.0
    """Decay constant ``tau`` in seconds."""

    axis: str | None = None
    """The name of the axis to accumulate statistics over."""


@processor_state
class RiverAdaptiveStandardScalerState:
    scaler: typing.Any = None
    axis: str | None = None
    axis_idx: int = 0


class RiverAdaptiveStandardScalerTransformer(
    BaseStatefulTransformer[
        RiverAdaptiveStandardScalerSettings,
        AxisArray,
        AxisArray,
        RiverAdaptiveStandardScalerState,
    ]
):
    """
    Apply the adaptive standard scaler from
    `river <https://riverml.xyz/latest/api/preprocessing/AdaptiveStandardScaler/>`_.

    This processes data sample-by-sample using River's online learning
    implementation. For a vectorized EWMA-based alternative, see
    :class:`AdaptiveStandardScalerTransformer`.
    """

    def _reset_state(self, message: AxisArray) -> None:
        from river import preprocessing

        axis = self.settings.axis
        if axis is None:
            axis = message.dims[0]
            self._state.axis_idx = 0
        else:
            self._state.axis_idx = message.get_axis_idx(axis)
        self._state.axis = axis

        alpha = _alpha_from_tau(self.settings.time_constant, message.axes[axis].gain)
        self._state.scaler = preprocessing.AdaptiveStandardScaler(fading_factor=alpha)

    def _process(self, message: AxisArray) -> AxisArray:
        data = message.data
        axis_idx = self._state.axis_idx
        if axis_idx != 0:
            data = np.moveaxis(data, axis_idx, 0)

        result = []
        for sample in data:
            x = {k: v for k, v in enumerate(sample.flatten().tolist())}
            self._state.scaler.learn_one(x)
            y = self._state.scaler.transform_one(x)
            k = sorted(y.keys())
            result.append(np.array([y[_] for _ in k]).reshape(sample.shape))

        result = np.stack(result)
        result = np.moveaxis(result, 0, axis_idx)
        return replace(message, data=result)


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
            time_constant=self.settings.time_constant,
            axis=self.settings.axis,
            accumulate=self.settings.accumulate,
        )
        self._state.vars_sq_ewma = EWMATransformer(
            time_constant=self.settings.time_constant,
            axis=self.settings.axis,
            accumulate=self.settings.accumulate,
        )

    @property
    def accumulate(self) -> bool:
        """Whether to accumulate statistics from incoming samples."""
        return self.settings.accumulate

    @accumulate.setter
    def accumulate(self, value: bool) -> None:
        """
        Set the accumulate mode and propagate to child EWMA transformers.

        Args:
            value: If True, update statistics with each sample.
                   If False, only apply current statistics without updating.
        """
        if self._state.samps_ewma is not None:
            self._state.samps_ewma.settings = replace(self._state.samps_ewma.settings, accumulate=value)
        if self._state.vars_sq_ewma is not None:
            self._state.vars_sq_ewma.settings = replace(self._state.vars_sq_ewma.settings, accumulate=value)

    def _process(self, message: AxisArray) -> AxisArray:
        # Update step (respects accumulate setting via child EWMAs)
        mean_message = self._state.samps_ewma(message)
        var_sq_message = self._state.vars_sq_ewma(replace(message, data=message.data**2))

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

    @ez.subscriber(BaseTransformerUnit.INPUT_SETTINGS)
    async def on_settings(self, msg: AdaptiveStandardScalerSettings) -> None:
        """
        Handle settings updates with smart reset behavior.

        Only resets state if `axis` changes (structural change).
        Changes to `time_constant` or `accumulate` are applied without
        resetting accumulated statistics.
        """
        old_axis = self.SETTINGS.axis
        self.apply_settings(msg)

        if msg.axis != old_axis:
            # Axis changed - need full reset
            self.create_processor()
        else:
            # Update accumulate on processor (propagates to child EWMAs)
            self.processor.accumulate = msg.accumulate
            # Also update own settings reference
            self.processor.settings = msg


# Convenience functions to support deprecated generator API
def scaler(time_constant: float = 1.0, axis: str | None = None) -> RiverAdaptiveStandardScalerTransformer:
    """Create a :class:`RiverAdaptiveStandardScalerTransformer` with the given parameters."""
    return RiverAdaptiveStandardScalerTransformer(
        settings=RiverAdaptiveStandardScalerSettings(time_constant=time_constant, axis=axis)
    )


def scaler_np(time_constant: float = 1.0, axis: str | None = None) -> AdaptiveStandardScalerTransformer:
    return AdaptiveStandardScalerTransformer(
        settings=AdaptiveStandardScalerSettings(time_constant=time_constant, axis=axis)
    )
