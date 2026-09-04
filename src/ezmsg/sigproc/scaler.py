"""Adaptive standard scaling using exponentially weighted moving statistics."""

import typing

import ezmsg.core as ez
import numpy as np
from array_api_compat import get_namespace
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
    # `accumulate` can be live-propagated into the child EWMAs (see
    # `update_settings` below) and `passthrough`/`reset_on_resume` are read live
    # in `__call__`/`__acall__`; `time_constant` and `axis` are baked into
    # the children during `_reset_state`.
    NONRESET_SETTINGS_FIELDS = frozenset({"accumulate", "passthrough", "reset_on_resume"})

    def _skip(self, message: AxisArray) -> bool:
        """Whether to bypass scaling, flagging a reset if this is a real gap.

        See :obj:`EWMASettings.reset_on_resume`. A reset here is enough for both
        children: `_reset_state` rebuilds them from scratch.
        """
        if not self.settings.passthrough:
            return False
        if self.settings.reset_on_resume and np.prod(message.data.shape) != 0:
            self._request_reset()
        return True

    def __call__(self, message: AxisArray) -> AxisArray:
        if self._skip(message):
            return message
        return super().__call__(message)

    async def __acall__(self, message: AxisArray) -> AxisArray:
        if self._skip(message):
            return message
        return await super().__acall__(message)

    def update_settings(self, new_settings: AdaptiveStandardScalerSettings) -> None:
        # Propagate accumulate into the existing child EWMAs before deferring
        # to the base logic, which would otherwise leave them with stale flags.
        if self._state.samps_ewma is not None and new_settings.accumulate != self.settings.accumulate:
            self.accumulate = new_settings.accumulate
        super().update_settings(new_settings)

    def _hash_message(self, message: AxisArray) -> int:
        # This transformer owns no array state of its own -- only the two child
        # EWMATransformers, which hash the message themselves and rebuild on
        # exactly the changes that matter. Hashing here would duplicate that work
        # and, on a reset, discard the children wholesale rather than letting
        # each rebuild the part of its state that actually went stale.
        return 0

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
        xp = get_namespace(message.data)

        # Update step (respects accumulate setting via child EWMAs)
        mean_message = self._state.samps_ewma(message)
        var_sq_message = self._state.vars_sq_ewma(replace(message, data=message.data**2))

        # Get step: safe division avoids warnings from zero/negative variance
        varis = var_sq_message.data - mean_message.data**2
        mask = varis > 0
        # Python scalars rather than ``xp.asarray(0.0, dtype=...)``: a wrapped
        # scalar is a real array that has to be built (and, on a GPU backend,
        # shipped) every message, and it drags the operands through the full
        # promotion rules instead of the weak-scalar ones. 1.09-1.32x on MLX,
        # M4 Pro, 30x256 through 512x1024.
        safe_varis = xp.where(mask, varis, 0.0)
        safe_std = xp.where(mask, xp.sqrt(safe_varis), 1.0)
        result = xp.where(mask, (message.data - mean_message.data) / safe_std, 0.0)
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

    INPUT_ACCUMULATE = ez.InputStream(bool)

    @ez.subscriber(INPUT_ACCUMULATE)
    async def on_accumulate(self, accumulate: bool) -> None:
        self.processor.accumulate = accumulate


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
