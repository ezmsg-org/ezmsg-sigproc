"""Adaptive standard scaling using exponentially weighted moving statistics."""

import typing

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
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


class AdaptiveStandardScalerSettings(EWMASettings):
    init_mean: npt.NDArray | float | None = None
    """Optional value used to seed the running mean on the first message,
    broadcast to the per-sample (non-``axis``) shape. When ``None`` (default)
    the mean is seeded from the first sample (matches river). Provide a known
    baseline (e.g. a training-session mean) together with ``init_std`` so a
    transient/outlier first sample does not anchor the z-score for
    ~3*``time_constant``."""

    init_std: npt.NDArray | float | None = None
    """Optional value used to seed the running standard deviation on the first
    message (paired with ``init_mean``). Seeds the second-moment EWMA to
    ``init_mean**2 + init_std**2`` so the very first output is a well-scaled
    z-score rather than 0. Ignored unless ``init_mean`` is also given."""


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
    # `update_settings` below) and `passthrough` is read live in
    # `__call__`/`__acall__`; `time_constant` and `axis` are baked into
    # the children during `_reset_state`.
    NONRESET_SETTINGS_FIELDS = frozenset({"accumulate", "passthrough"})

    def __call__(self, message: AxisArray) -> AxisArray:
        if self.settings.passthrough:
            return message
        return super().__call__(message)

    async def __acall__(self, message: AxisArray) -> AxisArray:
        if self.settings.passthrough:
            return message
        return await super().__acall__(message)

    def update_settings(self, new_settings: AdaptiveStandardScalerSettings) -> None:
        # Propagate accumulate into the existing child EWMAs before deferring
        # to the base logic, which would otherwise leave them with stale flags.
        if self._state.samps_ewma is not None and new_settings.accumulate != self.settings.accumulate:
            self.accumulate = new_settings.accumulate
        super().update_settings(new_settings)

    def _reset_state(self, message: AxisArray) -> None:
        # Optionally seed the mean and second-moment EWMAs from provided
        # statistics so a transient/outlier first sample does not anchor the
        # z-score for ~3*time_constant. E[x^2] = var + mean^2, so seed the
        # squared-EWMA with init_mean**2 + init_std**2. Both must be given.
        mean_init = self.settings.init_mean
        vars_sq_init = None
        if self.settings.init_mean is not None and self.settings.init_std is not None:
            xp = get_namespace(message.data)
            m = xp.asarray(self.settings.init_mean)
            s = xp.asarray(self.settings.init_std)
            vars_sq_init = m**2 + s**2
        self._state.samps_ewma = EWMATransformer(
            time_constant=self.settings.time_constant,
            axis=self.settings.axis,
            accumulate=self.settings.accumulate,
            init=mean_init,
        )
        self._state.vars_sq_ewma = EWMATransformer(
            time_constant=self.settings.time_constant,
            axis=self.settings.axis,
            accumulate=self.settings.accumulate,
            init=vars_sq_init,
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
        safe_varis = xp.where(mask, varis, xp.asarray(0.0, dtype=varis.dtype))
        std = safe_varis**0.5
        safe_std = xp.where(mask, std, xp.asarray(1.0, dtype=std.dtype))
        result = xp.where(
            mask, (message.data - mean_message.data) / safe_std, xp.asarray(0.0, dtype=message.data.dtype)
        )
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
