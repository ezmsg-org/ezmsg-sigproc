"""Extract a named axis from an AxisArray as a standalone array."""

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray, replace

from .util.segment import SettingsConfigChain


class ExtractAxisSettings(ez.Settings):
    axis: str = "freq"
    reference: str = "time"


class ExtractAxisData(SettingsConfigChain, BaseTransformer[ExtractAxisSettings, AxisArray, AxisArray]):
    def _config_own(self, message: AxisArray) -> tuple:
        # `reference` matters only for a linear axis.
        if hasattr(message.axes[self.settings.axis], "data"):
            return (self.settings.axis,)
        return self.settings.axis, self.settings.reference

    def _process(self, message: AxisArray) -> AxisArray:
        targ_ax = message.axes[self.settings.axis]
        if hasattr(targ_ax, "data"):
            # Extracted axis is of type CoordinateAxis
            return replace(
                message,
                data=targ_ax.data,
                dims=targ_ax.dims,
                axes={k: v for k, v in message.axes.items() if k in targ_ax.dims},
                stream_dim=message.stream_dim if message.stream_dim in targ_ax.dims else None,
                config_id=self._chained_config_id(message),
            )
            # Note: So far we don't have any transformers where the coordinate axis has its own axes,
            # but if that happens in the future, we'd need to consider how to handle that.

        else:
            # Extracted axis is of type LinearAxis
            #  LinearAxis can only yield a 1d array data which simplifies dims and axes.
            n = message.data.shape[message.get_axis_idx(self.settings.reference)]
            return replace(
                message,
                data=targ_ax.value(np.arange(n)),
                dims=[self.settings.reference],
                axes={self.settings.reference: message.axes[self.settings.reference]},
                stream_dim=message.stream_dim if message.stream_dim == self.settings.reference else None,
                config_id=self._chained_config_id(message),
            )


class ExtractAxisDataUnit(BaseTransformerUnit[ExtractAxisSettings, AxisArray, AxisArray, ExtractAxisData]):
    SETTINGS = ExtractAxisSettings
