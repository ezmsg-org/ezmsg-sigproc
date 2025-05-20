import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.sigproc.base import BaseTransformer, BaseTransformerUnit


class ExtractAxisSettings(ez.Settings):
    axis: str = "freq"
    reference: str = "time"


class ExtractAxisData(BaseTransformer[ExtractAxisSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        targ_ax = message.axes[self.settings.axis]
        if hasattr(targ_ax, "data"):
            # Extracted axis is of type CoordinateAxis
            return replace(
                message,
                data=targ_ax.data,
                dims=targ_ax.dims,
                axes={k: v for k, v in message.axes.items() if k in targ_ax.dims},
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
            )


class ExtractAxisDataUnit(
    BaseTransformerUnit[ExtractAxisSettings, AxisArray, AxisArray, ExtractAxisData]
):
    SETTINGS = ExtractAxisSettings
