import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ..base import BaseTransformer, BaseTransformerUnit


class LogSettings(ez.Settings):
    base: float = 10.0
    """The base of the logarithm. Default is 10."""

    clip_zero: bool = False
    """If True, clip the data to the minimum positive value of the data type before taking the log."""


class LogTransformer(BaseTransformer[LogSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        data = message.data
        if (
            self.settings.clip_zero
            and np.any(data <= 0)
            and np.issubdtype(data.dtype, np.floating)
        ):
            data = np.clip(data, a_min=np.finfo(data.dtype).tiny, a_max=None)
        return replace(message, data=np.log(data) / np.log(self.settings.base))


class Log(BaseTransformerUnit[LogSettings, AxisArray, AxisArray, LogTransformer]):
    SETTINGS = LogSettings


def log(
    base: float = 10.0,
    clip_zero: bool = False,
) -> LogTransformer:
    """
    Take the logarithm of the data. See :obj:`np.log` for more details.

    Args:
        base: The base of the logarithm. Default is 10.
        clip_zero: If True, clip the data to the minimum positive value of the data type before taking the log.

    Returns: :obj:`LogTransformer`.

    """
    return LogTransformer(LogSettings(base=base, clip_zero=clip_zero))
