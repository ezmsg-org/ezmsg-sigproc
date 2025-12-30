"""
Take the logarithm of the data.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

import ezmsg.core as ez
from array_api_compat import get_namespace
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class LogSettings(ez.Settings):
    base: float = 10.0
    """The base of the logarithm. Default is 10."""

    clip_zero: bool = False
    """If True, clip the data to the minimum positive value of the data type before taking the log."""


class LogTransformer(BaseTransformer[LogSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        data = message.data
        if self.settings.clip_zero:
            # Check if any values are <= 0 and dtype is floating point
            has_non_positive = bool(xp.any(data <= 0))
            is_floating = xp.isdtype(data.dtype, "real floating")
            if has_non_positive and is_floating:
                # Use smallest_normal (Array API equivalent of numpy's finfo.tiny)
                min_val = xp.finfo(data.dtype).smallest_normal
                data = xp.clip(data, min_val, None)
        return replace(message, data=xp.log(data) / xp.log(self.settings.base))


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
