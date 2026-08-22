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

from ezmsg.sigproc.util.array import is_float_dtype, np_finfo


class LogSettings(ez.Settings):
    base: float = 10.0
    """The base of the logarithm. Default is 10."""

    clip_zero: bool = False
    """If True, clip the data to the minimum positive value of the data type before taking the log."""


class LogTransformer(BaseTransformer[LogSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        data = message.data
        if self.settings.clip_zero and is_float_dtype(xp, data.dtype):
            # Clip unconditionally rather than guarding on ``xp.any(data <= 0)``.
            # That guard needs the answer on the host, which on a lazy backend
            # means a full device round-trip *per message* -- and it stalls the
            # pipeline right where MLX would otherwise be running ahead. The
            # clip itself is one fused elementwise pass; measured on an M4 Pro
            # it is 4.6-6.9x cheaper than the branch it replaces (30x256 through
            # 512x1024). Clipping when nothing needed clipping is a no-op on the
            # values, so only positive subnormals change, and raising those to
            # ``smallest_normal`` is what ``clip_zero`` is for anyway.
            finfo = np_finfo(data.dtype)
            if finfo is not None:
                data = xp.clip(data, finfo.smallest_normal, None)
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
