"""
Clips the data to be within the specified range.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

import ezmsg.core as ez
from array_api_compat import get_namespace
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class ClipSettings(ez.Settings):
    min: float | None = None
    """Lower clip bound. If None, no lower clipping is applied."""

    max: float | None = None
    """Upper clip bound. If None, no upper clipping is applied."""


class ClipTransformer(BaseTransformer[ClipSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        return replace(
            message,
            data=xp.clip(message.data, self.settings.min, self.settings.max),
        )


class Clip(BaseTransformerUnit[ClipSettings, AxisArray, AxisArray, ClipTransformer]):
    SETTINGS = ClipSettings


def clip(min: float | None = None, max: float | None = None) -> ClipTransformer:
    """
    Clips the data to be within the specified range.

    Args:
        min: Lower clip bound. If None, no lower clipping is applied.
        max: Upper clip bound. If None, no upper clipping is applied.

    Returns:
        :obj:`ClipTransformer`.
    """
    return ClipTransformer(ClipSettings(min=min, max=max))
