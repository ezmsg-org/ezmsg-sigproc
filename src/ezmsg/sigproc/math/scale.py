"""
Scale the data by a constant factor.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ..util.segment import SettingsConfigChain


class ScaleSettings(ez.Settings):
    scale: float | int = 1.0
    """Factor by which to scale the data magnitude.

    The output dtype follows the array library's promotion of the data dtype with the value's type:
    pass an ``int`` to keep an integer stream integer, a ``float`` to get floats."""


class ScaleTransformer(SettingsConfigChain, BaseTransformer[ScaleSettings, AxisArray, AxisArray]):
    def _config_own(self, message: AxisArray) -> tuple:
        # Scaling at all can change an integer input's dtype; scale 1.0 passes the input, and its id, through.
        return ()

    def _process(self, message: AxisArray) -> AxisArray:
        if self.settings.scale == 1.0:
            return message
        return replace(message, data=self.settings.scale * message.data, config_id=self._chained_config_id(message))


class Scale(BaseTransformerUnit[ScaleSettings, AxisArray, AxisArray, ScaleTransformer]):
    SETTINGS = ScaleSettings


def scale(scale: float | int = 1.0) -> ScaleTransformer:
    """
    Scale the data by a constant factor.

    Args:
        scale: Factor by which to scale the data magnitude.

    Returns: :obj:`ScaleTransformer`

    """
    return ScaleTransformer(ScaleSettings(scale=scale))
