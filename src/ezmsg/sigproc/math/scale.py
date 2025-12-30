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


class ScaleSettings(ez.Settings):
    scale: float = 1.0
    """Factor by which to scale the data magnitude."""


class ScaleTransformer(BaseTransformer[ScaleSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=self.settings.scale * message.data)


class Scale(BaseTransformerUnit[ScaleSettings, AxisArray, AxisArray, ScaleTransformer]):
    SETTINGS = ScaleSettings


def scale(scale: float = 1.0) -> ScaleTransformer:
    """
    Scale the data by a constant factor.

    Args:
        scale: Factor by which to scale the data magnitude.

    Returns: :obj:`ScaleTransformer`

    """
    return ScaleTransformer(ScaleSettings(scale=scale))
