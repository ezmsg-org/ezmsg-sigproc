"""
Element-wise power of the data.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class PowSettings(ez.Settings):
    exponent: float | int = 2.0
    """The exponent to raise the data to. Default is 2.0 (squaring).

    The output dtype follows the array library's promotion of the data dtype with the value's type:
    pass an ``int`` to keep an integer stream integer, a ``float`` to get floats.
    An integer stream cannot be raised to a negative ``int``."""


class PowTransformer(BaseTransformer[PowSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=message.data**self.settings.exponent)


class Pow(BaseTransformerUnit[PowSettings, AxisArray, AxisArray, PowTransformer]):
    SETTINGS = PowSettings


def pow(
    exponent: float | int = 2.0,
) -> PowTransformer:
    """
    Raise the data to an element-wise power. See :obj:`xp.pow` for more details.

    Args:
        exponent: The exponent to raise the data to. Default is 2.0.

    Returns: :obj:`PowTransformer`.

    """
    return PowTransformer(PowSettings(exponent=exponent))
