"""
Element-wise power of the data.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

import ezmsg.core as ez
from array_api_compat import get_namespace
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class PowSettings(ez.Settings):
    exponent: float = 2.0
    """The exponent to raise the data to. Default is 2.0 (squaring)."""


class PowTransformer(BaseTransformer[PowSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        return replace(message, data=xp.pow(message.data, self.settings.exponent))


class Pow(BaseTransformerUnit[PowSettings, AxisArray, AxisArray, PowTransformer]):
    SETTINGS = PowSettings


def pow(
    exponent: float = 2.0,
) -> PowTransformer:
    """
    Raise the data to an element-wise power. See :obj:`xp.pow` for more details.

    Args:
        exponent: The exponent to raise the data to. Default is 2.0.

    Returns: :obj:`PowTransformer`.

    """
    return PowTransformer(PowSettings(exponent=exponent))
