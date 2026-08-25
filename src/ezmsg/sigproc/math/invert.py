"""
Compute the multiplicative inverse (1/x) of the data.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class InvertSettings(ez.Settings):
    """This transform takes no parameters. See :obj:`AnscombeSettings` for why
    an empty settings class is needed rather than ``SETTINGS = None``."""


class InvertTransformer(BaseTransformer[InvertSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=1 / message.data)


class Invert(BaseTransformerUnit[InvertSettings, AxisArray, AxisArray, InvertTransformer]):
    SETTINGS = InvertSettings


def invert() -> InvertTransformer:
    """
    Take the inverse of the data.

    Returns: :obj:`InvertTransformer`.
    """
    return InvertTransformer()
