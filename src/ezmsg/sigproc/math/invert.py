import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ..base import BaseTransformer, BaseTransformerUnit


class InvertSettings(ez.Settings):
    pass


class InvertTransformer(BaseTransformer[InvertSettings, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=1 / message.data)


class Invert(BaseTransformerUnit[InvertSettings, AxisArray, InvertTransformer]):
    SETTINGS = InvertSettings


def invert() -> InvertTransformer:
    """
    Take the inverse of the data.

    Returns: :obj:`InvertTransformer`.
    """
    return InvertTransformer(InvertSettings())
