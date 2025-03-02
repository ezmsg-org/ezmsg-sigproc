from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ..base import BaseTransformer, BaseTransformerUnit


class InvertTransformer(BaseTransformer[None, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=1 / message.data)


class Invert(
    BaseTransformerUnit[None, AxisArray, AxisArray, InvertTransformer]
): ...  # SETTINGS = None


def invert() -> InvertTransformer:
    """
    Take the inverse of the data.

    Returns: :obj:`InvertTransformer`.
    """
    return InvertTransformer()
