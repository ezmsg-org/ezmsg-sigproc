import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ..base import BaseTransformer, BaseTransformerUnit


class AbsSettings:
    pass


class AbsTransformer(BaseTransformer[None, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=np.abs(message.data))


class Abs(
    BaseTransformerUnit[None, AxisArray, AxisArray, AbsTransformer]
): ...  # SETTINGS = None


def abs() -> AbsTransformer:
    """
    Take the absolute value of the data. See :obj:`np.abs` for more details.

    Returns: :obj:`AbsTransformer`.

    """
    return AbsTransformer()
