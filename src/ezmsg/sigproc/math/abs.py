import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ..base import BaseTransformer, BaseTransformerUnit


class AbsSettings(ez.Settings):
    pass


class AbsTransformer(BaseTransformer[AbsSettings, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(
            message,
            data=np.abs(message.data)
        )


class Abs(BaseTransformerUnit[AbsSettings, AxisArray, AbsTransformer]):
    SETTINGS = AbsSettings


def abs() -> AbsTransformer:
    """
    Take the absolute value of the data. See :obj:`np.abs` for more details.

    Returns: :obj:`AbsTransformer`.

    """
    return AbsTransformer(AbsSettings())
