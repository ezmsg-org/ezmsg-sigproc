"""
Take the absolute value of the data.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

from array_api_compat import get_namespace
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class AbsSettings:
    pass


class AbsTransformer(BaseTransformer[None, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        return replace(message, data=xp.abs(message.data))


class Abs(BaseTransformerUnit[None, AxisArray, AxisArray, AbsTransformer]): ...  # SETTINGS = None


def abs() -> AbsTransformer:
    """
    Take the absolute value of the data. See :obj:`np.abs` for more details.

    Returns: :obj:`AbsTransformer`.

    """
    return AbsTransformer()
