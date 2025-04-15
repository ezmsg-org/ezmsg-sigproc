import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ..base import BaseTransformer, BaseTransformerUnit


class ConstDifferenceSettings(ez.Settings):
    value: float = 0.0
    """number to subtract or be subtracted from the input data"""

    subtrahend: bool = True
    """If True (default) then value is subtracted from the input data. If False, the input data is subtracted from value."""


class ConstDifferenceTransformer(
    BaseTransformer[ConstDifferenceSettings, AxisArray, AxisArray]
):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(
            message,
            data=(message.data - self.settings.value)
            if self.settings.subtrahend
            else (self.settings.value - message.data),
        )


class ConstDifference(
    BaseTransformerUnit[
        ConstDifferenceSettings, AxisArray, AxisArray, ConstDifferenceTransformer
    ]
):
    SETTINGS = ConstDifferenceSettings


def const_difference(
    value: float = 0.0, subtrahend: bool = True
) -> ConstDifferenceTransformer:
    """
    result = (in_data - value) if subtrahend else (value - in_data)
    https://en.wikipedia.org/wiki/Template:Arithmetic_operations

    Args:
        value: number to subtract or be subtracted from the input data
        subtrahend: If True (default) then value is subtracted from the input data.
         If False, the input data is subtracted from value.

    Returns: :obj:`ConstDifferenceTransformer`.
    """
    return ConstDifferenceTransformer(
        ConstDifferenceSettings(value=value, subtrahend=subtrahend)
    )


# class DifferenceSettings(ez.Settings):
#     pass
#
#
# class Difference(ez.Unit):
#     SETTINGS = DifferenceSettings
#
#     INPUT_SIGNAL_1 = ez.InputStream(AxisArray)
#     INPUT_SIGNAL_2 = ez.InputStream(AxisArray)
#     OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
#
#     @ez.subscriber(INPUT_SIGNAL_2, zero_copy=True)
#     @ez.publisher(OUTPUT_SIGNAL)
#     async def on_input_2(self, message: AxisArray) -> typing.AsyncGenerator:
#         # TODO: buffer_2
#         # TODO: take buffer_1 - buffer_2 for ranges that align
#         # TODO: Drop samples from buffer_1 and buffer_2
#         if ret is not None:
#             yield self.OUTPUT_SIGNAL, ret
