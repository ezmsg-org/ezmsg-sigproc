from dataclasses import replace
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray

from ..base import GenAxisArray


@consumer
def const_difference(
    value: float = 0.0, subtrahend: bool = True
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    result = (in_data - value) if subtrahend else (value - in_data)
    https://en.wikipedia.org/wiki/Template:Arithmetic_operations
    """
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in: AxisArray = yield msg_out
        msg_out = replace(
            msg_in, data=(msg_in.data - value) if subtrahend else (value - msg_in.data)
        )


class ConstDifferenceSettings(ez.Settings):
    value: float = 0.0
    subtrahend: bool = True


class ConstDifference(GenAxisArray):
    SETTINGS = ConstDifferenceSettings

    def construct_generator(self):
        self.STATE.gen = const_difference(
            value=self.SETTINGS.value, subtrahend=self.SETTINGS.subtrahend
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
