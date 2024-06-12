import copy
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer, GenAxisArray
from ezmsg.util.messages.axisarray import AxisArray


@consumer
def const_difference(
    value: float = 0.0,
    subtrahend: bool = True
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    result = (in_data - value) if subtrahend else (value - in_data)
    https://en.wikipedia.org/wiki/Template:Arithmetic_operations
    """
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in: AxisArray = yield msg_out
        msg_out = copy.copy(msg_in)
        msg_out.data = (msg_out.data - value) if subtrahend else (value - msg_out.data)


class ConstDifferenceSettings(ez.Settings):
    value: float = 0.0
    subtrahend: bool = True


class ConstDifference(GenAxisArray):
    SETTINGS = ConstDifferenceSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = const_difference(
            value=self.SETTINGS.value,
            subtrahend=self.SETTINGS.subtrahend
        )

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> typing.AsyncGenerator:
        yield self.OUTPUT_SIGNAL, self.STATE.gen.send(message)
