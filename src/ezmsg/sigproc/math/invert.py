import copy
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer, GenAxisArray
from ezmsg.util.messages.axisarray import AxisArray


@consumer
def invert(
) -> typing.Generator[AxisArray, AxisArray, None]:
    msg_in = AxisArray(np.array([]), dims=[""])
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in = yield msg_out
        msg_out = copy.copy(msg_in)
        msg_out.data = 1 / msg_out.data


class InvertSettings(ez.Settings):
    pass


class Invert(GenAxisArray):
    SETTINGS = InvertSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = invert()

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> typing.AsyncGenerator:
        ret = self.STATE.gen.send(message)
        if ret is not None:
            yield self.OUTPUT_SIGNAL, ret
