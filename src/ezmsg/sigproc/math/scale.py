import copy
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer, GenAxisArray
from ezmsg.util.messages.axisarray import AxisArray


@consumer
def scale(
    scale: float = 1.0
) -> typing.Generator[AxisArray, AxisArray, None]:
    msg_in = AxisArray(np.array([]), dims=[""])
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in = yield msg_out
        msg_out = copy.copy(msg_in)
        msg_out.data = scale * msg_in.data


class ScaleSettings(ez.Settings):
    scale: float = 1.0


class Scale(GenAxisArray):
    SETTINGS = ScaleSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = scale(
            scale=self.SETTINGS.scale,
        )

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> typing.AsyncGenerator:
        ret = self.STATE.gen.send(message)
        if ret is not None:
            yield self.OUTPUT_SIGNAL, ret
