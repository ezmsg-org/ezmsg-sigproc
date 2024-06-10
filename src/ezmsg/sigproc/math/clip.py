import copy
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer, GenAxisArray
from ezmsg.util.messages.axisarray import AxisArray


@consumer
def clip(
    a_min: float,
    a_max: float
) -> typing.Generator[AxisArray, AxisArray, None]:
    msg_in = AxisArray(np.array([]), dims=[""])
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in = yield msg_out
        msg_out = copy.copy(msg_in)
        msg_out.data = np.clip(msg_out.data, a_min, a_max)


class ClipSettings(ez.Settings):
    a_min: float
    a_max: float


class Clip(GenAxisArray):
    SETTINGS = ClipSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = clip(
            a_min=self.SETTINGS.a_min,
            a_max=self.SETTINGS.a_max
        )

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> typing.AsyncGenerator:
        ret = self.STATE.gen.send(message)
        if ret is not None:
            yield self.OUTPUT_SIGNAL, ret
