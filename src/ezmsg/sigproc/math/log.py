import copy
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer, GenAxisArray
from ezmsg.util.messages.axisarray import AxisArray


@consumer
def log(
    base: float = 10.0,
) -> typing.Generator[AxisArray, AxisArray, None]:
    msg_in = AxisArray(np.array([]), dims=[""])
    msg_out = AxisArray(np.array([]), dims=[""])
    log_base = np.log(base)
    while True:
        msg_in = yield msg_out
        msg_out = copy.copy(msg_in)
        msg_out.data = np.log(msg_out.data) / log_base


class LogSettings(ez.Settings):
    base: float = 10.0


class Log(GenAxisArray):
    SETTINGS = LogSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = log(
            base=self.SETTINGS.base
        )

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> typing.AsyncGenerator:
        ret = self.STATE.gen.send(message)
        if ret is not None:
            yield self.OUTPUT_SIGNAL, ret
