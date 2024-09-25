from dataclasses import replace
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray

from ..base import GenAxisArray


@consumer
def scale(scale: float = 1.0) -> typing.Generator[AxisArray, AxisArray, None]:
    msg_in = AxisArray(np.array([]), dims=[""])
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in = yield msg_out
        msg_out = replace(msg_in, data=scale * msg_in.data)


class ScaleSettings(ez.Settings):
    scale: float = 1.0


class Scale(GenAxisArray):
    SETTINGS = ScaleSettings

    def construct_generator(self):
        self.STATE.gen = scale(
            scale=self.SETTINGS.scale,
        )
