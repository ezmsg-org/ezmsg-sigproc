from dataclasses import replace
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray

from ..base import GenAxisArray


@consumer
def clip(a_min: float, a_max: float) -> typing.Generator[AxisArray, AxisArray, None]:
    msg_in = AxisArray(np.array([]), dims=[""])
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in = yield msg_out
        msg_out = replace(msg_in, data=np.clip(msg_in.data, a_min, a_max))


class ClipSettings(ez.Settings):
    a_min: float
    a_max: float


class Clip(GenAxisArray):
    SETTINGS = ClipSettings

    def construct_generator(self):
        self.STATE.gen = clip(a_min=self.SETTINGS.a_min, a_max=self.SETTINGS.a_max)
