from dataclasses import replace
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray

from ..base import GenAxisArray


@consumer
def abs() -> typing.Generator[AxisArray, AxisArray, None]:
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in = yield msg_out
        msg_out = replace(msg_in, data=np.abs(msg_in.data))


class AbsSettings(ez.Settings):
    pass


class Abs(GenAxisArray):
    SETTINGS = AbsSettings

    def construct_generator(self):
        self.STATE.gen = abs()
