import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray, replace

from ..base import GenAxisArray


@consumer
def invert() -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Take the inverse of the data.

    Returns: A primed generator that, when passed an input message via `.send(msg)`, yields an :obj:`AxisArray`
     with the data payload containing the inversion of the input :obj:`AxisArray` data.

    """
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in: AxisArray = yield msg_out
        msg_out = replace(msg_in, data=1 / msg_in.data)


class InvertSettings(ez.Settings):
    pass


class Invert(GenAxisArray):
    SETTINGS = InvertSettings

    def construct_generator(self):
        self.STATE.gen = invert()
