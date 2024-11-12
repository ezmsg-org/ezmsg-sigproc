import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray, replace

from ..base import GenAxisArray


@consumer
def abs() -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Take the absolute value of the data. See :obj:`np.abs` for more details.

    Returns: A primed generator that, when passed an input message via `.send(msg)`, yields an :obj:`AxisArray`
     with the data payload containing the absolute value of the input :obj:`AxisArray` data.
    """
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in: AxisArray = yield msg_out
        msg_out = replace(msg_in, data=np.abs(msg_in.data))


class AbsSettings(ez.Settings):
    pass


class Abs(GenAxisArray):
    SETTINGS = AbsSettings

    def construct_generator(self):
        self.STATE.gen = abs()
