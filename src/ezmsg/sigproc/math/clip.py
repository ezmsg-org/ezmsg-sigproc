import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray, replace

from ..base import GenAxisArray


@consumer
def clip(a_min: float, a_max: float) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Clips the data to be within the specified range. See :obj:`np.clip` for more details.

    Args:
        a_min: Lower clip bound
        a_max: Upper clip bound

    Returns: A primed generator that, when passed an input message via `.send(msg)`, yields an :obj:`AxisArray`
     with the data payload containing the clipped version of the input :obj:`AxisArray` data.

    """
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in: AxisArray = yield msg_out
        msg_out = replace(msg_in, data=np.clip(msg_in.data, a_min, a_max))


class ClipSettings(ez.Settings):
    a_min: float
    a_max: float


class Clip(GenAxisArray):
    SETTINGS = ClipSettings

    def construct_generator(self):
        self.STATE.gen = clip(a_min=self.SETTINGS.a_min, a_max=self.SETTINGS.a_max)
