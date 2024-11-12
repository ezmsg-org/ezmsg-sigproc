import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray, replace

from ..base import GenAxisArray


@consumer
def scale(scale: float = 1.0) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Scale the data by a constant factor.

    Args:
        scale: Factor by which to scale the data magnitude.

    Returns: A primed generator that, when passed an input message via `.send(msg)`, yields an :obj:`AxisArray`
     with the data payload containing the input :obj:`AxisArray` data scaled by a constant factor.

    """
    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in: AxisArray = yield msg_out
        msg_out = replace(msg_in, data=scale * msg_in.data)


class ScaleSettings(ez.Settings):
    scale: float = 1.0


class Scale(GenAxisArray):
    SETTINGS = ScaleSettings

    def construct_generator(self):
        self.STATE.gen = scale(
            scale=self.SETTINGS.scale,
        )
