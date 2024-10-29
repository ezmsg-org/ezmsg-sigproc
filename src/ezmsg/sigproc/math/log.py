from dataclasses import replace
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray

from ..base import GenAxisArray


@consumer
def log(
    base: float = 10.0,
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Take the logarithm of the data. See :obj:`np.log` for more details.

    Args:
        base: The base of the logarithm. Default is 10.

    Returns: A primed generator that, when passed an input message via `.send(msg)`, yields an :obj:`AxisArray`
     with the data payload containing the logarithm of the input :obj:`AxisArray` data.

    """
    msg_out = AxisArray(np.array([]), dims=[""])
    log_base = np.log(base)
    while True:
        msg_in: AxisArray = yield msg_out
        msg_out = replace(msg_in, data=np.log(msg_in.data) / log_base)


class LogSettings(ez.Settings):
    base: float = 10.0


class Log(GenAxisArray):
    SETTINGS = LogSettings

    def construct_generator(self):
        self.STATE.gen = log(base=self.SETTINGS.base)
