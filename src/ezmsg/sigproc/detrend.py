"""Remove linear or constant trends from data along an axis."""

from ezmsg.baseproc import BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray, replace

from ezmsg.sigproc.ewma import EWMASettings, EWMATransformer


class DetrendTransformer(EWMATransformer):
    """
    Detrend the data using an exponentially weighted moving average (EWMA)
     estimate of the mean.
    """

    def _process(self, message):
        means = super()._process(message)
        return replace(message, data=message.data - means.data)


class DetrendUnit(BaseTransformerUnit[EWMASettings, AxisArray, AxisArray, DetrendTransformer]):
    SETTINGS = EWMASettings
