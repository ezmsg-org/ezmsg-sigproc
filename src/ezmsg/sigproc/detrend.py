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
        # Subtract the parent's (bias-corrected) EWMA estimate of the mean.
        # Reusing it keeps the baseline free of the cold-start warmup -- it is
        # the exact windowed mean from the first sample, so detrending is
        # well-behaved immediately instead of ramping up from a zero baseline.
        means = super()._process(message)
        return replace(message, data=message.data - means.data)


class DetrendUnit(BaseTransformerUnit[EWMASettings, AxisArray, AxisArray, DetrendTransformer]):
    SETTINGS = EWMASettings
