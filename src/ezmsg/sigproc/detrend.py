"""Remove linear or constant trends from data along an axis."""

from ezmsg.baseproc import BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray, replace

from ezmsg.sigproc.ewma import EWMASettings, EWMATransformer


class DetrendTransformer(EWMATransformer):
    """
    Detrend the data using an exponentially weighted moving average (EWMA)
     estimate of the mean.

    The estimate is bias-corrected, so after a reset it equals the first sample and the
     output starts at zero with no step. Unlike :obj:`EWMATransformer`, a new
     :attr:`~ezmsg.util.messages.axisarray.AxisArray.segment` always resets it: the level
     after a gap need not be the level before it, and re-converging from the first
     sample costs nothing.
    """

    def _reset_segment(self, message):
        self._reset_state(message)

    def _process(self, message):
        means = super()._process(message)
        return replace(message, data=message.data - means.data)


class DetrendUnit(BaseTransformerUnit[EWMASettings, AxisArray, AxisArray, DetrendTransformer]):
    SETTINGS = EWMASettings
