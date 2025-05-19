import scipy.signal as sps
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.sigproc.ewma import EWMATransformer, EWMASettings
from ezmsg.sigproc.base import BaseTransformerUnit


class DetrendTransformer(EWMATransformer):
    """
    Detrend the data using an exponentially weighted moving average (EWMA)
     estimate of the mean.
    """

    def _process(self, message):
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        means, self._state.zi = sps.lfilter(
            [self._state.alpha],
            [1.0, self._state.alpha - 1.0],
            message.data,
            axis=axis_idx,
            zi=self._state.zi,
        )
        return replace(message, data=message.data - means)


class DetrendUnit(
    BaseTransformerUnit[EWMASettings, AxisArray, AxisArray, DetrendTransformer]
):
    SETTINGS = EWMASettings
