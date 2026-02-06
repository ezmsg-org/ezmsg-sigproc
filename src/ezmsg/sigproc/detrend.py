import ezmsg.core as ez
import scipy.signal as sps
from ezmsg.baseproc import BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray, replace

from ezmsg.sigproc.ewma import EWMASettings, EWMATransformer


class DetrendTransformer(EWMATransformer):
    """
    Detrend the data using an exponentially weighted moving average (EWMA)
     estimate of the mean.
    """

    def _process(self, message):
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        if self.settings.accumulate:
            means, self._state.zi = sps.lfilter(
                [self._state.alpha],
                [1.0, self._state.alpha - 1.0],
                message.data,
                axis=axis_idx,
                zi=self._state.zi,
            )
        else:
            means, _ = sps.lfilter(
                [self._state.alpha],
                [1.0, self._state.alpha - 1.0],
                message.data,
                axis=axis_idx,
                zi=self._state.zi,
            )
        return replace(message, data=message.data - means)


class DetrendUnit(BaseTransformerUnit[EWMASettings, AxisArray, AxisArray, DetrendTransformer]):
    SETTINGS = EWMASettings

    @ez.subscriber(BaseTransformerUnit.INPUT_SETTINGS)
    async def on_settings(self, msg: EWMASettings) -> None:
        """
        Handle settings updates with smart reset behavior.

        Only resets state if `axis` changes (structural change).
        Changes to `time_constant` or `accumulate` are applied without
        resetting accumulated state.
        """
        old_axis = self.SETTINGS.axis
        self.apply_settings(msg)

        if msg.axis != old_axis:
            # Axis changed - need full reset
            self.create_processor()
        else:
            # Only accumulate or time_constant changed - keep state
            self.processor.settings = msg
