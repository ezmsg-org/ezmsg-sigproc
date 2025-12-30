import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class DenormalizeSettings(ez.Settings):
    low_rate: float = 2.0
    """Low end of probable rate after denormalization (Hz)."""

    high_rate: float = 40.0
    """High end of probable rate after denormalization (Hz)."""

    distribution: str = "uniform"
    """Distribution to sample rates from. Options are 'uniform', 'normal', or 'constant'."""


@processor_state
class DenormalizeState:
    gains: npt.NDArray | None = None
    offsets: npt.NDArray | None = None


class DenormalizeTransformer(BaseStatefulTransformer[DenormalizeSettings, AxisArray, AxisArray, DenormalizeState]):
    """
    Scales data from a normalized distribution (mean=0, std=1) to a denormalized
    distribution using random per-channel offsets and gains designed to keep the
    99.9% CIs between 0 and 2x the offset.

    This is useful for simulating realistic firing rates from normalized data.
    """

    def _reset_state(self, message: AxisArray) -> None:
        ax_ix = message.get_axis_idx("ch")
        nch = message.data.shape[ax_ix]
        arr_size = (nch, 1) if ax_ix == 0 else (1, nch)
        if self.settings.distribution == "uniform":
            self.state.offsets = np.random.uniform(2.0, 40.0, size=arr_size)
        elif self.settings.distribution == "normal":
            self.state.offsets = np.random.normal(
                loc=(self.settings.low_rate + self.settings.high_rate) / 2.0,
                scale=(self.settings.high_rate - self.settings.low_rate) / 6.0,
                size=arr_size,
            )
            self.state.offsets = np.clip(
                self.state.offsets,
                a_min=self.settings.low_rate,
                a_max=self.settings.high_rate,
            )
        elif self.settings.distribution == "constant":
            self.state.offsets = np.full(
                shape=arr_size,
                fill_value=(self.settings.low_rate + self.settings.high_rate) / 2.0,
            )
        else:
            raise ValueError(f"Invalid distribution: {self.settings.distribution}")
        # Input has std == 1
        # Desired output has range from 0 to 2*self.state.offsets within 99.9% confidence interval
        # For a standard normal distribution, 99.9% of data is within +/- 3.29 std devs.
        # So, gain = offset / 3.29 to scale the std dev appropriately.
        self.state.gains = self.state.offsets / 3.29

    def _process(self, message: AxisArray) -> AxisArray:
        denorm = message.data * self.state.gains + self.state.offsets
        return replace(
            message,
            data=np.clip(denorm, a_min=0.0, a_max=None),
        )


class DenormalizeUnit(BaseTransformerUnit[DenormalizeSettings, AxisArray, AxisArray, DenormalizeTransformer]):
    SETTINGS = DenormalizeSettings
