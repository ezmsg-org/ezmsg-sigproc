import numpy as np
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
    replace,
)
import ezmsg.core as ez

from .base import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)


class DownsampleSettings(ez.Settings):
    """
    Settings for :obj:`Downsample` node.
    """

    axis: str = "time"
    """The name of the axis along which to downsample."""

    target_rate: float | None = None
    """Desired rate after downsampling. The actual rate will be the nearest integer factor of the
            input rate that is the same or higher than the target rate."""

    factor: int | None = None
    """Explicitly specify downsample factor.  If specified, target_rate is ignored."""


@processor_state
class DownsampleState:
    q: int = 0
    """The integer downsampling factor. It will be determined based on the target rate."""

    s_idx: int = 0
    """Index of the next msg's first sample into the virtual rotating ds_factor counter."""


class DownsampleTransformer(
    BaseStatefulTransformer[DownsampleSettings, AxisArray, AxisArray, DownsampleState]
):
    """
    Downsampled data simply comprise every `factor`th sample.
    This should only be used following appropriate lowpass filtering.
    If your pipeline does not already have lowpass filtering then consider
    using the :obj:`Decimate` collection instead.
    """

    def _hash_message(self, message: AxisArray) -> int:
        return hash((message.axes[self.settings.axis].gain, message.key))

    def _reset_state(self, message: AxisArray) -> None:
        axis_info = message.get_axis(self.settings.axis)

        if self.settings.factor is not None:
            q = self.settings.factor
        elif self.settings.target_rate is None:
            q = 1
        else:
            q = int(1 / (axis_info.gain * self.settings.target_rate))
        if q < 1:
            ez.logger.warning(
                f"Target rate {self.settings.target_rate} cannot be achieved with input rate of {1 / axis_info.gain}."
                "Setting factor to 1."
            )
            q = 1
        self._state.q = q
        self._state.s_idx = 0

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis
        axis_info = message.get_axis(axis)
        axis_idx = message.get_axis_idx(axis)

        n_samples = message.data.shape[axis_idx]
        samples = (
            np.arange(self.state.s_idx, self.state.s_idx + n_samples) % self._state.q
        )
        if n_samples > 0:
            # Update state for next iteration.
            self._state.s_idx = samples[-1] + 1

        pub_samples = np.where(samples == 0)[0]
        if len(pub_samples) > 0:
            n_step = pub_samples[0].item()
            data_slice = pub_samples
        else:
            n_step = 0
            data_slice = slice(None, 0, None)
        msg_out = replace(
            message,
            data=slice_along_axis(message.data, data_slice, axis=axis_idx),
            axes={
                **message.axes,
                axis: replace(
                    axis_info,
                    gain=axis_info.gain * self._state.q,
                    offset=axis_info.offset + axis_info.gain * n_step,
                ),
            },
        )
        return msg_out


class Downsample(
    BaseTransformerUnit[DownsampleSettings, AxisArray, AxisArray, DownsampleTransformer]
):
    SETTINGS = DownsampleSettings


def downsample(
    axis: str = "time",
    target_rate: float | None = None,
    factor: int | None = None,
) -> DownsampleTransformer:
    return DownsampleTransformer(
        DownsampleSettings(axis=axis, target_rate=target_rate, factor=factor)
    )
