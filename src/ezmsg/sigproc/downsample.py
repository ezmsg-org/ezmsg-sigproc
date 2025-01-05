import typing
from math import factorial

import numpy as np
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
    replace,
)
from ezmsg.util.generator import consumer
import ezmsg.core as ez

from .base import BaseSignalTransformer, BaseSignalTransformerUnit


class DownsampleSettings(ez.Settings):
    """
    Settings for :obj:`Downsample` node.
    See :obj:`downsample` documentation for a description of the parameters.
    """

    axis: str = "time"
    """The name of the axis along which to downsample."""

    target_rate: float | None = None
    """Desired rate after downsampling. The actual rate will be the nearest integer factor of the
            input rate that is the same or higher than the target rate."""

    factor: int | None = None
    """Explicitly specify downsample factor.  If specified, target_rate is ignored."""


class DownsampleState(ez.State):
    q: int = 0
    """The integer downsampling factor. It will be determined based on the target rate."""

    s_idx: int = 0
    """Index of the next msg's first sample into the virtual rotating ds_factor counter."""

    hash: int = 0


class DownsampleTransformer(BaseSignalTransformer[DownsampleState, AxisArray, DownsampleSettings]):
    """
    Construct a generator that yields a downsampled version of the data .send() to it.
    Downsampled data simply comprise every `factor`th sample.
    This should only be used following appropriate lowpass filtering.
    If your pipeline does not already have lowpass filtering then consider
    using the :obj:`Decimate` collection instead.

    Args:
        axis: The name of the axis along which to downsample.
            Note: The axis must exist in the message .axes and be of type AxisArray.LinearAxis.
        target_rate: Desired rate after downsampling. The actual rate will be the nearest integer factor of the
            input rate that is the same or higher than the target rate.

    Returns:
        A primed generator object ready to receive an :obj:`AxisArray` via `.send(axis_array)`
        and yields an :obj:`AxisArray` with its data downsampled.
        Note that if a send chunk does not have sufficient samples to reach the
        next downsample interval then an :obj:`AxisArray` with size-zero data is yielded.
    """
    state_type = DownsampleState

    def check_metadata(self, message: AxisArray) -> bool:
        return self.state.hash != hash((message.axes[self.settings.axis].gain, message.key))

    def reset(self, message: AxisArray) -> None:
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
        self._state.hash = hash((axis_info.gain, message.key))

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis
        axis_info = message.get_axis(axis)
        axis_idx = message.get_axis_idx(axis)

        n_samples = message.data.shape[axis_idx]
        samples = np.arange(self._state.s_idx, self.state.s_idx + n_samples) % self._state.q
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


class Downsample(BaseSignalTransformerUnit[DownsampleState, DownsampleSettings, AxisArray]):
    SETTINGS = DownsampleSettings
    transformer_type = DownsampleTransformer


# TODO: downsample = lambda axis=None, target_rate=None: iter(DownsampleTransformer(DownsampleSettings(axis=axis, target_rate=target_rate)))
#  This will require __iter__ and send methods on BaseSignalTransformer

@consumer
def downsample(
    axis: str | None = None, target_rate: float | None = None, factor: int | None = None
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Construct a generator that yields a downsampled version of the data .send() to it.
    Downsampled data simply comprise every `factor`th sample.
    This should only be used following appropriate lowpass filtering.
    If your pipeline does not already have lowpass filtering then consider
    using the :obj:`Decimate` collection instead.

    Args:
        axis: The name of the axis along which to downsample.
            Note: The axis must exist in the message .axes and be of type AxisArray.LinearAxis.
        target_rate: Desired rate after downsampling. The actual rate will be the nearest integer factor of the
            input rate that is the same or higher than the target rate.
        factor: Explicitly specify downsample factor.  If specified, target_rate is ignored.

    Returns:
        A primed generator object ready to receive an :obj:`AxisArray` via `.send(axis_array)`
        and yields an :obj:`AxisArray` with its data downsampled.
        Note that if a send chunk does not have sufficient samples to reach the
        next downsample interval then an :obj:`AxisArray` with size-zero data is yielded.

    """
    msg_out = AxisArray(np.array([]), dims=[""])

    _tx = DownsampleTransformer(DownsampleSettings(axis=axis, target_rate=target_rate, factor=factor))

    while True:
        msg_in: AxisArray = yield msg_out
        msg_out = _tx.transform(msg_in)
