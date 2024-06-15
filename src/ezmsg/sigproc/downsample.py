import copy
from dataclasses import replace
import traceback
import typing

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.generator import consumer
import ezmsg.core as ez

from .base import GenAxisArray


@consumer
def downsample(
        axis: typing.Optional[str] = None,
        factor: int = 1
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Construct a generator that yields a downsampled version of the data .send() to it.
    Downsampled data simply comprise every `factor`th sample.
    This should only be used following appropriate lowpass filtering.
    If your pipeline does not already have lowpass filtering then consider
    using the :obj:`Decimate` collection instead.

    Args:
        axis: The name of the axis along which to downsample.
        factor: Downsampling factor.

    Returns:
        A primed generator object ready to receive a `.send(axis_array)`
        and yields the downsampled data.
        Note that if a send chunk does not have sufficient samples to reach the
        next downsample interval then `None` is yielded.

    """
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])

    if factor < 1:
        raise ValueError("Downsample factor must be at least 1 (no downsampling)")

    # state variables
    s_idx: int = 0  # Index of the next msg's first sample into the virtual rotating ds_factor counter.
    template: typing.Optional[AxisArray] = None

    while True:
        axis_arr_in = yield axis_arr_out

        if axis is None:
            axis = axis_arr_in.dims[0]
        axis_info = axis_arr_in.get_axis(axis)
        axis_idx = axis_arr_in.get_axis_idx(axis)

        if template is None:
            # Reset state variables
            s_idx = 0
            # Template used as a convenient struct for holding metadata and size-zero data.
            template = copy.deepcopy(axis_arr_in)
            template.axes[axis].gain *= factor
            template.data = slice_along_axis(template.data, slice(None, 0, None), axis=axis_idx)

        n_samples = axis_arr_in.data.shape[axis_idx]
        samples = np.arange(s_idx, s_idx + n_samples) % factor
        if n_samples > 0:
            # Update state for next iteration.
            s_idx = samples[-1] + 1

        pub_samples = np.where(samples == 0)[0]
        if len(pub_samples) > 0:
            n_step = pub_samples[0].item()
            data_slice = pub_samples
        else:
            n_step = 0
            data_slice = slice(None, 0, None)
        axis_arr_out = replace(
            axis_arr_in,
            data=slice_along_axis(axis_arr_in.data, data_slice, axis=axis_idx),
            axes={
                **axis_arr_in.axes,
                axis: replace(
                    axis_info,
                    gain=axis_info.gain * factor,
                    offset=axis_info.offset + axis_info.gain * n_step
                )
            }
        )


class DownsampleSettings(ez.Settings):
    """
    Settings for :obj:`Downsample` node.
    See :obj:`downsample` documentation for a description of the parameters.
    """
    axis: typing.Optional[str] = None
    factor: int = 1


class Downsample(GenAxisArray):
    """:obj:`Unit` for :obj:`bandpower`."""
    SETTINGS: DownsampleSettings

    def construct_generator(self):
        self.STATE.gen = downsample(
            axis=self.SETTINGS.axis,
            factor=self.SETTINGS.factor
        )
