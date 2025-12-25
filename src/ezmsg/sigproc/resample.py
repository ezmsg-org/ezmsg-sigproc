import asyncio
import math
import time

import ezmsg.core as ez
import numpy as np
import scipy.interpolate
from ezmsg.baseproc import (
    BaseConsumerUnit,
    BaseStatefulProcessor,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis
from ezmsg.util.messages.util import replace

from .util.axisarray_buffer import HybridAxisArrayBuffer, HybridAxisBuffer
from .util.buffer import UpdateStrategy


class ResampleSettings(ez.Settings):
    axis: str = "time"

    resample_rate: float | None = None
    """target resample rate in Hz. If None, the resample rate will be determined by the reference signal."""

    max_chunk_delay: float = np.inf
    """Maximum delay between outputs in seconds. If the delay exceeds this value, the transformer will extrapolate."""

    fill_value: str = "extrapolate"
    """
    Value to use for out-of-bounds samples.
    If 'extrapolate', the transformer will extrapolate.
    If 'last', the transformer will use the last sample.
    See scipy.interpolate.interp1d for more options.
    """

    buffer_duration: float = 2.0

    buffer_update_strategy: UpdateStrategy = "immediate"
    """
    The buffer update strategy. See :obj:`ezmsg.sigproc.util.buffer.UpdateStrategy`.
    If you expect to push data much more frequently than it is resampled, then "on_demand"
    might be more efficient. For most other scenarios, "immediate" is best.
    """


@processor_state
class ResampleState:
    src_buffer: HybridAxisArrayBuffer | None = None
    """
    Buffer for the incoming signal data. This is the source for training the interpolation function.
    Its contents are rarely empty because we usually hold back some data to allow for accurate
    interpolation and optionally extrapolation.
    """

    ref_axis_buffer: HybridAxisBuffer | None = None
    """
    The buffer for the reference axis (usually a time axis). The interpolation function
    will be evaluated at the reference axis values.
    When resample_rate is None, this buffer will be filled with the axis from incoming
    _reference_ messages.
    When resample_rate is not None (i.e., prescribed float resample_rate), this buffer
    is filled with a synthetic axis that is generated from the incoming signal messages.
    """

    last_ref_ax_val: float | None = None
    """
    The last value of the reference axis that was returned. This helps us to know
    what the _next_ returned value should be, and to avoid returning the same value.
    TODO: We can eliminate this variable if we maintain "by convention" that the
    reference axis always has 1 value at its start that we exclude from the resampling.
    """

    last_write_time: float = -np.inf
    """
    Wall clock time of the last write to the signal buffer.
    This is used to determine if we need to extrapolate the reference axis
    if we have not received an update within max_chunk_delay.
    """


class ResampleProcessor(BaseStatefulProcessor[ResampleSettings, AxisArray, AxisArray, ResampleState]):
    def _hash_message(self, message: AxisArray) -> int:
        ax_idx: int = message.get_axis_idx(self.settings.axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        ax = message.axes[self.settings.axis]
        gain = ax.gain if hasattr(ax, "gain") else None
        return hash((message.key, gain) + sample_shape)

    def _reset_state(self, message: AxisArray) -> None:
        """
        Reset the internal state based on the incoming message.
        """
        self.state.src_buffer = HybridAxisArrayBuffer(
            duration=self.settings.buffer_duration,
            axis=self.settings.axis,
            update_strategy=self.settings.buffer_update_strategy,
            overflow_strategy="grow",
        )
        if self.settings.resample_rate is not None:
            # If we are resampling at a prescribed rate, then we synthesize a reference axis
            self.state.ref_axis_buffer = HybridAxisBuffer(
                duration=self.settings.buffer_duration,
            )
            in_ax = message.axes[self.settings.axis]
            out_gain = 1 / self.settings.resample_rate
            t0 = in_ax.data[0] if hasattr(in_ax, "data") else in_ax.value(0)
            self.state.last_ref_ax_val = t0 - out_gain
        self.state.last_write_time = -np.inf

    def push_reference(self, message: AxisArray) -> None:
        ax = message.axes[self.settings.axis]
        ax_idx = message.get_axis_idx(self.settings.axis)
        if self.state.ref_axis_buffer is None:
            self.state.ref_axis_buffer = HybridAxisBuffer(
                duration=self.settings.buffer_duration,
                update_strategy=self.settings.buffer_update_strategy,
                overflow_strategy="grow",
            )
            t0 = ax.data[0] if hasattr(ax, "data") else ax.value(0)
            self.state.last_ref_ax_val = t0 - ax.gain
        self.state.ref_axis_buffer.write(ax, n_samples=message.data.shape[ax_idx])

    def _process(self, message: AxisArray) -> None:
        """
        Add a new data message to the buffer and update the reference axis if needed.
        """
        # Note: The src_buffer will copy and permute message if ax_idx != 0
        self.state.src_buffer.write(message)

        # If we are resampling at a prescribed rate (i.e., not by reference msgs),
        #  then we use this opportunity to extend our synthetic reference axis.
        ax_idx = message.get_axis_idx(self.settings.axis)
        if self.settings.resample_rate is not None and message.data.shape[ax_idx] > 0:
            in_ax = message.axes[self.settings.axis]
            in_t_end = in_ax.data[-1] if hasattr(in_ax, "data") else in_ax.value(message.data.shape[ax_idx] - 1)
            out_gain = 1 / self.settings.resample_rate
            prev_t_end = self.state.last_ref_ax_val
            n_synth = math.ceil((in_t_end - prev_t_end) * self.settings.resample_rate)
            synth_ref_axis = LinearAxis(unit="s", gain=out_gain, offset=prev_t_end + out_gain)
            self.state.ref_axis_buffer.write(synth_ref_axis, n_samples=n_synth)

        self.state.last_write_time = time.time()

    def __next__(self) -> AxisArray:
        if self.state.src_buffer is None or self.state.ref_axis_buffer is None:
            # If we have not received any data, or we require reference data
            #  that we do not yet have, then return an empty template.
            return AxisArray(data=np.array([]), dims=[""], axes={}, key="null")

        src = self.state.src_buffer
        ref = self.state.ref_axis_buffer

        # If we have no reference or the source is insufficient for interpolation
        #  then return the empty template
        if ref.is_empty() or src.available() < 3:
            src_axarr = src.peek(0)
            return replace(
                src_axarr,
                axes={
                    **src_axarr.axes,
                    self.settings.axis: ref.peek(0),
                },
            )

        # Build the reference xvec.
        #  Note: The reference axis buffer may grow upon `.peek()`
        #   as it flushes data from its deque to its buffer.
        ref_ax = ref.peek()
        if hasattr(ref_ax, "data"):
            ref_xvec = ref_ax.data
        else:
            ref_xvec = ref_ax.value(np.arange(ref.available()))

        # If we do not rely on an external reference, and we have not received new data in a while,
        #  then extrapolate our reference vector out beyond the delay limit.
        b_project = self.settings.resample_rate is not None and time.time() > (
            self.state.last_write_time + self.settings.max_chunk_delay
        )
        if b_project:
            n_append = math.ceil(self.settings.max_chunk_delay / ref_ax.gain)
            xvec_append = ref_xvec[-1] + np.arange(1, n_append + 1) * ref_ax.gain
            ref_xvec = np.hstack((ref_xvec, xvec_append))

        # Get source to train interpolation
        src_axarr = src.peek()
        src_axis = src_axarr.axes[self.settings.axis]
        x = src_axis.data if hasattr(src_axis, "data") else src_axis.value(np.arange(src_axarr.data.shape[0]))

        # Only resample at reference values that have not been interpolated over previously.
        b_ref = ref_xvec > self.state.last_ref_ax_val
        if not b_project:
            # Not extrapolating -- Do not resample beyond the end of the source buffer.
            b_ref = np.logical_and(b_ref, ref_xvec <= x[-1])
        ref_idx = np.where(b_ref)[0]

        if len(ref_idx) == 0:
            # Nothing to interpolate over; return empty data
            null_ref = replace(ref_ax, data=ref_ax.data[:0]) if hasattr(ref_ax, "data") else ref_ax
            return replace(
                src_axarr,
                data=src_axarr.data[:0, ...],
                axes={**src_axarr.axes, self.settings.axis: null_ref},
            )

        xnew = ref_xvec[ref_idx]

        # Identify source data indices around ref tvec with some padding for better interpolation.
        src_start_ix = max(0, np.where(x > xnew[0])[0][0] - 2 if np.any(x > xnew[0]) else 0)

        x = x[src_start_ix:]
        y = src_axarr.data[src_start_ix:]

        if isinstance(self.settings.fill_value, str) and self.settings.fill_value == "last":
            fill_value = (y[0], y[-1])
        else:
            fill_value = self.settings.fill_value
        f = scipy.interpolate.interp1d(
            x,
            y,
            kind="linear",
            axis=0,
            copy=False,
            bounds_error=False,
            fill_value=fill_value,
            assume_sorted=True,
        )

        # Calculate output
        resampled_data = f(xnew)

        # Create output message
        if hasattr(ref_ax, "data"):
            out_ax = replace(ref_ax, data=xnew)
        else:
            out_ax = replace(ref_ax, offset=xnew[0])
        result = replace(
            src_axarr,
            data=resampled_data,
            axes={
                **src_axarr.axes,
                self.settings.axis: out_ax,
            },
        )

        # Update the state. For state buffers, seek beyond samples that are no longer needed.
        # src: keep at least 1 sample before the final resampled value
        seek_ix = np.where(x >= xnew[-1])[0]
        if len(seek_ix) > 0:
            self.state.src_buffer.seek(max(0, src_start_ix + seek_ix[0] - 1))
        # ref: remove samples that have been sent to output
        self.state.ref_axis_buffer.seek(ref_idx[-1] + 1)
        self.state.last_ref_ax_val = xnew[-1]

        return result

    def send(self, message: AxisArray) -> AxisArray:
        self(message)
        return next(self)


class ResampleUnit(BaseConsumerUnit[ResampleSettings, AxisArray, ResampleProcessor]):
    SETTINGS = ResampleSettings
    INPUT_REFERENCE = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.subscriber(INPUT_REFERENCE, zero_copy=True)
    async def on_reference(self, message: AxisArray):
        self.processor.push_reference(message)

    @ez.publisher(OUTPUT_SIGNAL)
    async def gen_resampled(self):
        while True:
            result: AxisArray = next(self.processor)
            if np.prod(result.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, result
            else:
                await asyncio.sleep(0.001)
