import asyncio
import dataclasses
import time
import typing

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .base import (
    BaseStatefulProcessor,
    BaseConsumerUnit,
    processor_state,
)


class ResampleSettings(ez.Settings):
    axis: str = "time"

    resample_rate: float | None = None
    """target resample rate in Hz. If None, the resample rate will be determined by the reference signal."""

    max_chunk_delay: float = 0.0
    """Maximum delay between outputs in seconds. If the delay exceeds this value, the transformer will extrapolate."""

    fill_value: str = "extrapolate"
    """
    Value to use for out-of-bounds samples.
    If 'extrapolate', the transformer will extrapolate. 
    If 'last', the transformer will use the last sample.
    See scipy.interpolate.interp1d for more options.
    """


@dataclasses.dataclass
class ResampleBuffer:
    data: npt.NDArray
    tvec: npt.NDArray
    template: AxisArray
    last_update: float


@processor_state
class ResampleState:
    signal_buffer: ResampleBuffer | None = None
    ref_axis: tuple[typing.Union[AxisArray.TimeAxis, AxisArray.CoordinateAxis], int] = (
        AxisArray.TimeAxis(fs=1.0),
        0,
    )
    last_t_out: float | None = None


class ResampleProcessor(
    BaseStatefulProcessor[ResampleSettings, AxisArray, AxisArray, ResampleState]
):
    def _hash_message(self, message: AxisArray) -> int:
        ax_idx: int = message.get_axis_idx(self.settings.axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        ax = message.axes[self.settings.axis]
        in_fs = (1 / ax.gain) if hasattr(ax, "gain") else None
        return hash((message.key, in_fs) + sample_shape)

    def _reset_state(self, message: AxisArray) -> None:
        """
        Reset the internal state based on the incoming message.
        If resample_rate is None, the output is driven by the reference signal.
        The input will still determine the template (except the primary axis) and the buffer.
        """
        ax_idx: int = message.get_axis_idx(self.settings.axis)
        ax = message.axes[self.settings.axis]
        in_dat = message.data
        in_tvec = (
            ax.data
            if hasattr(ax, "data")
            else ax.value(np.arange(in_dat.shape[ax_idx]))
        )
        if ax_idx != 0:
            in_dat = np.moveaxis(in_dat, ax_idx, 0)

        if self.settings.resample_rate is None:
            # Output is driven by input.
            # We cannot include the resampled axis until we see reference data.
            out_axes = {
                k: v for k, v in message.axes.items() if k != self.settings.axis
            }
            # last_t_out also driven by reference data.
            # self.state.last_t_out = None
        else:
            out_axes = {
                **message.axes,
                self.settings.axis: AxisArray.TimeAxis(
                    fs=self.settings.resample_rate, offset=in_tvec[0]
                ),
            }
            self.state.last_t_out = in_tvec[0] - 1 / self.settings.resample_rate
        template = replace(message, data=in_dat[:0], axes=out_axes)
        self.state.signal_buffer = ResampleBuffer(
            data=in_dat[:0],
            tvec=in_tvec[:0],
            template=template,
            last_update=time.time(),
        )

    def _process(self, message: AxisArray) -> None:
        # The incoming message will be added to the buffer.
        buf = self.state.signal_buffer

        # If our outputs are driven by reference signal, create the template's output axis if not already created.
        if (
            self.settings.resample_rate is None
            and self.settings.axis not in self.state.signal_buffer.template.axes
        ):
            buf = self.state.signal_buffer
            buf.template.axes[self.settings.axis] = self.state.ref_axis[0]
            if hasattr(buf.template.axes[self.settings.axis], "gain"):
                buf.template = replace(
                    buf.template,
                    axes={
                        **buf.template.axes,
                        self.settings.axis: replace(
                            buf.template.axes[self.settings.axis],
                            offset=self.state.last_t_out,
                        ),
                    },
                )
                # Note: last_t_out was set on the first call to push_reference.

        # Append the new data to the buffer
        ax_idx: int = message.get_axis_idx(self.settings.axis)
        in_dat: npt.NDArray = message.data
        if ax_idx != 0:
            in_dat = np.moveaxis(in_dat, ax_idx, 0)
        ax = message.axes[self.settings.axis]
        in_tvec = (
            ax.data if hasattr(ax, "data") else ax.value(np.arange(in_dat.shape[0]))
        )
        buf.data = np.concatenate((buf.data, in_dat), axis=0)
        buf.tvec = np.hstack((buf.tvec, in_tvec))
        buf.last_update = time.time()

    def push_reference(self, message: AxisArray) -> None:
        ax = message.axes[self.settings.axis]
        ax_idx = message.get_axis_idx(self.settings.axis)
        n_new = message.data.shape[ax_idx]
        if self.state.ref_axis[1] == 0:
            self.state.ref_axis = (ax, n_new)
        else:
            if hasattr(ax, "gain"):
                # Rate and offset don't need to change; we simply increment our sample counter.
                self.state.ref_axis = (
                    self.state.ref_axis[0],
                    self.state.ref_axis[1] + n_new,
                )
            else:
                # Extend our time axis with the new data.
                new_tvec = np.concatenate(
                    (self.state.ref_axis[0].data, ax.data), axis=0
                )
                self.state.ref_axis = (
                    replace(self.state.ref_axis[0], data=new_tvec),
                    self.state.ref_axis[1] + n_new,
                )

        if self.settings.resample_rate is None and self.state.last_t_out is None:
            # This reference axis will become THE output axis.
            # If last_t_out has not previously been set, we set it to the sample before this reference data.
            if hasattr(self.state.ref_axis[0], "gain"):
                ref_tvec = self.state.ref_axis[0].value(np.arange(2))
            else:
                ref_tvec = self.state.ref_axis[0].data[:2]
            self.state.last_t_out = 2 * ref_tvec[0] - ref_tvec[1]

    def __next__(self) -> AxisArray:
        buf = self.state.signal_buffer

        if buf is None:
            return AxisArray(data=np.array([]), dims=[""], axes={}, key="null")

        # buffer is empty or ref-driven && empty-reference; return the empty template
        if (buf.tvec.size == 0) or (
            self.settings.resample_rate is None and self.state.ref_axis[1] < 3
        ):
            # Note: empty template's primary axis' offset might be meaningless.
            return buf.template

        # Identify the output timestamps at which we will resample the buffer
        b_project = False
        if self.settings.resample_rate is None:
            # Rely on reference signal to determine output timestamps
            if hasattr(self.state.ref_axis[0], "data"):
                ref_tvec = self.state.ref_axis[0].data
            else:
                n_avail = self.state.ref_axis[1]
                ref_tvec = self.state.ref_axis[0].value(np.arange(n_avail))
        else:
            # Get output timestamps from resample_rate and what we've collected so far
            t_begin = self.state.last_t_out + 1 / self.settings.resample_rate
            t_end = buf.tvec[-1]
            if self.settings.max_chunk_delay > 0 and time.time() > (
                buf.last_update + self.settings.max_chunk_delay
            ):
                # We've waiting too long between pushes. We will have to extrapolate.
                b_project = True
                t_end += self.settings.max_chunk_delay
            ref_tvec = np.arange(t_begin, t_end, 1 / self.settings.resample_rate)

        # Which samples can we resample?
        b_ref = ref_tvec > self.state.last_t_out
        if not b_project:
            b_ref = np.logical_and(b_ref, ref_tvec <= buf.tvec[-1])
        ref_idx = np.where(b_ref)[0]

        if len(ref_idx) < 2:
            # Not enough data to resample; return the empty template.
            return buf.template

        tnew = ref_tvec[ref_idx]
        # Slice buf to minimal range around tnew with some padding for better interpolation.
        buf_start_ix = max(0, np.searchsorted(buf.tvec, tnew[0]) - 2)
        buf_stop_ix = np.searchsorted(buf.tvec, tnew[-1], side="right") + 2
        x = buf.tvec[buf_start_ix:buf_stop_ix]
        y = buf.data[buf_start_ix:buf_stop_ix]
        if (
            isinstance(self.settings.fill_value, str)
            and self.settings.fill_value == "last"
        ):
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
        resampled_data = f(tnew)
        if hasattr(buf.template.axes[self.settings.axis], "data"):
            repl_axis = replace(buf.template.axes[self.settings.axis], data=tnew)
        else:
            repl_axis = replace(buf.template.axes[self.settings.axis], offset=tnew[0])
        result = replace(
            buf.template,
            data=resampled_data,
            axes={
                **buf.template.axes,
                self.settings.axis: repl_axis,
            },
        )

        # Update state to move past samples that are no longer be needed
        self.state.last_t_out = tnew[-1]
        buf.data = buf.data[max(0, buf_stop_ix - 3) :]
        buf.tvec = buf.tvec[max(0, buf_stop_ix - 3) :]
        buf.last_update = time.time()

        if self.settings.resample_rate is None:
            # Update self.state.ref_axis to remove samples that have been used in the output
            if hasattr(self.state.ref_axis[0], "data"):
                new_ref_ax = replace(
                    self.state.ref_axis[0],
                    data=self.state.ref_axis[0].data[ref_idx[-1] + 1 :],
                )
            else:
                next_offset = self.state.ref_axis[0].value(ref_idx[-1] + 1)
                new_ref_ax = replace(self.state.ref_axis[0], offset=next_offset)
            self.state.ref_axis = (new_ref_ax, self.state.ref_axis[1] - len(ref_idx))

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
