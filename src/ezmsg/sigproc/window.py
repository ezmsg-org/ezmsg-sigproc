import traceback
import typing

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
    sliding_win_oneaxis,
    replace,
)
from ezmsg.util.generator import consumer

from .base import GenAxisArray


@consumer
def windowing(
    axis: typing.Optional[str] = None,
    newaxis: str = "win",
    window_dur: typing.Optional[float] = None,
    window_shift: typing.Optional[float] = None,
    zero_pad_until: str = "input",
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Apply a sliding window along the specified axis to input streaming data.
    The `windowing` method is perhaps the most useful and versatile method in ezmsg.sigproc, but its parameterization
    can be difficult. Please read the argument descriptions carefully.

    Args:
        axis: The axis along which to segment windows.
            If None, defaults to the first dimension of the first seen AxisArray.
            Note: The windowed axis must be an AxisArray.LinearAxis, not an AxisArray.CoordinateAxis.
        newaxis: New axis on which windows are delimited, immediately
        preceding the target windowed axis. The data length along newaxis may be 0 if
        this most recent push did not provide enough data for a new window.
        If window_shift is None then the newaxis length will always be 1.
        window_dur: The duration of the window in seconds.
            If None, the function acts as a passthrough and all other parameters are ignored.
        window_shift: The shift of the window in seconds.
            If None (default), windowing operates in "1:1 mode", where each input yields exactly one most-recent window.
        zero_pad_until: Determines how the function initializes the buffer.
            Can be one of "input" (default), "full", "shift", or "none". If `window_shift` is None then this field is
            ignored and "input" is always used.

            - "input" (default) initializes the buffer with the input then prepends with zeros to the window size.
              The first input will always yield at least one output.
            - "shift" fills the buffer until `window_shift`.
              No outputs will be yielded until at least `window_shift` data has been seen.
            - "none" does not pad the buffer. No outputs will be yielded until at least `window_dur` data has been seen.

    Returns:
        A primed generator that accepts an :obj:`AxisArray` via `.send(axis_array)`
        and yields an :obj:`AxisArray` with the data payload containing a windowed version of the input data.
    """
    # Check arguments
    if newaxis is None:
        ez.logger.warning("`newaxis` must not be None. Setting to 'win'.")
        newaxis = "win"
    if window_shift is None and zero_pad_until != "input":
        ez.logger.warning(
            "`zero_pad_until` must be 'input' if `window_shift` is None. "
            f"Ignoring received argument value: {zero_pad_until}"
        )
        zero_pad_until = "input"
    elif window_shift is not None and zero_pad_until == "input":
        ez.logger.warning(
            "windowing is non-deterministic with `zero_pad_until='input'` as it depends on the size "
            "of the first input. We recommend using 'shift' when `window_shift` is float-valued."
        )
    msg_out = AxisArray(np.array([]), dims=[""])

    # State variables
    buffer: typing.Optional[npt.NDArray] = None
    window_samples: typing.Optional[int] = None
    window_shift_samples: typing.Optional[int] = None
    # Number of incoming samples to ignore. Only relevant when shift > window.:
    shift_deficit: int = 0
    b_1to1 = window_shift is None
    newaxis_warned: bool = b_1to1
    out_newaxis: typing.Optional[AxisArray.LinearAxis] = None
    out_dims: typing.Optional[typing.List[str]] = None

    check_inputs = {"samp_shape": None, "fs": None, "key": None}

    while True:
        msg_in: AxisArray = yield msg_out

        if window_dur is None:
            msg_out = msg_in
            continue

        axis = axis or msg_in.dims[0]
        axis_idx = msg_in.get_axis_idx(axis)
        axis_info = msg_in.get_axis(axis)
        fs = 1.0 / axis_info.gain

        if not newaxis_warned and newaxis in msg_in.dims:
            ez.logger.warning(
                f"newaxis {newaxis} present in input dims. Using {newaxis}_win instead"
            )
            newaxis_warned = True
            newaxis = f"{newaxis}_win"

        samp_shape = msg_in.data.shape[:axis_idx] + msg_in.data.shape[axis_idx + 1 :]

        # If buffer unset or input stats changed, create a new buffer
        b_reset = buffer is None
        b_reset = b_reset or samp_shape != check_inputs["samp_shape"]
        b_reset = b_reset or fs != check_inputs["fs"]
        b_reset = b_reset or msg_in.key != check_inputs["key"]
        if b_reset:
            # Update check variables
            check_inputs["samp_shape"] = samp_shape
            check_inputs["fs"] = fs
            check_inputs["key"] = msg_in.key

            window_samples = int(window_dur * fs)
            if not b_1to1:
                window_shift_samples = int(window_shift * fs)
            if zero_pad_until == "none":
                req_samples = window_samples
            elif zero_pad_until == "shift" and not b_1to1:
                req_samples = window_shift_samples
            else:  # i.e. zero_pad_until == "input"
                req_samples = msg_in.data.shape[axis_idx]
            n_zero = max(0, window_samples - req_samples)
            init_buffer_shape = (
                msg_in.data.shape[:axis_idx]
                + (n_zero,)
                + msg_in.data.shape[axis_idx + 1 :]
            )
            buffer = np.zeros(init_buffer_shape, dtype=msg_in.data.dtype)

        # Add new data to buffer.
        # Currently, we concatenate the new time samples and clip the output.
        # np.roll is not preferred as it returns a copy, and there's no way to construct a
        # rolling view of the data. In current numpy implementations, np.concatenate
        # is generally faster than np.roll and slicing anyway, but this could still
        # be a performance bottleneck for large memory arrays.
        # A circular buffer might be faster.
        buffer = np.concatenate((buffer, msg_in.data), axis=axis_idx)

        # Create a vector of buffer timestamps to track axis `offset` in output(s)
        buffer_offset = np.arange(buffer.shape[axis_idx]).astype(float)
        # Adjust so first _new_ sample at index 0
        buffer_offset -= buffer_offset[-msg_in.data.shape[axis_idx]]
        # Convert form indices to 'units' (probably seconds).
        buffer_offset *= axis_info.gain
        buffer_offset += axis_info.offset

        if not b_1to1 and shift_deficit > 0:
            n_skip = min(buffer.shape[axis_idx], shift_deficit)
            if n_skip > 0:
                buffer = slice_along_axis(buffer, slice(n_skip, None), axis_idx)
                buffer_offset = buffer_offset[n_skip:]
                shift_deficit -= n_skip

        # Prepare reusable parts of output
        if out_newaxis is None:
            out_dims = msg_in.dims[:axis_idx] + [newaxis] + msg_in.dims[axis_idx:]
            out_newaxis = replace(
                axis_info,
                gain=0.0 if b_1to1 else axis_info.gain * window_shift_samples,
                offset=0.0,  # offset modified per-msg below
            )

        # Generate outputs.
        # Preliminary copy of axes without the axes that we are modifying.
        out_axes = {k: v for k, v in msg_in.axes.items() if k not in [newaxis, axis]}

        # Update targeted (windowed) axis so that its offset is relative to the new axis
        # TODO: If we have `anchor_newest=True` then offset should be -win_dur
        out_axes[axis] = replace(axis_info, offset=0.0)

        # How we update .data and .axes[newaxis] depends on the windowing mode.
        if b_1to1:
            # one-to-one mode -- Each send yields exactly one window containing only the most recent samples.
            buffer = slice_along_axis(buffer, slice(-window_samples, None), axis_idx)
            out_dat = buffer.reshape(
                buffer.shape[:axis_idx] + (1,) + buffer.shape[axis_idx:]
            )
            out_newaxis = replace(out_newaxis, offset=buffer_offset[-window_samples])
        elif buffer.shape[axis_idx] >= window_samples:
            # Deterministic window shifts.
            # Note: After https://github.com/ezmsg-org/ezmsg/pull/152, add `window_shift_samples` as the last arg
            #  to `sliding_win_oneaxis` and remove the call to `slice_along_axis`.
            out_dat = sliding_win_oneaxis(buffer, window_samples, axis_idx)
            out_dat = slice_along_axis(
                out_dat, slice(None, None, window_shift_samples), axis_idx
            )
            offset_view = sliding_win_oneaxis(buffer_offset, window_samples, 0)[
                ::window_shift_samples
            ]
            out_newaxis = replace(out_newaxis, offset=offset_view[0, 0])

            # Drop expired beginning of buffer and update shift_deficit
            multi_shift = window_shift_samples * out_dat.shape[axis_idx]
            shift_deficit = max(0, multi_shift - buffer.shape[axis_idx])
            buffer = slice_along_axis(buffer, slice(multi_shift, None), axis_idx)
        else:
            # Not enough data to make a new window. Return empty data.
            empty_data_shape = (
                msg_in.data.shape[:axis_idx]
                + (0, window_samples)
                + msg_in.data.shape[axis_idx + 1 :]
            )
            out_dat = np.zeros(empty_data_shape, dtype=msg_in.data.dtype)
            # out_newaxis will have first timestamp in input... but mostly meaningless because output is size-zero.
            out_newaxis = replace(out_newaxis, offset=axis_info.offset)

        msg_out = replace(
            msg_in, data=out_dat, dims=out_dims, axes={**out_axes, newaxis: out_newaxis}
        )


class WindowSettings(ez.Settings):
    axis: typing.Optional[str] = None
    newaxis: typing.Optional[str] = None  # new axis for output. No new axes if None
    window_dur: typing.Optional[float] = None  # Sec. passthrough if None
    window_shift: typing.Optional[float] = None  # Sec. Use "1:1 mode" if None
    zero_pad_until: str = "full"  # "full", "shift", "input", "none"


class WindowState(ez.State):
    cur_settings: WindowSettings
    gen: typing.Generator


class Window(GenAxisArray):
    """:obj:`Unit` for :obj:`bandpower`."""

    SETTINGS = WindowSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = windowing(
            axis=self.SETTINGS.axis,
            newaxis=self.SETTINGS.newaxis,
            window_dur=self.SETTINGS.window_dur,
            window_shift=self.SETTINGS.window_shift,
            zero_pad_until=self.SETTINGS.zero_pad_until,
        )

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_signal(self, msg: AxisArray) -> typing.AsyncGenerator:
        try:
            out_msg = self.STATE.gen.send(msg)
            if out_msg.data.size > 0:
                if (
                    self.SETTINGS.newaxis is not None
                    or self.SETTINGS.window_dur is None
                ):
                    # Multi-win mode or pass-through mode.
                    yield self.OUTPUT_SIGNAL, out_msg
                else:
                    # We need to split out_msg into multiple yields, dropping newaxis.
                    axis_idx = out_msg.get_axis_idx("win")
                    win_axis = out_msg.axes["win"]
                    offsets = (
                        np.arange(out_msg.data.shape[axis_idx]) * win_axis.gain
                        + win_axis.offset
                    )
                    for msg_ix in range(out_msg.data.shape[axis_idx]):
                        # Need to drop 'win' and replace self.SETTINGS.axis from axes.
                        _out_axes = {
                            **{
                                k: v
                                for k, v in out_msg.axes.items()
                                if k not in ["win", self.SETTINGS.axis]
                            },
                            self.SETTINGS.axis: replace(
                                out_msg.axes[self.SETTINGS.axis], offset=offsets[msg_ix]
                            ),
                        }
                        _out_msg = replace(
                            out_msg,
                            data=slice_along_axis(out_msg.data, msg_ix, axis_idx),
                            dims=out_msg.dims[:axis_idx] + out_msg.dims[axis_idx + 1 :],
                            axes=_out_axes,
                        )
                        yield self.OUTPUT_SIGNAL, _out_msg
        except (StopIteration, GeneratorExit):
            ez.logger.debug(f"Window closed in {self.address}")
        except Exception:
            ez.logger.info(traceback.format_exc())
