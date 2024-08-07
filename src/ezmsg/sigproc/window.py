from dataclasses import replace
import traceback
from typing import AsyncGenerator, Optional, Tuple, List, Generator

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt

from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis, sliding_win_oneaxis
from ezmsg.util.generator import consumer

from .base import GenAxisArray


@consumer
def windowing(
    axis: Optional[str] = None,
    newaxis: str = "win",
    window_dur: Optional[float] = None,
    window_shift: Optional[float] = None,
    zero_pad_until: str = "input"
) -> Generator[AxisArray, AxisArray, None]:
    """
    Construct a generator that yields windows of data from an input :obj:`AxisArray`.

    Args:
        axis: The axis along which to segment windows.
            If None, defaults to the first dimension of the first seen AxisArray.
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
        A (primed) generator that accepts .send(an AxisArray object) and yields a list of windowed
        AxisArray objects. The list will always be length-1 if `newaxis` is not None or `window_shift` is None.
    """
    if newaxis is None:
        ez.logger.warning("`newaxis` must not be None. Setting to 'win'.")
        newaxis = "win"
    if window_shift is None and zero_pad_until != "input":
        ez.logger.warning("`zero_pad_until` must be 'input' if `window_shift` is None. "
                          f"Ignoring received argument value: {zero_pad_until}")
        zero_pad_until = "input"
    elif window_shift is not None and zero_pad_until == "input":
        ez.logger.warning("windowing is non-deterministic with `zero_pad_until='input'` as it depends on the size "
                          "of the first input. We recommend using 'shift' when `window_shift` is float-valued.")
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])

    # State variables
    prev_samp_shape: Optional[Tuple[int, ...]] = None
    prev_fs: Optional[float] = None
    buffer: Optional[npt.NDArray] = None
    window_samples: Optional[int] = None
    window_shift_samples: Optional[int] = None
    shift_deficit: int = 0  # Number of incoming samples to ignore. Only relevant when shift > window.
    b_1to1 = window_shift is None
    newaxis_warned: bool = b_1to1
    out_newaxis: Optional[AxisArray.Axis] = None

    while True:
        axis_arr_in = yield axis_arr_out

        if window_dur is None:
            axis_arr_out = axis_arr_in
            continue

        if axis is None:
            axis = axis_arr_in.dims[0]
        axis_idx = axis_arr_in.get_axis_idx(axis)
        axis_info = axis_arr_in.get_axis(axis)
        fs = 1.0 / axis_info.gain

        if not newaxis_warned and newaxis in axis_arr_in.dims:
            ez.logger.warning(f"newaxis {newaxis} present in input dims. Using {newaxis}_win instead")
            newaxis_warned = True
            newaxis = f"{newaxis}_win"

        samp_shape = axis_arr_in.data.shape[:axis_idx] + axis_arr_in.data.shape[axis_idx + 1:]

        # If buffer unset or input stats changed, create a new buffer
        if buffer is None or samp_shape != prev_samp_shape or fs != prev_fs:
            window_samples = int(window_dur * fs)
            if not b_1to1:
                window_shift_samples = int(window_shift * fs)
            if zero_pad_until == "none":
                req_samples = window_samples
            elif zero_pad_until == "shift" and not b_1to1:
                req_samples = window_shift_samples
            else:  # i.e. zero_pad_until == "input"
                req_samples = axis_arr_in.data.shape[axis_idx]
            n_zero = max(0, window_samples - req_samples)
            buffer = np.zeros(axis_arr_in.data.shape[:axis_idx] + (n_zero,) + axis_arr_in.data.shape[axis_idx + 1:])
            prev_samp_shape = samp_shape
            prev_fs = fs

        # Add new data to buffer.
        # Currently, we concatenate the new time samples and clip the output.
        # np.roll is not preferred as it returns a copy, and there's no way to construct a
        # rolling view of the data. In current numpy implementations, np.concatenate
        # is generally faster than np.roll and slicing anyway, but this could still
        # be a performance bottleneck for large memory arrays.
        # A circular buffer might be faster.
        buffer = np.concatenate((buffer, axis_arr_in.data), axis=axis_idx)

        # Create a vector of buffer timestamps to track axis `offset` in output(s)
        buffer_offset = np.arange(buffer.shape[axis_idx]).astype(float)
        # Adjust so first _new_ sample at index 0
        buffer_offset -= buffer_offset[-axis_arr_in.data.shape[axis_idx]]
        # Convert form indices to 'units' (probably seconds).
        buffer_offset *= axis_info.gain
        buffer_offset += axis_info.offset

        if not b_1to1 and shift_deficit > 0:
            n_skip = min(buffer.shape[axis_idx], shift_deficit)
            if n_skip > 0:
                buffer = slice_along_axis(buffer, np.s_[n_skip:], axis_idx)
                buffer_offset = buffer_offset[n_skip:]
                shift_deficit -= n_skip

        # Prepare reusable parts of output
        if out_newaxis is None:
            out_dims = axis_arr_in.dims[:axis_idx] + [newaxis] + axis_arr_in.dims[axis_idx:]
            out_newaxis = replace(
                axis_info,
                gain=0.0 if b_1to1 else axis_info.gain * window_shift_samples,
                offset=0.0  # offset modified per-msg below
            )

        # Generate outputs.
        # Preliminary copy of axes without the axes that we are modifying.
        out_axes = {k: v for k, v in axis_arr_in.axes.items() if k not in [newaxis, axis]}

        # Update targeted (windowed) axis so that its offset is relative to the new axis
        # TODO: If we have `anchor_newest=True` then offset should be -win_dur
        out_axes[axis] = replace(axis_info, offset=0.0)

        # How we update .data and .axes[newaxis] depends on the windowing mode.
        if b_1to1:
            # one-to-one mode -- Each send yields exactly one window containing only the most recent samples.
            buffer = slice_along_axis(buffer, np.s_[-window_samples:], axis_idx)
            out_dat = np.expand_dims(buffer, axis=axis_idx)
            out_newaxis = replace(out_newaxis, offset=buffer_offset[-window_samples])
        elif buffer.shape[axis_idx] >= window_samples:
            # Deterministic window shifts.
            out_dat = sliding_win_oneaxis(buffer, window_samples, axis_idx)
            out_dat = slice_along_axis(out_dat, np.s_[::window_shift_samples], axis_idx)
            offset_view = sliding_win_oneaxis(buffer_offset, window_samples, 0)[::window_shift_samples]
            out_newaxis = replace(out_newaxis, offset=offset_view[0, 0])

            # Drop expired beginning of buffer and update shift_deficit
            multi_shift = window_shift_samples * out_dat.shape[axis_idx]
            shift_deficit = max(0, multi_shift - buffer.shape[axis_idx])
            buffer = slice_along_axis(buffer, np.s_[multi_shift:], axis_idx)
        else:
            # Not enough data to make a new window. Return empty data.
            empty_data_shape = (
                    axis_arr_in.data.shape[:axis_idx]
                    + (0, window_samples)
                    + axis_arr_in.data.shape[axis_idx + 1:]
            )
            out_dat = np.zeros(empty_data_shape, dtype=axis_arr_in.data.dtype)
            # out_newaxis will have first timestamp in input... but mostly meaningless because output is size-zero.
            out_newaxis = replace(out_newaxis, offset=axis_info.offset)

        axis_arr_out = replace(
            axis_arr_in,
            data=out_dat,
            dims=out_dims,
            axes={
                **out_axes,
                newaxis: out_newaxis
            }
        )


class WindowSettings(ez.Settings):
    axis: Optional[str] = None
    newaxis: Optional[str] = None  # new axis for output. No new axes if None
    window_dur: Optional[float] = None  # Sec. passthrough if None
    window_shift: Optional[float] = None  # Sec. Use "1:1 mode" if None
    zero_pad_until: str = "full"  # "full", "shift", "input", "none"


class WindowState(ez.State):
    cur_settings: WindowSettings
    gen: Generator


class Window(GenAxisArray):
    """:obj:`Unit` for :obj:`bandpower`."""
    SETTINGS: WindowSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = windowing(
            axis=self.SETTINGS.axis,
            newaxis=self.SETTINGS.newaxis,
            window_dur=self.SETTINGS.window_dur,
            window_shift=self.SETTINGS.window_shift,
            zero_pad_until=self.SETTINGS.zero_pad_until
        )

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_signal(self, msg: AxisArray) -> AsyncGenerator:
        try:
            out_msg = self.STATE.gen.send(msg)
            if out_msg.data.size > 0:
                if self.SETTINGS.newaxis is not None or self.SETTINGS.window_dur is None:
                    # Multi-win mode or pass-through mode.
                    yield self.OUTPUT_SIGNAL, out_msg
                else:
                    # We need to split out_msg into multiple yields, dropping newaxis.
                    axis_idx = out_msg.get_axis_idx("win")
                    win_axis = out_msg.axes["win"]
                    offsets = np.arange(out_msg.data.shape[axis_idx]) * win_axis.gain + win_axis.offset
                    for msg_ix in range(out_msg.data.shape[axis_idx]):
                        # Need to drop 'win' and replace self.SETTINGS.axis from axes.
                        _out_axes = {
                            **{k: v for k, v in out_msg.axes.items() if k not in ["win", self.SETTINGS.axis]},
                            self.SETTINGS.axis: replace(
                                out_msg.axes[self.SETTINGS.axis],
                                offset=offsets[msg_ix]
                            )
                        }
                        _out_msg = replace(
                            out_msg,
                            data=slice_along_axis(out_msg.data, msg_ix, axis_idx),
                            dims=out_msg.dims[:axis_idx] + out_msg.dims[axis_idx + 1:],
                            axes=_out_axes
                        )
                        yield self.OUTPUT_SIGNAL, _out_msg
        except (StopIteration, GeneratorExit):
            ez.logger.debug(f"Window closed in {self.address}")
        except Exception:
            ez.logger.info(traceback.format_exc())
