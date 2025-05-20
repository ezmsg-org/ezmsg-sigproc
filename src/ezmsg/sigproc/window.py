import enum
import traceback
import typing

import ezmsg.core as ez
import numpy.typing as npt
import sparse
from array_api_compat import is_pydata_sparse_namespace, get_namespace
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
    sliding_win_oneaxis,
    replace,
)

from .base import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from .util.sparse import sliding_win_oneaxis as sparse_sliding_win_oneaxis
from .util.profile import profile_subpub


class Anchor(enum.Enum):
    BEGINNING = "beginning"
    END = "end"
    MIDDLE = "middle"


class WindowSettings(ez.Settings):
    axis: str | None = None
    newaxis: str | None = None  # new axis for output. No new axes if None
    window_dur: float | None = None  # Sec. passthrough if None
    window_shift: float | None = None  # Sec. Use "1:1 mode" if None
    zero_pad_until: str = "full"  # "full", "shift", "input", "none"
    anchor: str | Anchor = Anchor.BEGINNING


@processor_state
class WindowState:
    buffer: npt.NDArray | sparse.SparseArray | None = None

    window_samples: int | None = None

    window_shift_samples: int | None = None

    shift_deficit: int = 0
    """ Number of incoming samples to ignore. Only relevant when shift > window."""

    newaxis_warned: bool = False

    out_newaxis: AxisArray.LinearAxis | None = None

    out_dims: list[str] | None = None


class WindowTransformer(
    BaseStatefulTransformer[WindowSettings, AxisArray, AxisArray, WindowState]
):
    """
    Apply a sliding window along the specified axis to input streaming data.
    The `windowing` method is perhaps the most useful and versatile method in ezmsg.sigproc, but its parameterization
    can be difficult. Please read the argument descriptions carefully.
    """

    def __init__(self, *args, **kwargs) -> None:
        """

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
                If None (default), windowing operates in "1:1 mode",
                where each input yields exactly one most-recent window.
            zero_pad_until: Determines how the function initializes the buffer.
                Can be one of "input" (default), "full", "shift", or "none".
                If `window_shift` is None then this field is ignored and "input" is always used.

                - "input" (default) initializes the buffer with the input then prepends with zeros to the window size.
                  The first input will always yield at least one output.
                - "shift" fills the buffer until `window_shift`.
                  No outputs will be yielded until at least `window_shift` data has been seen.
                - "none" does not pad the buffer. No outputs will be yielded until
                  at least `window_dur` data has been seen.
            anchor: Determines the entry in `axis` that gets assigned `0`, which references the
                value in `newaxis`. Can be of class :obj:`Anchor` or a string representation of an :obj:`Anchor`.
        """
        super().__init__(*args, **kwargs)

        # Sanity-check settings
        # if self.settings.newaxis is None:
        #     ez.logger.warning("`newaxis=None` will be replaced with `newaxis='win'`.")
        #     object.__setattr__(self.settings, "newaxis", "win")
        if (
            self.settings.window_shift is None
            and self.settings.zero_pad_until != "input"
        ):
            ez.logger.warning(
                "`zero_pad_until` must be 'input' if `window_shift` is None. "
                f"Ignoring received argument value: {self.settings.zero_pad_until}"
            )
            object.__setattr__(self.settings, "zero_pad_until", "input")
        elif (
            self.settings.window_shift is not None
            and self.settings.zero_pad_until == "input"
        ):
            ez.logger.warning(
                "windowing is non-deterministic with `zero_pad_until='input'` as it depends on the size "
                "of the first input. We recommend using `zero_pad_until='shift'` when `window_shift` is float-valued."
            )
        try:
            object.__setattr__(self.settings, "anchor", Anchor(self.settings.anchor))
        except ValueError:
            raise ValueError(
                f"Invalid anchor: {self.settings.anchor}. Valid anchor are: {', '.join([e.value for e in Anchor])}"
            )

    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        axis_info = message.get_axis(axis)
        fs = 1.0 / axis_info.gain
        samp_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]

        return hash(samp_shape + (fs, message.key))

    def _reset_state(self, message: AxisArray) -> None:
        _newaxis = self.settings.newaxis or "win"
        if not self._state.newaxis_warned and _newaxis in message.dims:
            ez.logger.warning(
                f"newaxis {_newaxis} present in input dims. Using {_newaxis}_win instead"
            )
            self._state.newaxis_warned = True
            self.settings.newaxis = f"{_newaxis}_win"

        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        axis_info = message.get_axis(axis)
        fs = 1.0 / axis_info.gain

        xp = get_namespace(message.data)

        self._state.window_samples = int(self.settings.window_dur * fs)
        if self.settings.window_shift is not None:
            # If window_shift is None, we are in "1:1 mode" and window_shift_samples is not used.
            self._state.window_shift_samples = int(self.settings.window_shift * fs)
        if self.settings.zero_pad_until == "none":
            req_samples = self._state.window_samples
        elif (
            self.settings.zero_pad_until == "shift"
            and self.settings.window_shift is not None
        ):
            req_samples = self._state.window_shift_samples
        else:  # i.e. zero_pad_until == "input"
            req_samples = message.data.shape[axis_idx]
        n_zero = max(0, self._state.window_samples - req_samples)
        init_buffer_shape = (
            message.data.shape[:axis_idx]
            + (n_zero,)
            + message.data.shape[axis_idx + 1 :]
        )
        self._state.buffer = xp.zeros(init_buffer_shape, dtype=message.data.dtype)

        # Prepare reusable parts of output
        if self._state.out_newaxis is None:
            self._state.out_dims = (
                list(message.dims[:axis_idx])
                + [_newaxis]
                + list(message.dims[axis_idx:])
            )
            self._state.out_newaxis = replace(
                axis_info,
                gain=0.0
                if self.settings.window_shift is None
                else axis_info.gain * self._state.window_shift_samples,
                offset=0.0,  # offset modified per-msg below
            )

    def __call__(self, message: AxisArray) -> AxisArray:
        if self.settings.window_dur is None:
            # Shortcut for no windowing
            return message
        return super().__call__(message)

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        axis_info = message.get_axis(axis)

        xp = get_namespace(message.data)

        # Add new data to buffer.
        # Currently, we concatenate the new time samples and clip the output.
        # np.roll is not preferred as it returns a copy, and there's no way to construct a
        # rolling view of the data. In current numpy implementations, np.concatenate
        # is generally faster than np.roll and slicing anyway, but this could still
        # be a performance bottleneck for large memory arrays.
        # A circular buffer might be faster.
        self._state.buffer = xp.concatenate(
            (self._state.buffer, message.data), axis=axis_idx
        )

        # Create a vector of buffer timestamps to track axis `offset` in output(s)
        buffer_tvec = xp.asarray(range(self._state.buffer.shape[axis_idx]), dtype=float)

        # Adjust so first _new_ sample at index 0.
        buffer_tvec -= buffer_tvec[-message.data.shape[axis_idx]]
        # Convert form indices to 'units' (probably seconds).
        buffer_tvec *= axis_info.gain
        buffer_tvec += axis_info.offset

        if self.settings.window_shift is not None and self._state.shift_deficit > 0:
            n_skip = min(self._state.buffer.shape[axis_idx], self._state.shift_deficit)
            if n_skip > 0:
                self._state.buffer = slice_along_axis(
                    self._state.buffer, slice(n_skip, None), axis_idx
                )
                buffer_tvec = buffer_tvec[n_skip:]
                self._state.shift_deficit -= n_skip

        # Generate outputs.
        # Preliminary copy of axes without the axes that we are modifying.
        _newaxis = self.settings.newaxis or "win"
        out_axes = {k: v for k, v in message.axes.items() if k not in [_newaxis, axis]}

        # Update targeted (windowed) axis so that its offset is relative to the new axis
        if self.settings.anchor == Anchor.BEGINNING:
            out_axes[axis] = replace(axis_info, offset=0.0)
        elif self.settings.anchor == Anchor.END:
            out_axes[axis] = replace(axis_info, offset=-self.settings.window_dur)
        elif self.settings.anchor == Anchor.MIDDLE:
            out_axes[axis] = replace(axis_info, offset=-self.settings.window_dur / 2)

        # How we update .data and .axes[newaxis] depends on the windowing mode.
        if self.settings.window_shift is None:
            # one-to-one mode -- Each send yields exactly one window containing only the most recent samples.
            self._state.buffer = slice_along_axis(
                self._state.buffer, slice(-self._state.window_samples, None), axis_idx
            )
            out_dat = self._state.buffer.reshape(
                self._state.buffer.shape[:axis_idx]
                + (1,)
                + self._state.buffer.shape[axis_idx:]
            )
            win_offset = buffer_tvec[-self._state.window_samples]
        elif self._state.buffer.shape[axis_idx] >= self._state.window_samples:
            # Deterministic window shifts.
            sliding_win_fun = (
                sparse_sliding_win_oneaxis
                if is_pydata_sparse_namespace(xp)
                else sliding_win_oneaxis
            )
            out_dat = sliding_win_fun(
                self._state.buffer,
                self._state.window_samples,
                axis_idx,
                step=self._state.window_shift_samples,
            )
            offset_view = sliding_win_fun(buffer_tvec, self._state.window_samples, 0)[
                :: self._state.window_shift_samples
            ]
            win_offset = offset_view[0, 0]

            # Drop expired beginning of buffer and update shift_deficit
            multi_shift = self._state.window_shift_samples * out_dat.shape[axis_idx]
            self._state.shift_deficit = max(
                0, multi_shift - self._state.buffer.shape[axis_idx]
            )
            self._state.buffer = slice_along_axis(
                self._state.buffer, slice(multi_shift, None), axis_idx
            )
        else:
            # Not enough data to make a new window. Return empty data.
            empty_data_shape = (
                message.data.shape[:axis_idx]
                + (0, self._state.window_samples)
                + message.data.shape[axis_idx + 1 :]
            )
            out_dat = xp.zeros(empty_data_shape, dtype=message.data.dtype)
            # out_newaxis will have first timestamp in input... but mostly meaningless because output is size-zero.
            win_offset = axis_info.offset

        if self.settings.anchor == Anchor.END:
            win_offset += self.settings.window_dur
        elif self.settings.anchor == Anchor.MIDDLE:
            win_offset += self.settings.window_dur / 2
        self._state.out_newaxis = replace(self._state.out_newaxis, offset=win_offset)

        msg_out = replace(
            message,
            data=out_dat,
            dims=self._state.out_dims,
            axes={**out_axes, _newaxis: self._state.out_newaxis},
        )
        return msg_out


class Window(
    BaseTransformerUnit[WindowSettings, AxisArray, AxisArray, WindowTransformer]
):
    SETTINGS = WindowSettings
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        """
        override superclass on_signal so we can opt to yield once or multiple times after dropping the win axis.
        """
        # TODO: The transfomer overwrites settings.newaxis from None to "win",
        #  then we no longer know if the user wants to trim out the newaxis from the unit.
        xp = get_namespace(message.data)
        try:
            ret = self.processor(message)
            if ret.data.size > 0:
                if (
                    self.SETTINGS.newaxis is not None
                    or self.SETTINGS.window_dur is None
                ):
                    # Multi-win mode or pass-through mode.
                    yield self.OUTPUT_SIGNAL, ret
                else:
                    # We need to split out_msg into multiple yields, dropping newaxis.
                    axis_idx = ret.get_axis_idx("win")
                    win_axis = ret.axes["win"]
                    offsets = win_axis.value(
                        xp.asarray(range(ret.data.shape[axis_idx]))
                    )
                    for msg_ix in range(ret.data.shape[axis_idx]):
                        # Need to drop 'win' and replace self.SETTINGS.axis from axes.
                        _out_axes = {
                            **{
                                k: v
                                for k, v in ret.axes.items()
                                if k not in ["win", self.SETTINGS.axis]
                            },
                            self.SETTINGS.axis: replace(
                                ret.axes[self.SETTINGS.axis], offset=offsets[msg_ix]
                            ),
                        }
                        _ret = replace(
                            ret,
                            data=slice_along_axis(ret.data, msg_ix, axis_idx),
                            dims=ret.dims[:axis_idx] + ret.dims[axis_idx + 1 :],
                            axes=_out_axes,
                        )
                        yield self.OUTPUT_SIGNAL, _ret

        except Exception:
            ez.logger.info(traceback.format_exc())


def windowing(
    axis: str | None = None,
    newaxis: str | None = None,
    window_dur: float | None = None,
    window_shift: float | None = None,
    zero_pad_until: str = "full",
    anchor: str | Anchor = Anchor.BEGINNING,
) -> WindowTransformer:
    return WindowTransformer(
        WindowSettings(
            axis=axis,
            newaxis=newaxis,
            window_dur=window_dur,
            window_shift=window_shift,
            zero_pad_until=zero_pad_until,
            anchor=anchor,
        )
    )
