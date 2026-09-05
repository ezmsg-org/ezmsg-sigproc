"""Sliding and tumbling window segmentation of streaming data."""

import enum
import traceback
import typing

import ezmsg.core as ez
import numpy.typing as npt
import sparse
from array_api_compat import get_namespace, is_pydata_sparse_namespace
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import (
    AxisArray,
    replace,
    slice_along_axis,
    sliding_win_oneaxis,
)

from .util.array import xp_empty
from .util.buffer import HybridBuffer, UpdateStrategy
from .util.deprecation import warn_axis_deprecated
from .util.message import is_empty_along, resolve_configured_chunk_dim
from .util.profile import profile_subpub
from .util.sparse import sliding_win_oneaxis as sparse_sliding_win_oneaxis


class Anchor(enum.Enum):
    BEGINNING = "beginning"
    END = "end"
    MIDDLE = "middle"


class WindowSettings(ez.Settings):
    axis: str | None = None
    """.. deprecated:: 3.8
        Scheduled for removal in 4.0. The dimension messages accumulate along
        now comes from :attr:`~ezmsg.util.messages.axisarray.AxisArray.chunk_dim`;
        see :mod:`ezmsg.sigproc.util.deprecation`."""

    def __post_init__(self) -> None:
        warn_axis_deprecated(self)

    newaxis: str | None = None
    """Name of the axis windows are delimited on, inserted before ``axis``.

    ``None`` (default) means the *published* messages carry no window axis: the
    :obj:`Window` unit yields one message per window, each exactly ``window_dur``
    long with its own absolute offset. The transformer still emits a ``win`` axis
    in that case, because a transformer is 1-in/1-out and several windows may
    complete at once; the unit is what unbundles them.

    Set :attr:`batch_windows` to trade that per-window guarantee for fewer,
    larger messages."""
    window_dur: float | None = None  # Sec. passthrough if None
    window_shift: float | None = None  # Sec. Use "1:1 mode" if None
    zero_pad_until: str = "full"  # "full", "shift", "input", "none"
    anchor: str | Anchor = Anchor.BEGINNING

    batch_windows: bool = False
    """Emit all complete windows as one contiguous message ("batcher mode").

    Only meaningful with ``newaxis=None`` and ``window_shift == window_dur``,
    where consecutive windows tile the target axis exactly; requiring both is
    validated at construction.

    Off by default, because a message of exactly ``window_dur`` is what asking
    for a ``window_dur`` most obviously means -- and consumers with a fixed input
    size (an FFT, a binned aggregation) depend on it. Turn it on to re-chunk a
    stream purely for throughput, accepting that an emitted message is then a
    whole-number *multiple* of ``window_dur``, since one oversized input can
    complete several windows at once."""

    buffer_update_strategy: UpdateStrategy = "immediate"
    """When the backlog copies incoming samples into its own memory.
    See :obj:`ezmsg.sigproc.util.buffer.UpdateStrategy`.

    ``"immediate"`` (default, matching :obj:`~ezmsg.sigproc.sampler.Sampler` and
    :obj:`~ezmsg.sigproc.resample.Resample`) copies on every write. That is what
    makes the buffer safe behind a cross-process link: ezmsg marshals with
    pickle protocol 5 out-of-band buffers, so ``message.data`` is a *view* into
    a shared-memory slot that the publisher recycles every ``num_buffers``
    messages. Holding the array keeps the Python object alive but not its
    contents.

    ``"on_demand"`` defers the copy, saving ~3 us per message at 256 channels.
    Only safe when nothing recycles the incoming buffer -- i.e. a graph you know
    is single-process, where messages are passed by reference (``put_local``)
    rather than serialized."""


@processor_state
class WindowState:
    buffer: HybridBuffer | None = None
    """Backlog of samples awaiting a complete window.

    A ``HybridBuffer`` writing into preallocated memory, rather than re-growing
    one array per message. With 30-sample chunks feeding a 600-sample window, 19
    of every 20 calls produce no output, and ``concatenate((buffer, new))`` made
    each of those copy the *whole* backlog -- ~25 us at 256 channels, 85% of the
    call. Copying the new chunk into a fixed allocation instead is ~2 us and
    independent of how much is already buffered.

    Whether writes copy immediately is
    :attr:`WindowSettings.buffer_update_strategy`; the default copies, because
    incoming message data may be a view into memory the publisher recycles.
    """

    concat_buffer: npt.NDArray | sparse.SparseArray | None = None
    """Fallback backlog for namespaces ``HybridBuffer`` can't back.

    pydata/sparse arrays have no item assignment, so they cannot be written into
    a preallocated buffer; those streams keep the original grow-by-concatenate
    behaviour. Sparse windowing is not a throughput path.
    """

    buffer_len: int = 0
    """Samples buffered, mirrored here so the hot path skips a method call."""

    window_samples: int | None = None

    window_shift_samples: int | None = None

    shift_deficit: int = 0
    """ Number of incoming samples to ignore. Only relevant when shift > window."""

    newaxis_warned: bool = False

    out_newaxis: AxisArray.LinearAxis | None = None

    out_axis: AxisArray.LinearAxis | None = None
    """Target axis re-anchored per ``anchor``; constant for the life of the state."""

    out_dims: list[str] | None = None
    out_chunk_dim: str | None = None

    empty_out: npt.NDArray | sparse.SparseArray | None = None
    """Cached zero-window output, returned unchanged whenever no window is due."""


class WindowTransformer(BaseStatefulTransformer[WindowSettings, AxisArray, AxisArray, WindowState]):
    """
    Apply a sliding window along the specified axis to input streaming data.
    The `windowing` method is perhaps the most useful and versatile method in ezmsg.sigproc, but its parameterization
    can be difficult. Please read the argument descriptions carefully.

    Several windows can complete on one input, so the transformer -- being
    1-in/1-out -- represents them along a ``win`` axis. What reaches subscribers
    depends on the settings:

    ===========================  =========================================
    settings                     published messages
    ===========================  =========================================
    ``newaxis="win"``            one message, ``win`` axis, N windows
    ``newaxis=None``             N messages, no ``win`` axis, each exactly
                                 ``window_dur``
    ``newaxis=None`` +           one message, no ``win`` axis,
    ``batch_windows=True``       ``N * window_dur`` contiguous samples
    ===========================  =========================================

    The last row is "batcher mode", available only when ``window_shift ==
    window_dur`` so that windows tile the target axis exactly. It is the only
    mode the transformer can produce without a ``win`` axis, because tiling is
    what makes concatenation lossless.
    """

    # `anchor` only affects offset math in `_process`; every other field
    # sizes the buffer or drives the output axes in `_reset_state`.
    NONRESET_SETTINGS_FIELDS = frozenset({"anchor"})

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
        if self.settings.window_shift is None and self.settings.zero_pad_until != "input":
            ez.logger.warning(
                "`zero_pad_until` must be 'input' if `window_shift` is None; "
                f"coercing from {self.settings.zero_pad_until!r}. Window settings: "
                f"axis={self.settings.axis!r}, newaxis={self.settings.newaxis!r}, "
                f"window_dur={self.settings.window_dur!r}."
            )
            object.__setattr__(self.settings, "zero_pad_until", "input")
        elif self.settings.window_shift is not None and self.settings.zero_pad_until == "input":
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
        if self.settings.batch_windows:
            if self.settings.newaxis is not None:
                raise ValueError(
                    f"batch_windows requires newaxis=None, got newaxis={self.settings.newaxis!r}. "
                    "A window axis already batches windows into one message."
                )
            if self.settings.window_shift != self.settings.window_dur:
                raise ValueError(
                    f"batch_windows requires window_shift == window_dur, got "
                    f"window_shift={self.settings.window_shift!r} and window_dur={self.settings.window_dur!r}. "
                    "Windows that overlap or leave gaps do not tile the target axis, so they cannot be "
                    "concatenated without losing or duplicating samples."
                )
        if self.is_batcher and self.settings.anchor != Anchor.BEGINNING:
            # `anchor` says which sample of a window maps to 0 on the target axis.
            # Batcher mode has no per-window axis to anchor -- the target axis
            # carries absolute time -- so honouring it is impossible and ignoring
            # it would quietly discard the caller's intent.
            raise ValueError(
                f"anchor={self.settings.anchor.value!r} has no meaning with batch_windows=True: "
                "batched output carries absolute offsets on the target axis, so there is no "
                "window-relative origin to anchor. Leave batch_windows off to get one anchored "
                "message per window, or leave anchor at its default."
            )

    @property
    def is_batcher(self) -> bool:
        """Whether complete windows are emitted contiguously, with no ``win`` axis."""
        return (
            self.settings.batch_windows
            and self.settings.newaxis is None
            and self.settings.window_shift is not None
            and self.settings.window_shift == self.settings.window_dur
        )

    def _reset_state(self, message: AxisArray) -> None:
        _newaxis = self.settings.newaxis or "win"
        if not self._state.newaxis_warned and _newaxis in message.dims:
            ez.logger.warning(f"newaxis {_newaxis} present in input dims. Using {_newaxis}_win instead")
            self._state.newaxis_warned = True
            self.settings.newaxis = f"{_newaxis}_win"
            # Re-read: out_dims below must use the renamed axis, or it would
            # disagree with the axes key _process writes.
            _newaxis = self.settings.newaxis

        axis = resolve_configured_chunk_dim(self, message, self.settings.axis)
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
        elif self.settings.zero_pad_until == "shift" and self.settings.window_shift is not None:
            req_samples = self._state.window_shift_samples
        else:  # i.e. zero_pad_until == "input"
            req_samples = message.data.shape[axis_idx]
        n_zero = max(0, self._state.window_samples - req_samples)
        # Capacity has to cover a full window plus whatever a single message can
        # add on top of it; "grow" absorbs anything larger (e.g. an offline chunk
        # spanning many windows).
        capacity = self._state.window_samples + max(message.data.shape[axis_idx], self._state.window_samples)
        zero_shape = message.data.shape[:axis_idx] + (n_zero,) + message.data.shape[axis_idx + 1 :]
        self._state.buffer = None
        self._state.concat_buffer = None
        if is_pydata_sparse_namespace(xp):
            self._state.concat_buffer = xp.zeros(zero_shape, dtype=message.data.dtype)
        else:
            self._state.buffer = HybridBuffer(
                xp,
                capacity=capacity,
                other_shape=message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :],
                dtype=message.data.dtype,
                sample_axis=axis_idx,
                update_strategy=self.settings.buffer_update_strategy,
            )
            if n_zero:
                self._state.buffer.write(xp.zeros(zero_shape, dtype=message.data.dtype))
        self._state.buffer_len = n_zero

        # Prepare reusable parts of output. Rebuilt unconditionally: a reset means
        # the sample shape, sample rate or key changed, and every cached value
        # below is derived from one of those.
        self._state.out_axis = None
        self._state.empty_out = None
        if self.is_batcher:
            # Windows tile the target axis, so they need no axis of their own,
            # and the stream still grows along whichever dim it did before.
            self._state.out_dims = list(message.dims)
            self._state.out_newaxis = None
            self._state.out_chunk_dim = message.chunk_dim
        else:
            self._state.out_dims = list(message.dims[:axis_idx]) + [_newaxis] + list(message.dims[axis_idx:])
            self._state.out_newaxis = replace(
                axis_info,
                gain=0.0 if self.settings.window_shift is None else axis_info.gain * self._state.window_shift_samples,
                offset=0.0,  # offset modified per-msg below
            )
            # Successive messages now append along `newaxis`: its length is the
            # number of windows this message happened to yield, while the target
            # axis has become a fixed-length within-window axis. Declaring it
            # spares every downstream consumer from having to guess, and gets
            # the guess right where a "time" convention would not.
            self._state.out_chunk_dim = _newaxis

    def __call__(self, message: AxisArray) -> AxisArray:
        if self.settings.window_dur is None:
            # Shortcut for no windowing
            return message
        return super().__call__(message)

    def _buffer_write(self, xp, data, axis_idx: int) -> None:
        """Add *data*'s samples to the backlog.

        Under the default ``buffer_update_strategy="immediate"`` this copies, so
        the backlog no longer refers to the caller's array -- see that setting for
        why retaining it is unsafe across a cross-process link.
        """
        if self._state.buffer is not None:
            self._state.buffer.write(data)
        else:
            self._state.concat_buffer = xp.concatenate((self._state.concat_buffer, data), axis=axis_idx)
        self._state.buffer_len += data.shape[axis_idx]

    def _buffer_peek(self, xp, n: int, axis_idx: int, sample_shape: tuple[int, ...], dtype):
        """The oldest *n* buffered samples, as an array we own outright.

        ``HybridBuffer.peek`` hands back a view into its circular buffer whenever
        the read doesn't wrap, and the emitted window outlives the next write, so
        the copy is mandatory rather than defensive.
        """
        if self._state.buffer is None:
            return slice_along_axis(self._state.concat_buffer, slice(None, n), axis_idx)
        out_shape = sample_shape[:axis_idx] + (n,) + sample_shape[axis_idx:]
        return self._state.buffer.peek(n, out=xp_empty(xp, out_shape, dtype=dtype))

    def _drop_front(self, n: int, axis_idx: int) -> None:
        """Discard the oldest *n* buffered samples."""
        if self._state.buffer is not None:
            self._state.buffer.seek(n)
        else:
            self._state.concat_buffer = slice_along_axis(self._state.concat_buffer, slice(n, None), axis_idx)
        self._state.buffer_len -= min(n, self._state.buffer_len)

    def _process(self, message: AxisArray) -> AxisArray:
        axis = resolve_configured_chunk_dim(self, message, self.settings.axis)
        axis_idx = message.get_axis_idx(axis)
        axis_info = message.get_axis(axis)

        # Timestamp of the oldest buffered sample, computed before the new data
        # lands so `buffer_len` still describes the backlog it follows.
        buffer_t0 = axis_info.offset - self._state.buffer_len * axis_info.gain
        buffer_tlen = self._state.buffer_len + message.data.shape[axis_idx]

        xp = get_namespace(message.data)
        sample_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        if message.data.shape[axis_idx]:
            self._buffer_write(xp, message.data, axis_idx)

        if self.settings.window_shift is not None and self._state.shift_deficit > 0:
            n_skip = min(self._state.buffer_len, self._state.shift_deficit)
            if n_skip > 0:
                self._drop_front(n_skip, axis_idx)
                buffer_t0 += n_skip * axis_info.gain
                buffer_tlen -= n_skip
                self._state.shift_deficit -= n_skip

        # Generate outputs.
        # Preliminary copy of axes without the axes that we are modifying.
        _newaxis = self.settings.newaxis or "win"
        out_axes = {k: v for k, v in message.axes.items() if k not in [_newaxis, axis]}

        if self.is_batcher:
            # Windows tile the target axis exactly, so emit every complete one as a
            # single contiguous run and let the target axis keep absolute time.
            n_emit = (self._state.buffer_len // self._state.window_samples) * self._state.window_samples
            if n_emit:
                out_dat = self._buffer_peek(xp, n_emit, axis_idx, sample_shape, message.data.dtype)
                self._drop_front(n_emit, axis_idx)
                out_offset = buffer_t0
            else:
                if self._state.empty_out is None:
                    empty_shape = sample_shape[:axis_idx] + (0,) + sample_shape[axis_idx:]
                    self._state.empty_out = xp.zeros(empty_shape, dtype=message.data.dtype)
                out_dat = self._state.empty_out
                out_offset = axis_info.offset
            out_axes[axis] = replace(axis_info, offset=out_offset)
            return replace(
                message,
                data=out_dat,
                dims=self._state.out_dims,
                axes=out_axes,
                chunk_dim=self._state.out_chunk_dim,
            )

        # Update targeted (windowed) axis so that its offset is relative to the new axis.
        # The result depends only on the axis gain and the settings, both fixed for
        # the life of the state, so build it once rather than on every message.
        if self._state.out_axis is None:
            if self.settings.anchor == Anchor.BEGINNING:
                anchored_offset = 0.0
            elif self.settings.anchor == Anchor.END:
                anchored_offset = -self.settings.window_dur
            else:  # Anchor.MIDDLE
                anchored_offset = -self.settings.window_dur / 2
            self._state.out_axis = replace(axis_info, offset=anchored_offset)
        out_axes[axis] = self._state.out_axis

        # How we update .data and .axes[newaxis] depends on the windowing mode.
        if self.settings.window_shift is None:
            # one-to-one mode -- Each send yields exactly one window containing only the most recent samples.
            if self._state.buffer_len > self._state.window_samples:
                self._drop_front(self._state.buffer_len - self._state.window_samples, axis_idx)
            buffer = self._buffer_peek(xp, self._state.buffer_len, axis_idx, sample_shape, message.data.dtype)
            out_dat = buffer.reshape(buffer.shape[:axis_idx] + (1,) + buffer.shape[axis_idx:])
            win_offset = buffer_t0 + axis_info.gain * (buffer_tlen - self._state.window_samples)
        elif self._state.buffer_len >= self._state.window_samples:
            # Deterministic window shifts.
            buffer = self._buffer_peek(xp, self._state.buffer_len, axis_idx, sample_shape, message.data.dtype)
            sliding_win_fun = sparse_sliding_win_oneaxis if is_pydata_sparse_namespace(xp) else sliding_win_oneaxis
            out_dat = sliding_win_fun(
                buffer,
                self._state.window_samples,
                axis_idx,
                step=self._state.window_shift_samples,
            )
            win_offset = buffer_t0

            # Drop expired beginning of buffer and update shift_deficit
            multi_shift = self._state.window_shift_samples * out_dat.shape[axis_idx]
            self._state.shift_deficit = max(0, multi_shift - self._state.buffer_len)
            self._drop_front(multi_shift, axis_idx)
        else:
            # Not enough data to make a new window. Return empty data.
            # This is the common case when batching small chunks into large windows
            # (19 of every 20 calls at 30-sample input, 600-sample window), and the
            # allocation is identical every time: its shape comes from the sample
            # shape, which `_hash_message` already keys the state on.
            if self._state.empty_out is None:
                empty_data_shape = (
                    message.data.shape[:axis_idx] + (0, self._state.window_samples) + message.data.shape[axis_idx + 1 :]
                )
                self._state.empty_out = xp.zeros(empty_data_shape, dtype=message.data.dtype)
            out_dat = self._state.empty_out
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
            chunk_dim=self._state.out_chunk_dim,
        )
        return msg_out


class Window(BaseTransformerUnit[WindowSettings, AxisArray, AxisArray, WindowTransformer]):
    SETTINGS = WindowSettings
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        """
        override superclass on_signal so we can opt to yield once or multiple times after dropping the win axis.
        """
        # TODO: The transfomer overwrites settings.newaxis from None to "win",
        #  then we no longer know if the user wants to trim out the newaxis from the unit.
        xp = get_namespace(message.data)
        # Must resolve exactly as WindowTransformer does, or the emptiness gate
        # below checks a different dim than the one that was windowed. Resolved
        # from the *input*, since the output has `win` prepended.
        axis = resolve_configured_chunk_dim(self.processor, message, self.SETTINGS.axis)
        try:
            ret = self.processor(message)
            # Swallow only when no complete windows (or, in pass-through mode, no
            # samples) came out; emptiness along other axes (e.g. all channels
            # sliced away upstream) still flows to preserve stream cadence.
            if not is_empty_along(ret, (self.SETTINGS.newaxis or "win", axis)):
                if self.SETTINGS.newaxis is not None or self.SETTINGS.window_dur is None or "win" not in ret.dims:
                    # Multi-win mode, pass-through mode, or batcher mode -- the
                    # last of which already tiled its windows onto the target
                    # axis, so there is nothing to split.
                    yield self.OUTPUT_SIGNAL, ret
                else:
                    # We need to split out_msg into multiple yields, dropping newaxis.
                    axis_idx = ret.get_axis_idx("win")
                    win_axis = ret.axes["win"]
                    offsets = win_axis.value(xp.asarray(range(ret.data.shape[axis_idx])))
                    for msg_ix in range(ret.data.shape[axis_idx]):
                        # Need to drop 'win' and restore the target axis's absolute offset.
                        _out_axes = {
                            **{k: v for k, v in ret.axes.items() if k not in ["win", axis]},
                            axis: replace(ret.axes[axis], offset=offsets[msg_ix]),
                        }
                        _ret = replace(
                            ret,
                            data=slice_along_axis(ret.data, msg_ix, axis_idx),
                            dims=ret.dims[:axis_idx] + ret.dims[axis_idx + 1 :],
                            axes=_out_axes,
                            # Unbundling drops `win`, so the published stream is
                            # back to appending along the target axis: one
                            # message per window, each carrying its own offset.
                            chunk_dim=axis,
                        )
                        yield self.OUTPUT_SIGNAL, _ret

        except Exception:
            # Log loudly: this path used to swallow a KeyError on every message,
            # silently producing an empty stream for the whole run.
            ez.logger.error(traceback.format_exc())


def windowing(
    axis: str | None = None,
    newaxis: str | None = None,
    window_dur: float | None = None,
    window_shift: float | None = None,
    zero_pad_until: str = "full",
    anchor: str | Anchor = Anchor.BEGINNING,
    batch_windows: bool = False,
    buffer_update_strategy: UpdateStrategy = "immediate",
) -> WindowTransformer:
    return WindowTransformer(
        WindowSettings(
            axis=axis,
            newaxis=newaxis,
            window_dur=window_dur,
            window_shift=window_shift,
            zero_pad_until=zero_pad_until,
            anchor=anchor,
            batch_windows=batch_windows,
            buffer_update_strategy=buffer_update_strategy,
        )
    )
