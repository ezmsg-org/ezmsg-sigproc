"""Integer downsampling by selecting every Nth sample along an axis."""

import typing

import ezmsg.core as ez
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import (
    AxisArray,
    replace,
    slice_along_axis,
)

from .util.message import is_empty_along, resolve_chunk_dim


class DownsampleSettings(ez.Settings):
    """
    Settings for :obj:`Downsample` node.
    """

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

    axis: str = ""
    """The dimension being downsampled: the message's declared ``chunk_dim``."""


class DownsampleTransformer(BaseStatefulTransformer[DownsampleSettings, AxisArray, AxisArray, DownsampleState]):
    """
    Downsampled data simply comprise every `factor`th sample.
    This should only be used following appropriate lowpass filtering.
    If your pipeline does not already have lowpass filtering then consider
    using the :obj:`Decimate` collection instead.

    **The dimension is not configurable: it is always the one messages
    accumulate along.** The phase counter ``s_idx`` carries across messages so
    the kept samples form one arithmetic sequence over the whole stream rather
    than restarting per chunk. That is the entire point along the accumulating
    dimension, and it is meaningless along any other: a static axis has the same
    length every message, so the carried phase makes the *selection itself*
    rotate. Downsampling ``(time, freq)`` along ``freq`` by 2 alternates between
    bins ``[0, 2, 4]`` and ``[1, 3]`` -- different frequencies, and a different
    output length, on alternating messages.

    For a static axis use :obj:`Slicer` with ``"::2"``, which selects the same
    elements every time and holds no state to do it.

    The dimension comes from the message's
    :attr:`~ezmsg.util.messages.axisarray.AxisArray.chunk_dim`, so a
    ``Downsample`` placed after a windowing stage decimates *windows* without
    reconfiguration. When a producer does not declare one, :attr:`STREAMING_DIMS`
    supplies the fallback.
    """

    def _resolve_axis(self, message: AxisArray) -> str:
        """The dimension messages accumulate along, which is the only one to
        downsample. Falls back to :attr:`STREAMING_DIMS` when undeclared."""
        return resolve_chunk_dim(message, self.STREAMING_DIMS)

    def _hash_message(self, message: AxisArray) -> int:
        # The whole state is a decimation factor and the phase counter that walks
        # it -- both derived from the target axis' gain, neither from the other
        # dimensions. The base-class default would fold in the channel
        # fingerprint and reset the phase whenever the channels were relabelled,
        # which costs a hash it does not need and puts a sample-alignment step in
        # the output for a change that cannot affect which samples are kept.
        axis = self._resolve_axis(message)
        return hash((message.key, axis, getattr(message.axes.get(axis), "gain", None)))

    def _reset_state(self, message: AxisArray) -> None:
        self._state.axis = self._resolve_axis(message)
        axis_info = message.get_axis(self._state.axis)

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
        axis = self._state.axis
        axis_info = message.get_axis(axis)
        axis_idx = message.get_axis_idx(axis)

        n_samples = message.data.shape[axis_idx]
        q = self._state.q
        s_idx = self._state.s_idx
        if n_samples > 0:
            # Update state for next iteration. Equivalent to the old
            # ``(arange(s_idx, s_idx + n) % q)[-1] + 1``.
            self._state.s_idx = (s_idx + n_samples - 1) % q + 1

        # The kept samples are exactly those whose rotating counter is 0, which
        # is an arithmetic sequence: first at ``(-s_idx) % q``, then every q.
        # Expressing it as a strided slice instead of a gather is not just
        # faster (1.9-3.5x on MLX, measured M4 Pro, 30x256 through 512x1024, and
        # a view rather than a copy on NumPy) -- MLX rejects indexing by a NumPy
        # integer array outright, so the gather form did not work there at all.
        n_step = -s_idx % q
        if n_step < n_samples:
            data_slice = slice(n_step, None, q)
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


class Downsample(BaseTransformerUnit[DownsampleSettings, AxisArray, AxisArray, DownsampleTransformer]):
    SETTINGS = DownsampleSettings

    @ez.subscriber(BaseTransformerUnit.INPUT_SIGNAL)
    @ez.publisher(BaseTransformerUnit.OUTPUT_SIGNAL)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        """Skip the publish when no samples accumulated.

        At ``factor > 1`` most input chunks span less than one downsample
        period, so ``DownsampleTransformer._process`` returns a payload with
        a zero-length axis. Suppressing the broadcast in that case avoids
        shipping an empty AxisArray across SHM/socket every input chunk.
        Only emptiness along the downsampled axis is suppressed: a message
        that is empty along other axes (e.g. all channels sliced away
        upstream) still flows so downstream consumers keep its cadence.
        """
        result = await self.processor.__acall__(message)
        # The processor is the one that worked out which dimension that is.
        if result is not None and not is_empty_along(result, (self.processor.state.axis,)):
            yield self.OUTPUT_SIGNAL, result


def downsample(
    target_rate: float | None = None,
    factor: int | None = None,
) -> DownsampleTransformer:
    return DownsampleTransformer(DownsampleSettings(target_rate=target_rate, factor=factor))
