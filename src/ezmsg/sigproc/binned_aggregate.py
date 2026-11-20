"""Resample a signal axis to a lower rate by aggregating fixed-duration bins.

This is the dense-signal counterpart to the binning used by
``ezmsg.event.rate.EventRate``. Both reduce a high-rate axis to a lower-rate
"bin" axis, but historically they disagreed when ``bin_duration * fs`` is not an
integer:

* ``EventRate`` bins by a *fractional* ``samples_per_bin = bin_duration * fs``
  with a carry accumulator, so each bin spans exactly ``bin_duration`` in the
  input's time base and it labels the output gain as the nominal ``bin_duration``.
* ``Window`` (``Pow -> Window -> Aggregate``) bins by a *fixed*
  ``int(bin_duration * fs)`` sample count, so its gain is
  ``int(bin_duration * fs) / fs``.

At a clean rate (e.g. 30000 Hz) those coincide, but at an off-nominal rate
(e.g. 30012 Hz) they diverge in both gain and bin count, so two such streams
never share a grid.

:obj:`BinnedAggregateTransformer` applies one or more arbitrary
:obj:`AggregationFunction`\\ s per bin instead of only counting -- a tuple of
functions stacks its results on a trailing axis, e.g. ``(MIN, MAX)`` for a
display envelope -- but it does *not* define its own bin boundaries: it drives
them through the shared :obj:`ezmsg.sigproc.util.binning.BinSchedule`,
the single source of truth for the grid. Any consumer that goes through the same
schedule at the same ``bin_duration`` lands on the same grid by construction, so
two such streams align downstream (e.g. with :obj:`ezmsg.sigproc.merge.Merge`).
Set ``fractional=False`` to instead bin by a fixed ``int(bin_duration * fs)``
sample count (sample-locked grid, gain ``int(bin_duration * fs) / fs``), matching
:obj:`Window`.

.. note::
    The schedule reproduces ``EventRate``'s grid by *shared code*, not by a
    copied formula: ``BinSchedule`` is the boundary primitive both consume.
    ``test_bin_schedule.py`` pins ``fractional=True`` against a faithful port of
    ``EventRate``'s algorithm and ``fractional=False`` against ``Window``'s; the
    cross-package test against the real ``EventRate`` lives in ezmsg-event.
"""

import typing

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from array_api_compat import get_namespace
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
    resolve_configured_stream_dim,
)
from ezmsg.util.messages.axisarray import (
    AxisArray,
    replace,
    slice_along_axis,
)

from .aggregate import AggregationFunction, aggregate_slices, needs_coordinates
from .util.array import xp_copy
from .util.binning import BinSchedule, BinStep
from .util.deprecation import warn_axis_deprecated
from .util.message import is_empty_along, with_fingerprint
from .util.segment import ConfigChain

# Whether each operation's value is independent of how many samples the bin holds,
# which decides if `flush` may emit a short final bin next to full ones. Every
# AggregationFunction must appear here (a test enforces it).
_MEAN_LIKE: dict[AggregationFunction, bool] = {
    AggregationFunction.NONE: True,
    AggregationFunction.MAX: True,
    AggregationFunction.MIN: True,
    AggregationFunction.MEAN: True,
    AggregationFunction.MEDIAN: True,
    AggregationFunction.STD: True,
    AggregationFunction.SUM: False,
    AggregationFunction.NANMAX: True,
    AggregationFunction.NANMIN: True,
    AggregationFunction.NANMEAN: True,
    AggregationFunction.NANMEDIAN: True,
    AggregationFunction.NANSTD: True,
    AggregationFunction.NANSUM: False,
    AggregationFunction.ARGMIN: True,  # a time within the bin
    AggregationFunction.ARGMAX: True,
    AggregationFunction.TRAPEZOID: False,  # an integral grows with the bin's duration
}


class BinnedAggregateSettings(ez.Settings):
    """Settings for :obj:`BinnedAggregate`."""

    axis: str | None = None
    """.. deprecated:: 3.8
        Scheduled for removal in 4.0. The dimension messages accumulate along
        now comes from :attr:`~ezmsg.util.messages.axisarray.AxisArray.stream_dim`;
        see :mod:`ezmsg.sigproc.util.deprecation`."""

    def __post_init__(self) -> None:
        warn_axis_deprecated(self)

    bin_duration: float = 0.02
    """Output bin duration in seconds."""

    operation: AggregationFunction | tuple[AggregationFunction, ...] = AggregationFunction.MEAN
    """:obj:`AggregationFunction` applied within each bin.

    A tuple applies several aggregations to the same bins and stacks the results
    on a new trailing axis named by ``newaxis``, coordinate-labelled with each
    function's value (``"min"``, ``"max"``, ...). ``(MIN, MAX)`` is the envelope
    used to put a high-rate signal on a screen without stride-decimation
    clipping the peaks: it keeps each bin's extremes exactly, so a spike shows
    at full amplitude no matter where in the bin it fell.

    A one-element tuple still produces the trailing axis -- the output shape
    follows the *type* of this field, not the number of functions, so a caller
    that builds its tuple programmatically gets a stable shape.
    """

    newaxis: str = "metric"
    """Name of the trailing axis added when ``operation`` is a tuple."""

    passthrough: bool = False
    """Forward messages untouched, as if this node were not in the graph.

    A separate flag rather than a sentinel ``bin_duration`` so the bin rate
    survives being switched off and on -- a consumer toggling this at runtime
    (say, because the view zoomed to a range where binning would cost detail)
    should not have to remember and restore the rate itself.

    Switching back off starts a fresh schedule and an empty carry: nothing from
    before the gap is spliced onto the first bin after it.

    Note that toggling changes the *shape* of the output: a tuple ``operation``
    adds a trailing axis, and turning it off takes that axis away again.
    Everything downstream has to be able to absorb that -- a fixed-layout sink
    will have to reallocate, and a plot will have to rebuild."""

    fractional: bool = True
    """If True (default), bins span a *fractional* ``bin_duration * fs`` samples
    with a carry accumulator across chunks; each bin spans exactly ``bin_duration``
    in the input's time base and the output gain is the nominal ``bin_duration``.
    This matches ``ezmsg.event.rate.EventRate``. If False, bins span a *fixed*
    ``int(bin_duration * fs)`` samples and the output gain is
    ``int(bin_duration * fs) / fs`` (sample-locked, matching :obj:`Window`)."""

    phase: float | typing.Callable[[float], float] = 0.0
    """Offset of the bin grid from the stream start, in seconds: boundaries fall at
    ``stream_start + phase + k * bin_duration``, rounded to whole input samples, and
    the first bin is shortened (see :obj:`~ezmsg.sigproc.util.binning.BinSchedule`).

    A callable is called with the input sample rate at reset, for a phase that
    depends on it. For example, ``-compute_pad_length(bp_settings, fs) / fs`` puts
    each bin's end where a :obj:`~ezmsg.sigproc.butterworthzerophase.ButterworthZeroPhase`
    upstream ends its output messages, so a bin closes with the message that
    completes it rather than with the next one. Use a module-level function (or
    a :func:`functools.partial` of one) so the settings stay picklable."""


@processor_state
class BinnedAggregateState:
    axis: str = ""
    """The resolved stream dimension, fixed at reset so every later use agrees."""

    schedule: BinSchedule | None = None
    """Shared bin-boundary schedule (see :obj:`ezmsg.sigproc.util.binning`). Owns
    the sample rate, samples-per-bin, output gain, global bin index, and carried
    sample *count* -- the boundary arithmetic this transformer shares with
    ``EventRate``. This transformer adds only the *data* carry below."""

    carry: typing.Any = None
    """Raw leftover samples of the open partial bin, carried across chunks so
    aggregation works for any operation (not just sums). Its length is kept in
    sync with ``schedule.carry_count``."""

    template: AxisArray | None = None
    """The input's layout with no data, taken at reset (the layout cannot change
    without one). `flush` labels the open bin from it."""

    next_offset: float = 0.0
    """Stream-axis offset just past the last input sample, where the next chunk would start."""

    last_attrs: dict | None = None
    last_config_id: int = 0
    """The last input's attrs and config_id, which can change without a reset; `flush` gives them to the open bin."""

    metric_axis: AxisArray.CoordinateAxis | None = None
    """The trailing axis attached to every multi-operation output.

    It depends only on settings, so it is built once here rather than rebuilt
    per message -- and since it is the same object every time, downstream
    identity checks on the axis stay cheap. ``None`` for a scalar operation."""

    configs: ConfigChain | None = None
    """Derives the output's config_id from the input's (see :class:`~ezmsg.sigproc.util.segment.ConfigChain`)."""


class BinnedAggregateTransformer(
    BaseStatefulTransformer[BinnedAggregateSettings, AxisArray, AxisArray, BinnedAggregateState]
):
    """Bin a signal axis at a fixed bin rate and aggregate within each bin.

    Unlike :obj:`AggregateTransformer` (which collapses a whole axis) or
    :obj:`RangedAggregateTransformer` (which aggregates static coordinate
    bands), this reduces a high-rate axis to a regularly-binned lower-rate
    axis, carrying the open partial bin across message boundaries.

    ``settings.operation`` may be a tuple, in which case every function is
    applied to the same bins and the results stack on a trailing ``newaxis``.
    The bins are computed once and sliced once, so N aggregations cost far less
    than N copies of this transformer, and -- more importantly -- they are
    guaranteed to describe the same bins.
    """

    # `passthrough` is read live in `__call__`/`__acall__`, which short-circuit
    # before the state is ever consulted; everything else is baked into the
    # schedule and the metric axis during `_reset_state`.
    NONRESET_SETTINGS_FIELDS = frozenset({"passthrough"})

    def __call__(self, message: AxisArray) -> AxisArray:
        if self.settings.passthrough:
            self._request_reset()
            return message
        return super().__call__(message)

    async def __acall__(self, message: AxisArray) -> AxisArray:
        if self.settings.passthrough:
            self._request_reset()
            return message
        return await super().__acall__(message)

    def _reset_state(self, message: AxisArray) -> None:
        self._state.axis = resolve_configured_stream_dim(self, message, self.settings.axis, legacy_default="time")
        axis_info = message.get_axis(self._state.axis)
        fs = 1.0 / axis_info.gain
        phase = self.settings.phase
        schedule = BinSchedule(
            bin_duration=self.settings.bin_duration,
            fractional=self.settings.fractional,
            phase=phase(fs) if callable(phase) else phase,
        )
        schedule.reset(fs)
        self._state.schedule = schedule
        self._state.metric_axis = (
            with_fingerprint(
                AxisArray.CoordinateAxis(
                    data=np.array([op.value for op in self._operations]),
                    dims=[self.settings.newaxis],
                )
            )
            if self._multi
            else None
        )
        self._state.carry = None
        metric = self._state.metric_axis
        self._state.configs = ConfigChain(
            self._state.axis, schedule.output_gain, None if metric is None else metric.fingerprint
        )
        # Copied empty, not viewed, so the input's buffer is not kept alive.
        axis_idx = message.get_axis_idx(self._state.axis)
        self._state.template = replace(
            message, data=xp_copy(slice_along_axis(message.data, slice(0, 0), axis=axis_idx))
        )

    @property
    def _operations(self) -> tuple[AggregationFunction, ...]:
        """``operation`` as a tuple, regardless of how it was given."""
        op = self.settings.operation
        return op if isinstance(op, tuple) else (op,)

    @property
    def _multi(self) -> bool:
        """Whether to emit a trailing metric axis."""
        return isinstance(self.settings.operation, tuple)

    def _work_coordinates(self, work, axis_idx: int, axis_info, carry_len: int) -> npt.NDArray:
        """Coordinate value of every sample in the ``[carry ++ new]`` work array.

        The carried samples precede this message, so the vector starts
        ``carry_len`` samples *before* the message's own offset. This is why the
        work array cannot use :func:`aggregate_slices`' message-based helper: it
        spans more than the message does.
        """
        n = work.shape[axis_idx]
        return axis_info.value(np.arange(n) - carry_len)

    def _aggregate(self, work, slices, axis_idx: int, coordinates):
        """Reduce every bin. Multi-op results stack on a new *trailing* axis.

        Trailing rather than in place, so the binned axis keeps its position and
        every existing consumer of a single-op stream sees an unchanged shape.
        """
        xp = get_namespace(work)
        if not self._multi:
            return aggregate_slices(work, slices, axis_idx, self.settings.operation, coordinates=coordinates)
        return xp.stack(
            [aggregate_slices(work, slices, axis_idx, op, coordinates=coordinates) for op in self._operations],
            axis=-1,
        )

    def _out_dims(self, message: AxisArray) -> list[str]:
        dims = list(message.dims)
        return dims + [self.settings.newaxis] if self._multi else dims

    def _out_axes(self, message: AxisArray, step: BinStep) -> dict:
        axis_info = message.get_axis(self._state.axis)
        axes = {
            **message.axes,
            self._state.axis: replace(axis_info, gain=step.output_gain, offset=step.output_offset),
        }
        if self._multi:
            axes[self.settings.newaxis] = self._state.metric_axis
        return axes

    def _empty_like(self, message: AxisArray, axis_idx: int, step: BinStep) -> AxisArray:
        xp = get_namespace(message.data)
        data = slice_along_axis(message.data, slice(0, 0), axis=axis_idx)
        if self._multi:
            # Zero-length along the binned axis, but the metric axis must still
            # be its full width or the empty message would not match the shape
            # of the ones around it.
            data = xp.stack([data] * len(self._operations), axis=-1)
        return replace(
            message,
            data=data,
            dims=self._out_dims(message),
            axes=self._out_axes(message, step),
            config_id=self._state.configs.update(message.config_id),
        )

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self._state.axis
        axis_info = message.get_axis(axis)
        axis_idx = message.get_axis_idx(axis)
        xp = get_namespace(message.data)

        carry = self._state.carry
        n_new = message.data.shape[axis_idx]
        self._state.next_offset = axis_info.value(n_new)
        self._state.last_attrs = message.attrs
        self._state.last_config_id = message.config_id

        # The schedule owns all boundary/gain/offset arithmetic; this transformer
        # only slices and aggregates the data at the cut points it returns.
        step = self._state.schedule.advance(
            n_new=n_new,
            in_offset=axis_info.offset,
            gain_in=axis_info.gain,
        )

        if step.n_bins == 0:
            # No bin completes in this chunk; grow the carry and emit nothing.
            # `xp_copy` is intentional. A chunk with no samples leaves the carry as it was: a
            # zero-length carry would be an open bin with nothing in it, which `flush` would emit.
            if n_new:
                self._state.carry = (
                    xp_copy(message.data) if carry is None else xp.concat((carry, message.data), axis=axis_idx)
                )
            return self._empty_like(message, axis_idx, step)

        # Prepend the carried partial-bin samples so bin 0 spans carry + current.
        carry_len = 0 if carry is None else carry.shape[axis_idx]
        work = message.data if carry is None else xp.concat((carry, message.data), axis=axis_idx)
        ends_work = step.cut_points
        starts_work = [0] + ends_work[:-1]
        slices = [slice(s, e) for s, e in zip(starts_work, ends_work)]

        coordinates = (
            self._work_coordinates(work, axis_idx, axis_info, carry_len)
            if needs_coordinates(self._operations)
            else None
        )
        stacked = self._aggregate(work, slices, axis_idx, coordinates)

        # Leftover after the last completed bin becomes the next chunk's carry
        # (its length is step.carry_count, tracked by the schedule).
        # Copied, not viewed, intentionally.
        last_work = ends_work[-1]
        self._state.carry = (
            xp_copy(slice_along_axis(work, slice(last_work, None), axis=axis_idx)) if step.carry_count > 0 else None
        )

        return replace(
            message,
            data=stacked,
            dims=self._out_dims(message),
            axes=self._out_axes(message, step),
            config_id=self._state.configs.update(message.config_id),
        )

    def flush(self, message: AxisArray | None = None) -> AxisArray | None:
        """Process ``message`` if given, then emit the open partial bin.

        The partial bin is aggregated over the samples it has, which are the first
        samples the full bin would have had, and is labelled with that bin's place
        on the grid. It is emitted only when every operation is mean-like (see
        ``_MEAN_LIKE``): a sum over a short bin is not comparable with the full ones,
        so then nothing is emitted for it and the open bin is kept, completing as
        usual if the stream continues after all.

        Once the partial bin is emitted it is gone, and the next message starts a
        fresh grid at its own offset, as after a gap.
        """
        out = None if message is None else self(message)
        carry = self._state.carry
        if self.settings.passthrough or carry is None or not all(_MEAN_LIKE[op] for op in self._operations):
            return out
        st = self._state
        template = replace(st.template, attrs=st.last_attrs, config_id=st.last_config_id)
        axis_info = replace(template.get_axis(st.axis), offset=st.next_offset)
        axis_idx = template.get_axis_idx(self._state.axis)
        # No new samples: closes no bin, only reports where the open one sits.
        step = self._state.schedule.advance(n_new=0, in_offset=axis_info.offset, gain_in=axis_info.gain)
        n = carry.shape[axis_idx]
        coordinates = (
            self._work_coordinates(carry, axis_idx, axis_info, n) if needs_coordinates(self._operations) else None
        )
        data = self._aggregate(carry, [slice(0, n)], axis_idx, coordinates)
        self._state.carry = None
        self._request_reset()
        if out is not None and out.data.shape[axis_idx] > 0:
            return replace(out, data=get_namespace(data).concat((out.data, data), axis=axis_idx))
        tail = replace(
            template,
            data=data,
            dims=self._out_dims(template),
            axes=self._out_axes(template, step),
            config_id=self._state.configs.update(template.config_id),
        )
        return self._verify_output(tail) if self.VERIFY_OUTPUT_CONFIG else tail


class BinnedAggregate(BaseTransformerUnit[BinnedAggregateSettings, AxisArray, AxisArray, BinnedAggregateTransformer]):
    SETTINGS = BinnedAggregateSettings

    @ez.subscriber(BaseTransformerUnit.INPUT_SIGNAL)
    @ez.publisher(BaseTransformerUnit.OUTPUT_SIGNAL)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        """Suppress empty publishes when a chunk spans less than one bin.

        As with :obj:`Downsample`, most input chunks at a high input rate close
        no new bin, yielding a zero-length payload; broadcasting those wastes a
        round-trip across SHM/socket. Only emptiness along the binned axis is
        suppressed: a message that is empty along other axes (e.g. all channels
        sliced away upstream) still flows so downstream consumers keep its
        cadence. An empty message that starts a new segment also flows, since
        downstream units flush and reset on the segment change.
        """
        segment = getattr(message, "segment", None)
        new_segment = self._segment is not BaseTransformerUnit._segment and segment != self._segment
        if new_segment:
            # Release what the processor holds before the new segment resets it.
            tail = await self.processor.aflush()
            if tail is not None:
                yield self.OUTPUT_SIGNAL, tail
        self._segment = segment
        result = await self.processor.__acall__(message)
        if result is not None and (new_segment or not is_empty_along(result, (self.processor.state.axis,))):
            yield self.OUTPUT_SIGNAL, result
