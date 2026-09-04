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
)
from ezmsg.util.messages.axisarray import (
    AxisArray,
    replace,
    slice_along_axis,
)

from .aggregate import AggregationFunction, aggregate_slices, needs_coordinates
from .util.array import xp_copy
from .util.binning import BinSchedule, BinStep
from .util.message import is_empty_along, with_fingerprint


class BinnedAggregateSettings(ez.Settings):
    """Settings for :obj:`BinnedAggregate`."""

    axis: str = "time"
    """The name of the axis to bin and aggregate along."""

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


@processor_state
class BinnedAggregateState:
    schedule: BinSchedule | None = None
    """Shared bin-boundary schedule (see :obj:`ezmsg.sigproc.util.binning`). Owns
    the sample rate, samples-per-bin, output gain, global bin index, and carried
    sample *count* -- the boundary arithmetic this transformer shares with
    ``EventRate``. This transformer adds only the *data* carry below."""

    carry: typing.Any = None
    """Raw leftover samples of the open partial bin, carried across chunks so
    aggregation works for any operation (not just sums). Its length is kept in
    sync with ``schedule.carry_count``."""

    metric_axis: AxisArray.CoordinateAxis | None = None
    """The trailing axis attached to every multi-operation output.

    It depends only on settings, so it is built once here rather than rebuilt
    per message -- and since it is the same object every time, downstream
    identity checks on the axis stay cheap. ``None`` for a scalar operation."""


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
        axis_info = message.get_axis(self.settings.axis)
        schedule = BinSchedule(
            bin_duration=self.settings.bin_duration,
            fractional=self.settings.fractional,
        )
        schedule.reset(1.0 / axis_info.gain)
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
        axis_info = message.get_axis(self.settings.axis)
        axes = {
            **message.axes,
            self.settings.axis: replace(axis_info, gain=step.output_gain, offset=step.output_offset),
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
        )

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis
        axis_info = message.get_axis(axis)
        axis_idx = message.get_axis_idx(axis)
        xp = get_namespace(message.data)

        carry = self._state.carry

        # The schedule owns all boundary/gain/offset arithmetic; this transformer
        # only slices and aggregates the data at the cut points it returns.
        step = self._state.schedule.advance(
            n_new=message.data.shape[axis_idx],
            in_offset=axis_info.offset,
            gain_in=axis_info.gain,
        )

        if step.n_bins == 0:
            # No bin completes in this chunk; grow the carry and emit nothing.
            # `xp_copy` is intentional.
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
        )


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
        cadence.
        """
        result = await self.processor.__acall__(message)
        if result is not None and not is_empty_along(result, (self.SETTINGS.axis,)):
            yield self.OUTPUT_SIGNAL, result
