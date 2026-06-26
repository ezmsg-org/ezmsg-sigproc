"""Resample a signal axis to a lower rate by aggregating fixed-duration bins.

This is the dense-signal counterpart to the binning used by
``ezmsg.event.rate.EventRate`` (spike rate from threshold-crossing events).
Both reduce a high-rate axis to a lower-rate "bin" axis, but historically they
disagreed at off-nominal sample rates:

* ``EventRate`` bins by a *fractional* ``samples_per_bin = bin_duration * fs``
  with a carry accumulator, so its bins track wall-clock time and it labels the
  output gain as the nominal ``bin_duration``.
* ``Window`` (the usual SBP path: ``Pow -> Window -> Aggregate``) bins by a
  *fixed* ``int(bin_duration * fs)`` sample count, so its gain is
  ``int(bin_duration * fs) / fs``.

At a clean rate (e.g. 30000 Hz) those coincide, but at a real recording rate
(e.g. ~30012 Hz) they diverge in both gain and bin count, so a downstream
``Merge``/``AlignAlongAxis`` of the two branches never aligns.

:obj:`BinnedAggregateTransformer` applies an arbitrary :obj:`AggregationFunction`
per bin instead of only counting, but it does *not* define its own bin
boundaries: it drives them through the shared :obj:`ezmsg.sigproc.util.binning.BinSchedule`,
the single source of truth for the grid. Driving the SBP branch through this with
``operation=MEAN`` puts it on the same grid as the spike-rate branch -- because
both go through the same schedule -- so the two merge trivially with no post-hoc
reconciler/aligner required. Set ``fractional=False`` to instead bin by a fixed
``int(bin_duration * fs)`` sample count (sample-locked grid, gain
``int(bin_duration * fs) / fs``), matching :obj:`Window`.

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

from .aggregate import AGGREGATORS, AggregationFunction
from .util.binning import BinSchedule, BinStep


class BinnedAggregateSettings(ez.Settings):
    """Settings for :obj:`BinnedAggregate`."""

    axis: str = "time"
    """The name of the axis to bin and aggregate along."""

    bin_duration: float = 0.02
    """Output bin duration in seconds."""

    operation: AggregationFunction = AggregationFunction.MEAN
    """:obj:`AggregationFunction` applied within each bin."""

    fractional: bool = True
    """If True (default), bins span a *fractional* ``bin_duration * fs`` samples
    with a carry accumulator across chunks; bins track wall-clock time and the
    output gain is the nominal ``bin_duration``. This matches
    ``ezmsg.event.rate.EventRate``. If False, bins span a *fixed*
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


class BinnedAggregateTransformer(
    BaseStatefulTransformer[BinnedAggregateSettings, AxisArray, AxisArray, BinnedAggregateState]
):
    """Bin a signal axis at a fixed bin rate and aggregate within each bin.

    Unlike :obj:`AggregateTransformer` (which collapses a whole axis) or
    :obj:`RangedAggregateTransformer` (which aggregates static coordinate
    bands), this reduces a high-rate axis to a regularly-binned lower-rate
    axis, carrying the open partial bin across message boundaries.
    """

    def _hash_message(self, message: AxisArray) -> int:
        return hash((message.axes[self.settings.axis].gain, message.key))

    def _reset_state(self, message: AxisArray) -> None:
        axis_info = message.get_axis(self.settings.axis)
        schedule = BinSchedule(
            bin_duration=self.settings.bin_duration,
            fractional=self.settings.fractional,
        )
        schedule.reset(1.0 / axis_info.gain)
        self._state.schedule = schedule
        self._state.carry = None

    def _aggregate(self, xp, segment, axis_idx: int):
        op = self.settings.operation
        func_name = op.value
        if hasattr(xp, func_name):
            return getattr(xp, func_name)(segment, axis=axis_idx)
        # nan-variants etc. are not in the Array API; fall back to numpy.
        result = AGGREGATORS[op](np.asarray(segment), axis=axis_idx)
        return xp.asarray(result) if xp is not np else result

    def _empty_like(self, message: AxisArray, axis_idx: int, step: BinStep) -> AxisArray:
        axis_info = message.get_axis(self.settings.axis)
        return replace(
            message,
            data=slice_along_axis(message.data, slice(0, 0), axis=axis_idx),
            axes={
                **message.axes,
                self.settings.axis: replace(axis_info, gain=step.output_gain, offset=step.output_offset),
            },
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
            self._state.carry = message.data if carry is None else xp.concat((carry, message.data), axis=axis_idx)
            return self._empty_like(message, axis_idx, step)

        # Prepend the carried partial-bin samples so bin 0 spans carry + current.
        work = message.data if carry is None else xp.concat((carry, message.data), axis=axis_idx)
        ends_work = step.cut_points
        starts_work = [0] + ends_work[:-1]

        bins = [
            self._aggregate(xp, slice_along_axis(work, slice(s, e), axis=axis_idx), axis_idx)
            for s, e in zip(starts_work, ends_work)
        ]
        stacked = xp.stack(bins, axis=axis_idx)

        # Leftover after the last completed bin becomes the next chunk's carry
        # (its length is step.carry_count, tracked by the schedule).
        last_work = ends_work[-1]
        self._state.carry = (
            slice_along_axis(work, slice(last_work, None), axis=axis_idx) if step.carry_count > 0 else None
        )

        return replace(
            message,
            data=stacked,
            axes={
                **message.axes,
                axis: replace(axis_info, gain=step.output_gain, offset=step.output_offset),
            },
        )


class BinnedAggregate(BaseTransformerUnit[BinnedAggregateSettings, AxisArray, AxisArray, BinnedAggregateTransformer]):
    SETTINGS = BinnedAggregateSettings

    @ez.subscriber(BaseTransformerUnit.INPUT_SIGNAL)
    @ez.publisher(BaseTransformerUnit.OUTPUT_SIGNAL)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        """Suppress empty publishes when a chunk spans less than one bin.

        As with :obj:`Downsample`, most input chunks at a high input rate close
        no new bin, yielding a zero-length payload; broadcasting those wastes a
        round-trip across SHM/socket.
        """
        result = await self.processor.__acall__(message)
        if result is not None and result.data.size > 0:
            yield self.OUTPUT_SIGNAL, result


def binned_aggregate(
    axis: str = "time",
    bin_duration: float = 0.02,
    operation: AggregationFunction = AggregationFunction.MEAN,
    fractional: bool = True,
) -> BinnedAggregateTransformer:
    """Bin a signal axis at a fixed bin rate and aggregate within each bin.

    Args:
        axis: The name of the axis to bin and aggregate along.
        bin_duration: Output bin duration in seconds.
        operation: :obj:`AggregationFunction` applied within each bin.
        fractional: If True, fractional ``bin_duration * fs`` bins with a carry
            accumulator (wall-clock grid, nominal gain), matching
            ``ezmsg.event.rate.EventRate``. If False, fixed
            ``int(bin_duration * fs)`` sample bins (sample-locked grid).

    Returns:
        :obj:`BinnedAggregateTransformer`
    """
    return BinnedAggregateTransformer(
        BinnedAggregateSettings(
            axis=axis,
            bin_duration=bin_duration,
            operation=operation,
            fractional=fractional,
        )
    )
