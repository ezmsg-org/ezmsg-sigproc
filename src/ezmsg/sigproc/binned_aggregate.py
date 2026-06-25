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

:obj:`BinnedAggregateTransformer` follows ``EventRate``'s fractional+carry
boundary definition (including the sub-sample output offset) but applies an
arbitrary :obj:`AggregationFunction` per bin instead of only counting. Driving
the SBP branch through this with ``operation=MEAN`` puts it on the same grid as
the spike-rate branch by construction, so the two merge trivially -- no
post-hoc reconciler/aligner required. Set ``fractional=False`` to instead bin by
a fixed ``round(bin_duration * fs)`` sample count (sample-locked grid, gain
``round(bin_duration * fs) / fs``), matching :obj:`Window`'s legacy behaviour.

.. note::
    The grid is reproduced from ``EventRate``'s boundary *formula*
    (``B(m) = int(m * bin_duration * fs)``), not by sharing code with it.
    ``EventRate`` lives in a separate package, so the agreement is by
    construction and is not enforced by a cross-package regression test here.
    The closed-form ``int(m * spb)`` is in fact more numerically stable than an
    iterative carry accumulator, so for very long streams the two could differ
    by at most a sub-sample rounding at a single boundary.
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
    ``round(bin_duration * fs)`` samples and the output gain is
    ``round(bin_duration * fs) / fs`` (sample-locked, matching :obj:`Window`)."""


@processor_state
class BinnedAggregateState:
    fs: float | None = None
    """Input sample rate, cached from the first message's time-axis gain."""

    samples_per_bin: float = 0.0
    """Bin width in input samples (fractional when ``settings.fractional``)."""

    out_gain: float = 0.0
    """Gain (seconds per bin) of the output axis."""

    n_bins_done: int = 0
    """Count of bins already emitted across the stream. Bin boundaries are taken
    from this *global* index (``B(m) = int(m * samples_per_bin)``) rather than a
    per-chunk accumulator, so the output is identical regardless of how the
    input is chunked -- and reproduces ``EventRate``'s grid (gain, offset, bin
    count) by construction. This makes offline (large chunks) and live (small
    chunks) feature streams agree exactly."""

    carry: typing.Any = None
    """Raw leftover samples of the open partial bin, carried across chunks so
    aggregation works for any operation (not just sums)."""


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
        fs = 1.0 / axis_info.gain
        spb = self.settings.bin_duration * fs
        if not self.settings.fractional:
            spb = float(round(spb))
            if spb < 1.0:
                ez.logger.warning(
                    f"bin_duration {self.settings.bin_duration}s rounds to <1 sample at "
                    f"fs={fs}; clamping to 1 sample per bin."
                )
                spb = 1.0
        elif spb < 1.0:
            # Fewer than one input sample per bin means the output rate exceeds
            # the input rate. Bins would span zero samples (NaN aggregates), so
            # this is a misconfiguration rather than a supported mode.
            ez.logger.warning(
                f"bin_duration {self.settings.bin_duration}s is shorter than one sample "
                f"at fs={fs} (samples_per_bin={spb:.4g} < 1); output will contain empty "
                f"bins. Increase bin_duration or reduce the input rate."
            )
        self._state.fs = fs
        self._state.samples_per_bin = spb
        self._state.out_gain = self.settings.bin_duration if self.settings.fractional else spb / fs
        self._state.n_bins_done = 0
        self._state.carry = None

    def _aggregate(self, xp, segment, axis_idx: int):
        op = self.settings.operation
        func_name = op.value
        if hasattr(xp, func_name):
            return getattr(xp, func_name)(segment, axis=axis_idx)
        # nan-variants etc. are not in the Array API; fall back to numpy.
        result = AGGREGATORS[op](np.asarray(segment), axis=axis_idx)
        return xp.asarray(result) if xp is not np else result

    def _empty_like(self, message: AxisArray, axis_idx: int, offset: float) -> AxisArray:
        axis_info = message.get_axis(self.settings.axis)
        return replace(
            message,
            data=slice_along_axis(message.data, slice(0, 0), axis=axis_idx),
            axes={
                **message.axes,
                self.settings.axis: replace(axis_info, gain=self._state.out_gain, offset=offset),
            },
        )

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis
        axis_info = message.get_axis(axis)
        axis_idx = message.get_axis_idx(axis)
        xp = get_namespace(message.data)

        n = message.data.shape[axis_idx]
        spb = self._state.samples_per_bin
        gain_in = axis_info.gain

        m_done = self._state.n_bins_done
        carry = self._state.carry
        n_carry = 0 if carry is None else carry.shape[axis_idx]

        # Global sample index of carry[0] (== bin boundary B(m_done)); the
        # incoming chunk's first sample follows the carried samples.
        in_done = int(m_done * spb)
        data0_global = in_done + n_carry
        avail_end = data0_global + n

        # Nominal start time of the first (m_done-th) output bin. Derived from the
        # global bin index, so it is identical regardless of chunking and matches
        # EventRate's nominal offset (== stream_start + m_done * bin_duration).
        output_offset = axis_info.offset - data0_global * gain_in + m_done * self._state.out_gain

        # Global bin boundaries B(m) = int(m * spb) for the bins that complete
        # within the data available so far. Indexing by the *global* m (not a
        # per-chunk offset) is what makes the output chunk-invariant.
        b_global = np.empty(0, dtype=np.int64)
        if spb > 0 and avail_end >= int((m_done + 1) * spb):
            # A bin m completes once B(m) = int(m*spb) <= avail_end, i.e.
            # m < (avail_end + 1) / spb. int((avail_end + 1) / spb) is a correct
            # upper bound on the largest such m (even when spb < 1, where a
            # plain int(avail_end/spb) could undercount); +2 is slack, and the
            # `candidates <= avail_end` filter below trims any overshoot.
            k_est = max(int((avail_end + 1) / spb) - m_done, 0) + 2
            ms = m_done + 1 + np.arange(k_est)
            candidates = (ms * spb).astype(np.int64)
            b_global = candidates[candidates <= avail_end]

        if b_global.size == 0:
            # No bin completes in this chunk; grow the carry and emit nothing.
            self._state.carry = (
                message.data if carry is None else xp.concat((carry, message.data), axis=axis_idx)
            )
            return self._empty_like(message, axis_idx, output_offset)

        # Prepend the carried partial-bin samples so bin 0 spans carry + current.
        work = message.data if carry is None else xp.concat((carry, message.data), axis=axis_idx)
        ends_work = (b_global - in_done).tolist()
        starts_work = [0] + ends_work[:-1]

        bins = [
            self._aggregate(xp, slice_along_axis(work, slice(s, e), axis=axis_idx), axis_idx)
            for s, e in zip(starts_work, ends_work)
        ]
        stacked = xp.stack(bins, axis=axis_idx)

        # Leftover after the last completed bin becomes the next chunk's carry.
        last_work = ends_work[-1]
        self._state.carry = (
            slice_along_axis(work, slice(last_work, None), axis=axis_idx)
            if last_work < work.shape[axis_idx]
            else None
        )
        self._state.n_bins_done = m_done + len(ends_work)

        return replace(
            message,
            data=stacked,
            axes={
                **message.axes,
                axis: replace(axis_info, gain=self._state.out_gain, offset=output_offset),
            },
        )


class BinnedAggregate(
    BaseTransformerUnit[BinnedAggregateSettings, AxisArray, AxisArray, BinnedAggregateTransformer]
):
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
            ``round(bin_duration * fs)`` sample bins (sample-locked grid).

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
