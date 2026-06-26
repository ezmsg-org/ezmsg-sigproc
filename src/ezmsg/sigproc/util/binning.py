"""Shared bin-boundary scheduling for fixed-duration binning.

A *bin schedule* answers one question, identically for every consumer that
needs it: given an input sample rate, a target bin duration, and a stream of
chunks (each of some length, with a partial bin carried across boundaries),
*where do the bin boundaries fall* -- and what output ``gain``/``offset`` label
the resulting low-rate axis?

This is the single piece of arithmetic that historically diverged between
:obj:`ezmsg.sigproc.window.Window` (which bins by a *fixed* ``int(bin_duration *
fs)`` sample count) and ``ezmsg.event.rate.EventRate`` (which bins by a
*fractional* ``bin_duration * fs`` with a carry accumulator). At a clean rate
the two coincide; at a real recording rate (e.g. ~30012 Hz) they diverge in both
gain and bin count, so a downstream ``Merge``/``AlignAlongAxis`` of the two
branches never aligns.

:obj:`BinSchedule` makes the boundary rule a *single source of truth*. Any
consumer (dense aggregation, event counting, window segmentation) that drives
its boundaries through one :obj:`BinSchedule`, parameterized identically, is
guaranteed to land on the same grid -- by construction, not by coincidence.

The schedule deliberately holds only *counts and indices*; it never touches
array data. Each consumer keeps its own data carry (raw samples for an arbitrary
aggregator, a scalar partial-sum for a counter), because the right carry
representation depends on the operation, but the *boundaries* are shared.

Boundary definition
-------------------
With ``spb`` samples per bin (fractional when :attr:`fractional`), the global
sample index of the ``m``-th bin boundary is ``B(m) = int(m * spb)``. Indexing by
the *global* bin index ``m`` (rather than a per-chunk accumulator) makes the
output identical regardless of how the input stream is chunked, and reproduces
``EventRate``'s grid (gain, offset, bin count). The closed form ``int(m * spb)``
is also more numerically stable than an iterative carry accumulator: there is no
drift to accumulate over a long stream.
"""

from dataclasses import dataclass, field

import ezmsg.core as ez
import numpy as np


@dataclass
class BinStep:
    """Result of advancing a :obj:`BinSchedule` by one chunk.

    ``cut_points`` are bin *ends* expressed as indices into the consumer's
    ``work`` array -- i.e. the concatenation of its carried partial-bin samples
    followed by the new chunk. Bin ``i`` spans ``work[starts[i]:cut_points[i]]``
    where ``starts = [0] + cut_points[:-1]``. The samples after the final cut
    (``work[cut_points[-1]:]``) are the new partial bin and become the next
    chunk's carry.
    """

    cut_points: list[int]
    """Bin-end indices into the ``[carry ++ new]`` work array (one per closed bin)."""

    n_bins: int
    """Number of bins that close in this chunk (``== len(cut_points)``)."""

    output_gain: float
    """Gain (seconds per bin) to label the output axis with."""

    output_offset: float
    """Nominal start time of the first bin closing in this chunk. Derived from the
    global bin index, so it is identical regardless of chunking (and matches
    ``EventRate``: ``stream_start + bins_before * output_gain``)."""

    carry_count: int
    """Number of input samples in the still-open partial bin after the last cut.
    The consumer must keep exactly this many trailing ``work`` samples as carry."""


@dataclass
class BinSchedule:
    """Stateful, backend-agnostic schedule of fixed-duration bin boundaries.

    Construct once with the target bin duration and mode, then :meth:`reset`
    with the input sample rate before streaming, and call :meth:`advance` once
    per input chunk. The schedule tracks the global bin index and the carried
    sample count; it never holds array data.

    Args:
        bin_duration: Output bin duration in seconds.
        fractional: If True (default), bins span a *fractional* ``bin_duration *
            fs`` samples; bin widths track the nominal duration and the output
            gain is exactly ``bin_duration`` (matching ``EventRate``). If False,
            bins span a *fixed* ``int(bin_duration * fs)`` samples and the
            output gain is ``int(bin_duration * fs) / fs`` (sample-locked,
            matching :obj:`Window`).
    """

    bin_duration: float
    fractional: bool = True

    fs: float | None = field(default=None, init=False)
    """Input sample rate, set by :meth:`reset`."""

    spb: float = field(default=0.0, init=False)
    """Bin width in input samples (fractional when :attr:`fractional`)."""

    n_bins_done: int = field(default=0, init=False)
    """Count of bins emitted so far across the stream (global bin index)."""

    carry_count: int = field(default=0, init=False)
    """Trailing input samples in the open partial bin, carried across chunks."""

    _out_gain: float = field(default=0.0, init=False)

    def reset(self, fs: float) -> None:
        """(Re)initialize the schedule for an input stream at rate ``fs``.

        Emits the same sub-sample warnings the dense binner did, so callers that
        relied on those messages keep getting them from one place.
        """
        spb = self.bin_duration * fs
        if not self.fractional:
            # Truncate (not round) so the sample-locked grid matches Window's
            # ``int(window_dur * fs)`` exactly for *any* rate, not just when the
            # fractional part happens to be < 0.5.
            spb = float(int(spb))
            if spb < 1.0:
                ez.logger.warning(
                    f"bin_duration {self.bin_duration}s is shorter than one sample at "
                    f"fs={fs}; clamping to 1 sample per bin."
                )
                spb = 1.0
        elif spb < 1.0:
            # Fewer than one input sample per bin means the output rate exceeds
            # the input rate. Bins would span zero samples (NaN aggregates), so
            # this is a misconfiguration rather than a supported mode.
            ez.logger.warning(
                f"bin_duration {self.bin_duration}s is shorter than one sample "
                f"at fs={fs} (samples_per_bin={spb:.4g} < 1); output will contain empty "
                f"bins. Increase bin_duration or reduce the input rate."
            )
        self.fs = fs
        self.spb = spb
        self._out_gain = self.bin_duration if self.fractional else spb / fs
        self.n_bins_done = 0
        self.carry_count = 0

    @property
    def output_gain(self) -> float:
        """Gain (seconds per bin) of the output axis for this schedule."""
        return self._out_gain

    def advance(self, n_new: int, in_offset: float, gain_in: float) -> BinStep:
        """Advance the schedule by a chunk of ``n_new`` new input samples.

        Args:
            n_new: Number of new input samples in this chunk.
            in_offset: ``offset`` of the incoming chunk's first *new* sample
                (i.e. ``message.axes[axis].offset``).
            gain_in: ``gain`` of the input axis (``1 / fs``).

        Returns:
            A :obj:`BinStep`. The schedule's :attr:`n_bins_done` and
            :attr:`carry_count` are updated to reflect the bins it reports closed.
        """
        spb = self.spb
        m_done = self.n_bins_done
        n_carry = self.carry_count

        # Global sample index of carry[0] (== bin boundary B(m_done)); the
        # incoming chunk's first new sample follows the carried samples.
        in_done = int(m_done * spb)
        data0_global = in_done + n_carry
        avail_end = data0_global + n_new

        # Nominal start time of the first (m_done-th) output bin. Derived from the
        # global bin index, so it is identical regardless of chunking and matches
        # EventRate's nominal offset (== stream_start + m_done * output_gain).
        output_offset = in_offset - data0_global * gain_in + m_done * self._out_gain

        # Global bin boundaries B(m) = int(m * spb) for the bins that complete
        # within the data available so far. Indexing by the *global* m (not a
        # per-chunk offset) is what makes the output chunk-invariant.
        cut_points: list[int] = []
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
            cut_points = (b_global - in_done).tolist()

        work_len = n_carry + n_new
        if cut_points:
            self.carry_count = work_len - cut_points[-1]
            self.n_bins_done = m_done + len(cut_points)
        else:
            # No bin completes; everything seen so far stays in the open bin.
            self.carry_count = work_len

        return BinStep(
            cut_points=cut_points,
            n_bins=len(cut_points),
            output_gain=self._out_gain,
            output_offset=output_offset,
            carry_count=self.carry_count,
        )
