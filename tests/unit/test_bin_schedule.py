"""Pin :obj:`BinSchedule` to the two grids it must reproduce.

The whole point of the shared schedule is that, parameterized identically, it
lands on the *same* bin boundaries as the two units that historically disagreed:

* ``fractional=True``  must match ``ezmsg.event.rate.EventRate`` (which bins by a
  fractional ``bin_duration * fs`` with a carry accumulator).
* ``fractional=False`` must match ``ezmsg.sigproc.window.Window`` (which bins by
  ``int(bin_duration * fs)`` fixed samples).

``EventRate`` lives in a separate package that isn't a test dependency here, so
its boundary algorithm is *ported faithfully below* as ``eventrate_bin_ends`` (a
line-for-line port of ``BinnedKernelActivation._process_events``'s boundary math
in ezmsg-event ``kernel_activation.py``) and we assert the schedule matches it.
The corresponding cross-package test against the *real* ``EventRate`` lives in
ezmsg-event, where the import is available.
"""

import numpy as np
import pytest

from ezmsg.sigproc.util.binning import BinSchedule


# --- Faithful port of EventRate's per-chunk boundary algorithm ----------------
# Mirrors ezmsg.event.kernel_activation.BinnedKernelActivation._process_events:
#   samples_per_bin = bin_duration * fs
#   total = n + accumulator;  n_bins = int(total / spb)
#   first_bin_end = spb - accumulator;  ends = (first_bin_end + arange(n_bins)*spb).astype(int)
#   accumulator = total - n_bins * spb
#   output_offset = input_offset - accumulator_before / fs
def eventrate_grid(chunk_lengths, fs, bin_duration):
    """Return (global_bin_end_samples, per_nonempty_chunk_offsets) the way
    EventRate would compute them for the given chunking (stream_start == 0)."""
    spb = bin_duration * fs
    acc = 0.0
    global_start = 0
    ends: list[int] = []
    offsets: list[float] = []
    for n in chunk_lengths:
        total = n + acc
        n_bins = int(total / spb)
        in_offset = global_start / fs
        if n_bins > 0:
            first_bin_end = spb - acc
            bin_end_samples = (first_bin_end + np.arange(n_bins) * spb).astype(np.int64)
            ends.extend(global_start + int(be) for be in bin_end_samples)
            offsets.append(in_offset - acc / fs)
            acc = total - n_bins * spb
        else:
            acc = total
        global_start += n
    return ends, offsets


def schedule_grid(chunk_lengths, fs, bin_duration, fractional):
    """Drive a BinSchedule over the same chunking and return the global bin-end
    samples and the per-nonempty-chunk output offsets."""
    sched = BinSchedule(bin_duration=bin_duration, fractional=fractional)
    sched.reset(fs)
    gain_in = 1.0 / fs
    global_start = 0
    ends: list[int] = []
    offsets: list[float] = []
    for n in chunk_lengths:
        m_before = sched.n_bins_done
        step = sched.advance(n_new=n, in_offset=global_start * gain_in, gain_in=gain_in)
        # Global bin end of bin m (1-indexed global) is int(m * spb).
        ends.extend(int(m * sched.spb) for m in range(m_before + 1, sched.n_bins_done + 1))
        if step.n_bins > 0:
            offsets.append(step.output_offset)
        global_start += n
    return ends, offsets


@pytest.mark.parametrize("fs", [30000.0, 30012.0, 30030.0])
@pytest.mark.parametrize(
    "chunk_lengths",
    [
        [50000],  # single big chunk
        [1] * 5000,  # worst-case fragmentation
        [7] * 700 + [101] * 90,  # uneven
        [777] * 64,  # uneven vs spb
    ],
)
def test_fractional_matches_eventrate_grid(fs, chunk_lengths):
    """fractional=True reproduces EventRate's bin boundaries and offsets exactly,
    for any chunking and any (including off-nominal) rate."""
    bin_duration = 0.02
    ref_ends, ref_offsets = eventrate_grid(chunk_lengths, fs, bin_duration)
    got_ends, got_offsets = schedule_grid(chunk_lengths, fs, bin_duration, fractional=True)

    assert got_ends == ref_ends
    assert len(got_offsets) == len(ref_offsets)
    np.testing.assert_allclose(got_offsets, ref_offsets, rtol=0, atol=1e-12)


@pytest.mark.parametrize("fs", [30000.0, 30012.0, 30030.0])
def test_fractional_grid_is_chunk_invariant(fs):
    """The fractional grid does not depend on how the stream is chunked."""
    bin_duration = 0.02
    whole, _ = schedule_grid([50000], fs, bin_duration, fractional=True)
    frag, _ = schedule_grid([1] * 50000, fs, bin_duration, fractional=True)
    assert whole == frag


@pytest.mark.parametrize("fs", [30000.0, 30012.0, 30030.0])
def test_integer_mode_matches_window_boundaries(fs):
    """fractional=False bins on Window's grid: fixed int(bin_duration*fs) samples,
    boundaries at multiples of that count.

    fs=30030 is the discriminating case: 0.02*30030 = 600.6, so Window's
    int()-truncation gives 600 while round() would give 601. The schedule must
    truncate to stay on Window's grid.
    """
    bin_duration = 0.02
    window_samples = int(bin_duration * fs)  # exactly Window's window_dur sizing
    n_total = 50000

    ends, _ = schedule_grid([n_total], fs, bin_duration, fractional=False)
    n_bins = n_total // window_samples
    expected = [m * window_samples for m in range(1, n_bins + 1)]
    assert ends == expected

    sched = BinSchedule(bin_duration=bin_duration, fractional=False)
    sched.reset(fs)
    assert sched.spb == float(window_samples)
    assert sched.output_gain == pytest.approx(window_samples / fs)


def test_carry_count_tracks_open_bin():
    """carry_count equals the number of samples in the still-open partial bin."""
    fs = 1000.0  # spb = 20 samples per 0.02 s bin
    sched = BinSchedule(bin_duration=0.02, fractional=True)
    sched.reset(fs)

    step = sched.advance(n_new=50, in_offset=0.0, gain_in=1.0 / fs)
    # 50 samples -> 2 full bins (40 samples) + 10 leftover.
    assert step.n_bins == 2
    assert step.carry_count == 10
    assert sched.carry_count == 10

    step = sched.advance(n_new=10, in_offset=50 / fs, gain_in=1.0 / fs)
    # 10 carried + 10 new = 20 -> exactly 1 more bin, no leftover.
    assert step.n_bins == 1
    assert step.carry_count == 0
