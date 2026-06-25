import asyncio
import copy
import logging

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.modify import ModifyAxisSettings, ModifyAxisTransformer
from frozendict import frozendict

from ezmsg.sigproc.aggregate import AggregateSettings, AggregateTransformer, AggregationFunction
from ezmsg.sigproc.binned_aggregate import (
    BinnedAggregate,
    BinnedAggregateSettings,
    BinnedAggregateTransformer,
)
from ezmsg.sigproc.math.pow import PowSettings, PowTransformer
from ezmsg.sigproc.window import WindowSettings, WindowTransformer
from tests.helpers.empty_time import check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg
from tests.helpers.util import assert_messages_equal


def _sig_msgs(sig: np.ndarray, fs: float, block_size: int) -> list[AxisArray]:
    n = sig.shape[0]
    msgs = []
    for start in range(0, n, block_size):
        chunk = sig[start : start + block_size]
        msgs.append(
            AxisArray(
                data=chunk,
                dims=["time", "ch"],
                axes=frozendict(
                    {
                        "time": AxisArray.TimeAxis(fs=fs, offset=start / fs),
                        "ch": AxisArray.CoordinateAxis(data=np.arange(sig.shape[1]).astype(str), dims=["ch"]),
                    }
                ),
                key="test_binned_aggregate",
            )
        )
    return msgs


def _ref_binned(x: np.ndarray, spb: float, op=np.mean) -> np.ndarray:
    """Ground truth: aggregate x over global bins [int((m-1)*spb), int(m*spb))."""
    n = x.shape[0]
    n_bins = int(n / spb)
    return np.stack([op(x[int((m - 1) * spb) : int(m * spb)], axis=0) for m in range(1, n_bins + 1)], axis=0)


def _run(proc: BinnedAggregateTransformer, msgs: list[AxisArray]) -> list[AxisArray]:
    out = []
    for msg in msgs:
        res = proc(msg)
        if res.data.size:
            out.append(res)
    return out


@pytest.mark.parametrize("block_size", [1, 7, 64, 100000])
@pytest.mark.parametrize("fs", [30000.0, 30012.0])
def test_matches_global_bin_reference(block_size: int, fs: float):
    """Output equals the global-bin numpy reference regardless of chunking."""
    bin_dur = 0.02
    sig = np.random.default_rng(0).standard_normal((30000, 3))
    spb = bin_dur * fs

    in_msgs = _sig_msgs(sig, fs, block_size)
    backup = [copy.deepcopy(m) for m in in_msgs]

    proc = BinnedAggregateTransformer(
        axis="time", bin_duration=bin_dur, operation=AggregationFunction.MEAN, fractional=True
    )
    out = _run(proc, in_msgs)

    assert_messages_equal(in_msgs, backup)  # input not mutated

    data = np.concatenate([m.data for m in out], axis=0)
    ref = _ref_binned(sig, spb, op=np.mean)
    assert data.shape == ref.shape
    np.testing.assert_allclose(data, ref, rtol=0, atol=1e-12)

    # Fractional grid is labelled with the nominal gain and a zero offset.
    assert out[0].axes["time"].gain == pytest.approx(bin_dur)
    assert out[0].axes["time"].offset == pytest.approx(0.0)


@pytest.mark.parametrize("fs", [30000.0, 30012.0])
def test_chunk_invariance(fs: float):
    """Single-chunk and heavily-fragmented streams give identical output."""
    sig = np.random.default_rng(1).standard_normal((50000, 2))
    proc_a = BinnedAggregateTransformer(axis="time", bin_duration=0.02, fractional=True)
    proc_b = BinnedAggregateTransformer(axis="time", bin_duration=0.02, fractional=True)

    whole = np.concatenate([m.data for m in _run(proc_a, _sig_msgs(sig, fs, 50000))], axis=0)
    # Worst-case fragmentation: one sample per message.
    fragmented = np.concatenate([m.data for m in _run(proc_b, _sig_msgs(sig, fs, 1))], axis=0)

    assert whole.shape == fragmented.shape
    np.testing.assert_array_equal(whole, fragmented)


@pytest.mark.parametrize("fs", [30000.0, 30012.0])
def test_integer_mode_matches_window(fs: float):
    """fractional=False reproduces the legacy Window+Aggregate(mean) grid+values."""
    bin_dur = 0.02
    sig = np.random.default_rng(2).standard_normal((30000, 4))

    # Legacy SBP-style path: square -> window(bins) -> mean -> rename.
    win = [
        PowTransformer(PowSettings(exponent=2.0)),
        WindowTransformer(
            WindowSettings(axis="time", newaxis="win", window_dur=bin_dur, window_shift=bin_dur, zero_pad_until="none")
        ),
        AggregateTransformer(AggregateSettings(axis="time", operation=AggregationFunction.MEAN)),
        ModifyAxisTransformer(ModifyAxisSettings(name_map={"win": "time"})),
    ]
    msg = _sig_msgs(sig, fs, 30000)[0]
    w = msg
    for t in win:
        w = t(w)

    proc = BinnedAggregateTransformer(axis="time", bin_duration=bin_dur, fractional=False)
    b = proc(_sig_msgs(sig**2, fs, 30000)[0])

    assert b.data.shape == w.data.shape
    assert b.axes["time"].gain == pytest.approx(w.axes["time"].gain)
    np.testing.assert_allclose(b.data, w.data, rtol=0, atol=1e-12)


def test_sum_operation():
    """SUM aggregation sums each bin (basis for delegating EventRate's count)."""
    fs = 1000.0
    sig = np.ones((100, 2))
    proc = BinnedAggregateTransformer(
        axis="time", bin_duration=0.02, operation=AggregationFunction.SUM, fractional=True
    )
    out = proc(_sig_msgs(sig, fs, 100)[0])
    # spb = 20 -> 5 bins of 20 ones -> sum 20 each.
    assert out.data.shape == (5, 2)
    np.testing.assert_allclose(out.data, 20.0)


def test_empty_message_propagates():
    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, fractional=True)
    result = proc(make_empty_msg())
    check_empty_result(result)
    check_state_not_corrupted(proc, make_msg())


def test_output_offset_tracks_stream_grid():
    """Each output message is labelled with the nominal start time of its first
    bin (stream_start + bins_before * bin_duration), independent of chunking.

    This is the property the module exists for: it is what lets a downstream
    Merge align this branch with the EventRate branch.
    """
    fs = 30012.0
    bin_dur = 0.02
    stream_start = 12.5  # arbitrary non-zero stream offset
    sig = np.random.default_rng(7).standard_normal((30000, 2))

    msgs = []
    for start in range(0, sig.shape[0], 777):  # uneven chunking
        chunk = sig[start : start + 777]
        msgs.append(
            AxisArray(
                data=chunk,
                dims=["time", "ch"],
                axes=frozendict(
                    {
                        "time": AxisArray.TimeAxis(fs=fs, offset=stream_start + start / fs),
                        "ch": AxisArray.CoordinateAxis(data=np.arange(2).astype(str), dims=["ch"]),
                    }
                ),
                key="offset_grid",
            )
        )

    proc = BinnedAggregateTransformer(axis="time", bin_duration=bin_dur, fractional=True)
    out = _run(proc, msgs)

    bins_before = 0
    for m in out:
        assert m.axes["time"].gain == pytest.approx(bin_dur)
        expected_offset = stream_start + bins_before * bin_dur
        assert m.axes["time"].offset == pytest.approx(expected_offset)
        bins_before += m.data.shape[0]


def test_integer_mode_subsample_clamps_with_warning(caplog):
    """fractional=False with a sub-sample bin_duration clamps to 1 sample/bin
    and warns, rather than producing zero-width bins."""
    fs = 1000.0
    sig = np.ones((50, 2))
    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.0001, fractional=False)
    with caplog.at_level(logging.WARNING, logger="ezmsg"):
        out = proc(_sig_msgs(sig, fs, 50)[0])

    assert any("clamping to 1 sample" in r.message for r in caplog.records)
    # One bin per input sample; mean of a single sample is itself.
    assert out.data.shape == (50, 2)
    assert out.axes["time"].gain == pytest.approx(1.0 / fs)
    np.testing.assert_allclose(out.data, 1.0)


def test_fractional_subsample_warns(caplog):
    """fractional=True with a sub-sample bin_duration warns about empty bins."""
    fs = 1000.0
    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.0001, fractional=True)
    with caplog.at_level(logging.WARNING, logger="ezmsg"):
        proc(_sig_msgs(np.ones((50, 2)), fs, 50)[0])
    assert any("shorter than one sample" in r.message for r in caplog.records)


def test_state_resets_on_fs_change():
    """A change in the input gain (fs) restarts the binning grid and discards
    any carried partial bin from the previous stream."""
    bin_dur = 0.02
    proc = BinnedAggregateTransformer(axis="time", bin_duration=bin_dur, fractional=True)

    # First stream at fs1: 30 samples (spb=600) leaves a partial bin in carry.
    proc(_sig_msgs(np.ones((30, 2)), 30000.0, 30)[0])
    assert proc._state.carry is not None  # partial bin pending

    # Second stream at a different fs forces _reset_state.
    fs2 = 30012.0
    out = proc(_sig_msgs(np.ones((30000, 2)), fs2, 30000)[0])

    assert proc._state.fs == fs2
    assert proc._state.samples_per_bin == pytest.approx(bin_dur * fs2)
    # Output offset reflects the new stream start (0.0), not stale fs1 state.
    assert out.axes["time"].offset == pytest.approx(0.0)
    assert np.all(np.isfinite(out.data))


def test_unit_suppresses_empty_publishes():
    """The BinnedAggregate Unit publishes nothing when a chunk closes no bin,
    but publishes once a bin completes."""
    fs = 1000.0  # spb = 20 samples per 0.02s bin
    unit = BinnedAggregate(BinnedAggregateSettings(axis="time", bin_duration=0.02))
    unit.create_processor()

    async def drive(msg):
        return [m async for m in unit.on_signal(msg)]

    # 10 samples: no bin completes -> nothing published.
    assert asyncio.run(drive(_sig_msgs(np.ones((10, 2)), fs, 10)[0])) == []
    # 40 more samples (50 total -> 2 bins): a publish occurs.
    published = asyncio.run(drive(_sig_msgs(np.ones((40, 2)), fs, 40)[0]))
    assert len(published) == 1
    _, msg_out = published[0]
    assert msg_out.data.shape[0] > 0
