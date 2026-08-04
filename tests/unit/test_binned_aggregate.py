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


@pytest.mark.parametrize("fs", [30000.0, 30012.0, 30030.0])
def test_integer_mode_matches_window(fs: float):
    """fractional=False reproduces the legacy Window+Aggregate(mean) grid+values.

    fs=30030 (0.02*fs = 600.6) is the case that distinguishes Window's
    int()-truncation from round(): the integer grid must truncate to match.
    """
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

    assert proc._state.schedule.fs == fs2
    assert proc._state.schedule.spb == pytest.approx(bin_dur * fs2)
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


# ---- multi-operation (trailing metric axis) --------------------------------

MINMAX = (AggregationFunction.MIN, AggregationFunction.MAX)


def test_tuple_operation_stacks_on_a_trailing_axis():
    fs = 1000.0
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((100, 2))
    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=MINMAX, fractional=True)
    out = proc(_sig_msgs(sig, fs, 100)[0])

    # spb = 20 -> 5 bins, 2 channels, 2 metrics.
    assert out.dims == ["time", "ch", "metric"]
    assert out.data.shape == (5, 2, 2)
    np.testing.assert_allclose(out.data[..., 0], _ref_binned(sig, 20.0, np.min))
    np.testing.assert_allclose(out.data[..., 1], _ref_binned(sig, 20.0, np.max))


def test_metric_axis_is_labelled_from_the_enum():
    """Consumers read names, not positions -- and the names are the enum values."""
    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=MINMAX)
    out = proc(_sig_msgs(np.ones((100, 2)), 1000.0, 100)[0])
    metric = out.axes["metric"]
    assert list(metric.data) == ["min", "max"]
    assert metric.dims == ["metric"]


def test_newaxis_is_configurable():
    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=MINMAX, newaxis="bound")
    out = proc(_sig_msgs(np.ones((100, 2)), 1000.0, 100)[0])
    assert out.dims == ["time", "ch", "bound"]
    assert list(out.axes["bound"].data) == ["min", "max"]


def test_scalar_operation_shape_is_unchanged():
    """The trailing axis appears only for a tuple, so existing streams are
    untouched."""
    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=AggregationFunction.MEAN)
    out = proc(_sig_msgs(np.ones((100, 2)), 1000.0, 100)[0])
    assert out.dims == ["time", "ch"]
    assert out.data.shape == (5, 2)
    assert "metric" not in out.axes


def test_single_element_tuple_still_produces_the_axis():
    """Shape follows the type of `operation`, not the count, so a caller that
    builds its tuple programmatically gets a stable shape."""
    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=(AggregationFunction.MAX,))
    out = proc(_sig_msgs(np.ones((100, 2)), 1000.0, 100)[0])
    assert out.dims == ["time", "ch", "metric"]
    assert out.data.shape == (5, 2, 1)
    assert list(out.axes["metric"].data) == ["max"]


def test_multi_op_matches_running_each_op_separately():
    """The whole point of a tuple is that the bins are identical; prove they are
    by comparing against separate single-op transformers on the same chunking."""
    fs = 1000.0
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((503, 3))
    msgs = _sig_msgs(sig, fs, 37)  # chunk size coprime with the bin size

    multi = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=MINMAX)
    only_min = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=AggregationFunction.MIN)
    only_max = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=AggregationFunction.MAX)

    got = np.concatenate([m.data for m in _run(multi, copy.deepcopy(msgs))], axis=0)
    exp_min = np.concatenate([m.data for m in _run(only_min, copy.deepcopy(msgs))], axis=0)
    exp_max = np.concatenate([m.data for m in _run(only_max, copy.deepcopy(msgs))], axis=0)

    np.testing.assert_allclose(got[..., 0], exp_min)
    np.testing.assert_allclose(got[..., 1], exp_max)


def test_multi_op_carries_partial_bins_across_chunks():
    """A peak in a bin split across two messages must survive.

    This is the property that makes the envelope usable for spike-bearing data:
    the extremes are exact regardless of where the message boundary falls.
    """
    fs = 1000.0
    sig = np.zeros((60, 1))
    sig[25, 0] = 9.0  # bin 1 (samples 20..39), and chunk boundary is at 30
    sig[35, 0] = -4.0

    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=MINMAX)
    out = np.concatenate([m.data for m in _run(proc, _sig_msgs(sig, fs, 30))], axis=0)

    assert out.shape == (3, 1, 2)
    np.testing.assert_allclose(out[1, 0, 0], -4.0)  # min of the split bin
    np.testing.assert_allclose(out[1, 0, 1], 9.0)  # max of the split bin


def test_multi_op_empty_message_keeps_the_metric_width():
    """An empty payload must still be the right shape, or it will not
    concatenate with the messages around it."""
    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=MINMAX)
    out = proc(_sig_msgs(np.ones((5, 2)), 1000.0, 5)[0])  # < one bin -> empty
    assert out.data.shape == (0, 2, 2)
    assert out.dims == ["time", "ch", "metric"]


def test_multi_op_axis_gain_matches_single_op():
    """The metric axis must not disturb the bin rate."""
    msgs = _sig_msgs(np.ones((100, 2)), 1000.0, 100)
    multi = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=MINMAX)(copy.deepcopy(msgs[0]))
    single = BinnedAggregateTransformer(axis="time", bin_duration=0.02)(copy.deepcopy(msgs[0]))
    assert multi.axes["time"].gain == single.axes["time"].gain
    assert multi.axes["time"].offset == single.axes["time"].offset


def test_multi_op_non_time_axis():
    """Binning a non-leading axis must still append the metric axis at the end."""
    proc = BinnedAggregateTransformer(axis="ch", bin_duration=2.0, operation=MINMAX, fractional=False)
    msg = AxisArray(
        data=np.arange(40, dtype=float).reshape(4, 10),
        dims=["time", "ch"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=10.0, offset=0.0),
                "ch": AxisArray.TimeAxis(fs=1.0, offset=0.0),
            }
        ),
        key="test_binned_aggregate_ch",
    )
    out = proc(msg)
    assert out.dims == ["time", "ch", "metric"]
    # 10 channels binned 2 at a time -> 5 bins.
    assert out.data.shape == (4, 5, 2)
    np.testing.assert_allclose(out.data[0, :, 0], [0, 2, 4, 6, 8])
    np.testing.assert_allclose(out.data[0, :, 1], [1, 3, 5, 7, 9])


def test_metric_axis_is_built_once_not_per_message():
    """It depends only on settings, so every output should carry the same
    object -- both to avoid rebuilding it per message and so a downstream
    identity check on the axis stays cheap."""
    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=MINMAX)
    outs = _run(proc, _sig_msgs(np.ones((200, 2)), 1000.0, 40))
    assert len(outs) > 1
    first = outs[0].axes["metric"]
    assert all(o.axes["metric"] is first for o in outs)


# ---- operations that need the axis coordinates ------------------------------
#
# These three used to be wrong here, because BinnedAggregate ran the raw numpy
# function with no x-coordinates while its sibling RangedAggregate handled them.
# Both now go through aggregate_slices.


def test_trapezoid_integrates_against_the_axis_not_sample_counts():
    """Previously returned 19.0 for this input -- an integral in samples."""
    fs = 1000.0
    proc = BinnedAggregateTransformer(
        axis="time", bin_duration=0.02, operation=AggregationFunction.TRAPEZOID, fractional=True
    )
    out = proc(_sig_msgs(np.ones((100, 2)), fs, 100)[0])

    assert out.data.shape == (5, 2)
    # 20 samples at 1 ms spacing spans 19 intervals; amplitude 1 -> 0.019 s.
    np.testing.assert_allclose(out.data, 0.019)


def test_argmax_returns_a_time_not_a_within_bin_index():
    """Previously returned 0..spb-1. The useful answer is when the peak was."""
    fs = 1000.0
    sig = np.zeros((60, 1))
    sig[5, 0] = 1.0  # bin 0
    sig[27, 0] = 1.0  # bin 1
    sig[59, 0] = 1.0  # bin 2

    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=AggregationFunction.ARGMAX)
    out = proc(_sig_msgs(sig, fs, 60)[0])

    np.testing.assert_allclose(out.data[:, 0], [0.005, 0.027, 0.059])


def test_argmax_time_is_correct_across_a_message_boundary():
    """The coordinate vector has to span the carry, which starts *before* this
    message's offset -- otherwise a peak in a split bin is reported late."""
    fs = 1000.0
    sig = np.zeros((60, 1))
    sig[32, 0] = 1.0  # bin 1 (samples 20..39), split by the 30-sample chunking

    proc = BinnedAggregateTransformer(axis="time", bin_duration=0.02, operation=AggregationFunction.ARGMAX)
    out = np.concatenate([m.data for m in _run(proc, _sig_msgs(sig, fs, 30))], axis=0)

    np.testing.assert_allclose(out[1, 0], 0.032)


def test_argmin_matches_ranged_aggregate_semantics():
    """The two aggregators should answer the same question the same way."""
    from ezmsg.sigproc.aggregate import RangedAggregateSettings, RangedAggregateTransformer

    fs = 1000.0
    rng = np.random.default_rng(3)
    sig = rng.standard_normal((40, 1))
    msg = _sig_msgs(sig, fs, 40)[0]

    binned = BinnedAggregateTransformer(
        axis="time", bin_duration=0.02, operation=AggregationFunction.ARGMIN, fractional=False
    )(copy.deepcopy(msg))
    # Same two groups, expressed as coordinate bands: [0, 0.019] and [0.02, 0.039].
    ranged = RangedAggregateTransformer(
        RangedAggregateSettings(
            axis="time",
            bands=[(0.0, 0.0195), (0.02, 0.0395)],
            operation=AggregationFunction.ARGMIN,
        )
    )(copy.deepcopy(msg))

    np.testing.assert_allclose(binned.data, ranged.data)


def test_tuple_may_mix_value_and_coordinate_operations():
    fs = 1000.0
    sig = np.zeros((40, 1))
    sig[7, 0] = 3.0
    sig[25, 0] = 5.0

    proc = BinnedAggregateTransformer(
        axis="time", bin_duration=0.02, operation=(AggregationFunction.MAX, AggregationFunction.ARGMAX)
    )
    out = proc(_sig_msgs(sig, fs, 40)[0])

    assert list(out.axes["metric"].data) == ["max", "argmax"]
    np.testing.assert_allclose(out.data[:, 0, 0], [3.0, 5.0])  # peak values
    np.testing.assert_allclose(out.data[:, 0, 1], [0.007, 0.025])  # peak times
