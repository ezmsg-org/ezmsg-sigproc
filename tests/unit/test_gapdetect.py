import asyncio

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.gapdetect import GapDetectSettings, GapDetectTransformer

FS = 100.0
N = 10  # samples per chunk: 0.1 s


def chunk(offset, n=N, fs=FS, segment=0):
    return AxisArray(
        np.zeros((n, 2)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=offset)},
        key="src",
        stream_dim="time",
        segment=segment,
    )


def segments(proc, msgs):
    return [proc(m).segment for m in msgs]


def test_continuous_stream_stays_in_one_segment():
    proc = GapDetectTransformer(tolerance=0.002)
    jitter = np.random.default_rng(0).uniform(-0.001, 0.001, 50)
    assert segments(proc, [chunk(i * 0.1 + j) for i, j in enumerate(jitter)]) == [0] * 50


def test_forward_and_backward_jumps_start_new_segments():
    proc = GapDetectTransformer(tolerance=0.002)
    offsets = [0.0, 0.1, 0.2, 5.0, 5.1, 1.0, 1.1]  # gap forward, then a step back
    assert segments(proc, [chunk(o) for o in offsets]) == [0, 0, 0, 1, 1, 2, 2]


def test_upstream_segment_change_starts_a_new_segment_and_keeps_counting():
    proc = GapDetectTransformer(tolerance=0.002)
    ups = [5, 5, 6, 6, 6]
    out = segments(proc, [chunk(i * 0.1, segment=u) for i, u in enumerate(ups)])
    # The tracker survives the base class's reset on an upstream change: a gap after it is still counted.
    out.append(proc(chunk(9.0, segment=6)).segment)
    assert out == [0, 0, 1, 1, 1, 2]


def test_upstream_change_and_gap_together_count_once():
    proc = GapDetectTransformer(tolerance=0.002)
    assert segments(proc, [chunk(0.0, segment=0), chunk(9.0, segment=1)]) == [0, 1]


def test_sample_rate_change_is_a_new_segment():
    proc = GapDetectTransformer(tolerance=0.002)
    # Offsets continue exactly, so only the layout change can account for the new segment.
    assert segments(proc, [chunk(0.0), chunk(0.1), chunk(0.2, fs=200.0), chunk(0.25, fs=200.0)]) == [0, 0, 1, 1]


def test_empty_chunk_does_not_break_the_stream():
    proc = GapDetectTransformer(tolerance=0.002)
    assert segments(proc, [chunk(0.0), chunk(0.1, n=0), chunk(0.1), chunk(0.2)]) == [0, 0, 0, 0]


def test_coordinate_stream_axis_passes_through_untouched():
    proc = GapDetectTransformer(tolerance=0.002)
    msg = AxisArray(
        np.zeros((3, 2)),
        dims=["time", "ch"],
        axes={"time": AxisArray.CoordinateAxis(np.array([0.0, 0.5, 7.0]), dims=["time"], unit="s")},
        stream_dim="time",
        segment=4,
    )
    assert proc(msg) is msg


def test_tolerance_update_applies_without_a_break():
    proc = GapDetectTransformer(tolerance=0.002)
    out = segments(proc, [chunk(0.0), chunk(0.1)])
    proc.update_settings(GapDetectSettings(tolerance=0.1))
    out += segments(proc, [chunk(0.25), chunk(0.35)])  # 50 ms late: within the new tolerance
    assert out == [0, 0, 0, 0]


def test_async_path_matches():
    proc = GapDetectTransformer(tolerance=0.002)

    async def run():
        return [(await proc.__acall__(chunk(o, segment=u))).segment for o, u in [(0, 0), (0.1, 1), (5, 1)]]

    assert asyncio.run(run()) == [0, 1, 2]


def test_tolerance_is_required():
    with pytest.raises(TypeError):
        GapDetectSettings()
