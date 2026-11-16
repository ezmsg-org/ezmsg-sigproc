import asyncio
import logging

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, compute_config_id

from ezmsg.sigproc.streamstamp import StreamStampSettings, StreamStampTransformer

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
    proc = StreamStampTransformer(tolerance=0.002)
    jitter = np.random.default_rng(0).uniform(-0.001, 0.001, 50)
    assert segments(proc, [chunk(i * 0.1 + j) for i, j in enumerate(jitter)]) == [0] * 50


def test_forward_and_backward_jumps_start_new_segments():
    proc = StreamStampTransformer(tolerance=0.002)
    offsets = [0.0, 0.1, 0.2, 5.0, 5.1, 1.0, 1.1]  # gap forward, then a step back
    assert segments(proc, [chunk(o) for o in offsets]) == [0, 0, 0, 1, 1, 2, 2]


def test_upstream_segment_change_starts_a_new_segment_and_keeps_counting():
    proc = StreamStampTransformer(tolerance=0.002)
    ups = [5, 5, 6, 6, 6]
    out = segments(proc, [chunk(i * 0.1, segment=u) for i, u in enumerate(ups)])
    # The tracker survives the base class's reset on an upstream change: a gap after it is still counted.
    out.append(proc(chunk(9.0, segment=6)).segment)
    assert out == [0, 0, 1, 1, 1, 2]


def test_upstream_change_and_gap_together_count_once():
    proc = StreamStampTransformer(tolerance=0.002)
    assert segments(proc, [chunk(0.0, segment=0), chunk(9.0, segment=1)]) == [0, 1]


def test_sample_rate_change_is_a_new_segment():
    proc = StreamStampTransformer(tolerance=0.002)
    # Offsets continue exactly, so only the layout change can account for the new segment.
    assert segments(proc, [chunk(0.0), chunk(0.1), chunk(0.2, fs=200.0), chunk(0.25, fs=200.0)]) == [0, 0, 1, 1]


def test_empty_chunk_does_not_break_the_stream():
    proc = StreamStampTransformer(tolerance=0.002)
    assert segments(proc, [chunk(0.0), chunk(0.1, n=0), chunk(0.1), chunk(0.2)]) == [0, 0, 0, 0]


def test_coordinate_stream_axis_keeps_its_segment_but_gets_a_config_id():
    proc = StreamStampTransformer(tolerance=0.002)
    msg = AxisArray(
        np.zeros((3, 2)),
        dims=["time", "ch"],
        axes={"time": AxisArray.CoordinateAxis(np.array([0.0, 0.5, 7.0]), dims=["time"], unit="s")},
        stream_dim="time",
        segment=4,
    )
    out = proc(msg)
    assert out.segment == 4 and out.config_id == compute_config_id(msg) != 0
    np.testing.assert_array_equal(out.data, msg.data)


def test_tolerance_update_applies_without_a_break():
    proc = StreamStampTransformer(tolerance=0.002)
    out = segments(proc, [chunk(0.0), chunk(0.1)])
    proc.update_settings(StreamStampSettings(tolerance=0.1))
    out += segments(proc, [chunk(0.25), chunk(0.35)])  # 50 ms late: within the new tolerance
    assert out == [0, 0, 0, 0]


def test_async_path_matches():
    proc = StreamStampTransformer(tolerance=0.002)

    async def run():
        return [(await proc.__acall__(chunk(o, segment=u))).segment for o, u in [(0, 0), (0.1, 1), (5, 1)]]

    assert asyncio.run(run()) == [0, 1, 2]


def test_tolerance_is_required():
    with pytest.raises(TypeError):
        StreamStampSettings()


def test_tolerance_none_keeps_segments_but_stamps_config_id():
    proc = StreamStampTransformer(tolerance=None)
    msgs = [chunk(0.0, segment=7), chunk(5.0, segment=7), chunk(1.0, segment=9)]
    outs = [proc(m) for m in msgs]
    assert [o.segment for o in outs] == [7, 7, 9]
    assert {o.config_id for o in outs} == {compute_config_id(msgs[0])}


def test_config_id_follows_the_layout():
    proc = StreamStampTransformer(tolerance=0.002)
    msgs = [chunk(0.0), chunk(0.1), chunk(0.2, fs=200.0), chunk(0.25, fs=200.0)]
    ids = [proc(m).config_id for m in msgs]
    assert ids == [compute_config_id(m) for m in msgs]
    assert ids[0] == ids[1] != ids[2] == ids[3] != 0


def test_config_id_is_computed_only_on_a_layout_change(monkeypatch):
    import ezmsg.sigproc.streamstamp as mod

    calls = []
    monkeypatch.setattr(mod, "compute_config_id", lambda m: calls.append(m) or 123)
    proc = StreamStampTransformer(tolerance=0.002)
    assert {proc(chunk(i * 0.1)).config_id for i in range(10)} == {123}
    assert len(calls) == 1


def test_declared_config_id_is_kept():
    proc = StreamStampTransformer(tolerance=0.002)
    for i, cid in enumerate([5, 5, 6]):
        m = chunk(i * 0.1)
        m.config_id = cid
        assert proc(m).config_id == cid
    # Stops declaring without a layout change: gets the computed id.
    assert proc(chunk(0.3)).config_id == compute_config_id(chunk(0.3))


def test_does_not_warn_about_undeclared_input(caplog):
    from ezmsg.baseproc import stateful

    stateful._warned_undeclared.discard(StreamStampTransformer)  # as if first in the process
    with caplog.at_level(logging.WARNING, logger="ezmsg.baseproc.stateful"):
        proc = StreamStampTransformer(tolerance=0.002)
        for i in range(3):
            proc(chunk(i * 0.1))
    assert not [r for r in caplog.records if r.name == "ezmsg.baseproc.stateful"]


def test_detection_can_be_switched_off_and_on_while_running():
    proc = StreamStampTransformer(tolerance=0.002)
    assert segments(proc, [chunk(0.0), chunk(5.0)]) == [0, 1]
    proc.update_settings(StreamStampSettings(tolerance=None))
    assert segments(proc, [chunk(9.0, segment=4), chunk(20.0, segment=4)]) == [4, 4]
    proc.update_settings(StreamStampSettings(tolerance=0.002))
    # The jump from the last chunk seen before or during the pause is not a gap: counting starts afresh.
    assert segments(proc, [chunk(50.0, segment=4), chunk(50.1, segment=4), chunk(60.0, segment=4)]) == [0, 0, 1]


def test_config_id_follows_a_dtype_or_unit_change():
    from ezmsg.util.messages.util import replace

    proc = StreamStampTransformer(tolerance=None)
    a = chunk(0.0)
    b = replace(chunk(0.1), data=np.zeros((N, 2), dtype=np.float32))
    c = replace(b, axes={"time": replace(b.axes["time"], unit="ms")})
    msgs = [a, b, c]
    ids = [proc(m).config_id for m in msgs]
    assert ids == [compute_config_id(m) for m in msgs]
    assert len(set(ids)) == 3
