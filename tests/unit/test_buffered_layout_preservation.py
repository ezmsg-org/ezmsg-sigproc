"""Buffered processors preserve the input's dimension order.

`HybridAxisArrayBuffer` adopts the layout of the first message it sees rather than
normalizing the target axis to the front, so a time-last (``["ch", "time"]``)
pipeline stays time-last end to end -- no transpose per chunk, and samples stay
contiguous for the filters downstream.

Each test drives the same data through a processor twice, once per layout, and
asserts the outputs are transposes of each other. That pins both halves of the
contract: the layout is preserved, *and* preserving it does not change any values.
"""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.align import AlignAlongAxisProcessor, AlignAlongAxisSettings
from ezmsg.sigproc.resample import ResampleProcessor, ResampleSettings
from ezmsg.sigproc.sampler import SamplerSettings, SamplerTransformer
from ezmsg.sigproc.util.message import SampleTriggerMessage

FS = 100.0
N_CH = 3


def _msg(block: np.ndarray, offset: float, time_last: bool, key: str = "test") -> AxisArray:
    """Wrap a (n_time, N_CH) block, laid out per ``time_last``."""
    data = np.ascontiguousarray(block.T) if time_last else block
    dims = ["ch", "time"] if time_last else ["time", "ch"]
    return AxisArray(
        data=data,
        dims=dims,
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=FS, offset=offset),
                "ch": AxisArray.CoordinateAxis(data=np.arange(N_CH).astype(str), dims=["ch"]),
            }
        ),
        key=key,
    )


def _time_first(msg: AxisArray) -> np.ndarray:
    """Output data as (n_time, N_CH) regardless of how it was laid out."""
    arr = np.asarray(msg.data)
    return arr.T if msg.dims == ["ch", "time"] else arr


def _chunks(x: np.ndarray, n: int):
    for i in range(0, x.shape[0], n):
        yield i, x[i : i + n]


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def test_sampler_preserves_layout():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, N_CH))
    period = (-0.05, 0.25)
    trigger_ts = [1.0, 3.0]

    def run(time_last: bool):
        proc = SamplerTransformer(
            settings=SamplerSettings(buffer_dur=2.0, axis="time", period=period, estimate_alignment=True)
        )
        fired = set()
        out = []
        for i, block in _chunks(x, 25):
            offset = i / FS
            # Fire once the buffer already spans the window start (ts + period[0]);
            # firing earlier makes the sampler correctly drop it as unsatisfiable.
            for ts in trigger_ts:
                if ts not in fired and offset >= ts:
                    proc(SampleTriggerMessage(timestamp=ts, period=period, value="t"))
                    fired.add(ts)
            out.extend(proc(_msg(block, offset, time_last)))
        return out

    got_tf, got_tl = run(False), run(True)

    assert len(got_tl) == len(got_tf) == len(trigger_ts)
    for a, b in zip(got_tf, got_tl):
        assert a.dims == ["time", "ch"] and b.dims == ["ch", "time"]
        np.testing.assert_array_equal(_time_first(b), _time_first(a))
        assert b.axes["time"].offset == a.axes["time"].offset


# ---------------------------------------------------------------------------
# Align
# ---------------------------------------------------------------------------


def test_align_preserves_layout_per_input():
    rng = np.random.default_rng(1)
    xa = rng.standard_normal((400, N_CH))
    xb = rng.standard_normal((400, N_CH))

    def run(time_last_a: bool, time_last_b: bool):
        proc = AlignAlongAxisProcessor(settings=AlignAlongAxisSettings(axis="time", buffer_dur=5.0))
        pairs = []
        for (i, ba), (_, bb) in zip(_chunks(xa, 40), _chunks(xb, 40)):
            offset = i / FS
            got = proc(_msg(ba, offset, time_last_a, key="a"))
            if got is not None:
                pairs.append(got)
            got = proc.push_b(_msg(bb, offset, time_last_b, key="b"))
            if got is not None:
                pairs.append(got)
        return pairs

    ref = run(False, False)
    # Each input's layout is tracked independently: A time-last, B time-first.
    mixed = run(True, False)

    assert len(mixed) == len(ref)
    for (ra, rb), (ma, mb) in zip(ref, mixed):
        assert ma.dims == ["ch", "time"], "input A was time-last"
        assert mb.dims == ["time", "ch"], "input B was time-first"
        np.testing.assert_array_equal(_time_first(ma), _time_first(ra))
        np.testing.assert_array_equal(_time_first(mb), _time_first(rb))


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fill_value", ["extrapolate", "last"])
def test_resample_preserves_layout(fill_value):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((500, N_CH))

    def run(time_last: bool):
        proc = ResampleProcessor(
            settings=ResampleSettings(
                axis="time",
                resample_rate=40.0,
                buffer_duration=2.0,
                fill_value=fill_value,
            )
        )
        out = []
        for i, block in _chunks(x, 50):
            proc(_msg(block, i / FS, time_last))
            while True:
                result = next(proc)
                ax_idx = result.get_axis_idx("time") if "time" in result.dims else 0
                if result.data.shape[ax_idx] == 0:
                    break
                out.append(result)
        return out

    got_tf, got_tl = run(False), run(True)

    assert len(got_tl) == len(got_tf) > 0
    for a, b in zip(got_tf, got_tl):
        assert a.dims == ["time", "ch"] and b.dims == ["ch", "time"]
        np.testing.assert_allclose(_time_first(b), _time_first(a), rtol=1e-12, atol=1e-12)
        assert b.axes["time"].offset == a.axes["time"].offset


def test_resample_output_reference_preserves_layout():
    """`output_reference` gathers reference data by index; that must follow the axis."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((400, N_CH))
    ref_dat = rng.standard_normal((400, N_CH))

    def run(time_last: bool):
        proc = ResampleProcessor(
            settings=ResampleSettings(axis="time", resample_rate=None, buffer_duration=2.0, output_reference=True)
        )
        out = []
        for i, block in _chunks(x, 40):
            offset = i / FS
            proc.push_reference(_msg(ref_dat[i : i + 40], offset, time_last))
            proc(_msg(block, offset, time_last))
            while True:
                result = next(proc)
                ax_idx = result.get_axis_idx("time") if "time" in result.dims else 0
                if result.data.shape[ax_idx] == 0:
                    break
                out.append((result, proc.state.reference_output))
        return out

    got_tf, got_tl = run(False), run(True)

    assert len(got_tl) == len(got_tf) > 0
    assert any(r is not None for _, r in got_tf)
    for (a, ra), (b, rb) in zip(got_tf, got_tl):
        np.testing.assert_allclose(_time_first(b), _time_first(a), rtol=1e-12, atol=1e-12)
        assert (ra is None) == (rb is None)
        if ra is not None:
            assert ra.dims == ["time", "ch"] and rb.dims == ["ch", "time"]
            np.testing.assert_allclose(_time_first(rb), _time_first(ra), rtol=1e-12, atol=1e-12)
