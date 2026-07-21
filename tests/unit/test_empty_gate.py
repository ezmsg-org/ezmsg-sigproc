"""Publish gates must swallow only time-empty results, not channel-empty ones.

A unit that suppresses empty publishes (Downsample, BinnedAggregate, Window) or
uses emptiness as a drain-loop terminator (ResampleUnit, ResampleConcat) should
key on the time-like axis, not ``data.size``. A message that is empty only along
other axes -- e.g. an upstream Slicer with ``on_empty="warn"`` removed every
channel while time samples remain -- must still flow so downstream consumers
keep the stream's cadence.
"""

import asyncio

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.binned_aggregate import BinnedAggregate, BinnedAggregateSettings
from ezmsg.sigproc.downsample import Downsample, DownsampleSettings
from ezmsg.sigproc.resampleconcat import ResampleConcat, ResampleConcatSettings
from ezmsg.sigproc.util.message import has_samples_along, is_empty_along
from ezmsg.sigproc.window import Window, WindowSettings
from tests.helpers.empty_time import make_msg


def _null_template() -> AxisArray:
    """ResampleProcessor's pre-init placeholder: no time axis at all."""
    return AxisArray(data=np.array([]), dims=[""], axes={}, key="null")


def _drive(unit, msg):
    async def _run():
        return [m async for m in unit.on_signal(msg)]

    return asyncio.run(_run())


def test_is_empty_along():
    msg = make_msg(n_time=10, n_ch=0)
    assert is_empty_along(msg, ("ch",))
    assert not is_empty_along(msg, ("time",))
    assert is_empty_along(msg, ("time", "ch"))
    # Dims not present in the message are ignored.
    assert not is_empty_along(msg, ("win",))
    msg = make_msg(n_time=0, n_ch=3)
    assert is_empty_along(msg, ("time",))
    assert not is_empty_along(msg, ("ch",))
    msg = make_msg(n_time=0, n_ch=0)
    assert is_empty_along(msg, ("time",))
    assert is_empty_along(msg, ("ch",))


def test_has_samples_along():
    assert has_samples_along(make_msg(n_time=10, n_ch=0), "time")
    assert not has_samples_along(make_msg(n_time=10, n_ch=0), "ch")
    assert not has_samples_along(make_msg(n_time=0, n_ch=3), "time")
    # A message lacking the dim entirely (e.g. the resample pre-init null
    # template) has no samples along it.
    assert not has_samples_along(_null_template(), "time")


def test_downsample_gate():
    unit = Downsample(DownsampleSettings(axis="time", target_rate=50.0))
    unit.create_processor()
    # Zero channels but nonzero time: the (5, 0) result must be published.
    published = _drive(unit, make_msg(n_time=10, n_ch=0, fs=100.0))
    assert len(published) == 1
    _, msg_out = published[0]
    assert msg_out.data.shape == (5, 0)
    # Zero time: swallowed.
    assert _drive(unit, make_msg(n_time=0, n_ch=0, fs=100.0)) == []


def test_binned_aggregate_gate():
    unit = BinnedAggregate(BinnedAggregateSettings(axis="time", bin_duration=0.02))
    unit.create_processor()
    # 40 samples @ 1 kHz close 2 bins; zero channels must not suppress the publish.
    published = _drive(unit, make_msg(n_time=40, n_ch=0, fs=1000.0))
    assert len(published) == 1
    _, msg_out = published[0]
    assert msg_out.data.shape == (2, 0)
    # 10 more samples close no bin: swallowed.
    assert _drive(unit, make_msg(n_time=10, n_ch=0, fs=1000.0)) == []


def test_window_gate():
    settings = WindowSettings(axis="time", newaxis="win", window_dur=0.1, window_shift=0.1)
    unit = Window(settings)
    unit.create_processor()
    # 30 samples @ 100 Hz -> 3 complete 10-sample windows; zero channels must
    # not suppress the publish.
    published = _drive(unit, make_msg(n_time=30, n_ch=0, fs=100.0))
    assert len(published) == 1
    _, msg_out = published[0]
    assert msg_out.dims == ["win", "time", "ch"]
    assert msg_out.data.shape == (3, 10, 0)
    # 5 more samples complete no window: swallowed.
    assert _drive(unit, make_msg(n_time=5, n_ch=0, fs=100.0)) == []


class _StubProcessor:
    """Stands in for ResampleConcatProcessor: yields canned chunks."""

    def __init__(self, items):
        self._items = iter(items)

    def __next__(self):
        return next(self._items)


def test_resampleconcat_drain_gate():
    unit = ResampleConcat(ResampleConcatSettings())
    # A chunk that is empty along the concatenated feature axis is a real
    # output; only an empty resample ("time") axis terminates the drain.
    unit.processor = _StubProcessor(
        [
            make_msg(n_time=5, n_ch=0),
            make_msg(n_time=0, n_ch=3),
            make_msg(n_time=5, n_ch=3),  # must NOT be reached
        ]
    )
    drained = list(unit._drain())
    assert len(drained) == 1
    assert drained[0].data.shape == (5, 0)
    # None also terminates the drain.
    unit.processor = _StubProcessor([None])
    assert list(unit._drain()) == []
    # The pre-init null template (no time axis at all) also terminates the
    # drain instead of being published.
    unit.processor = _StubProcessor([_null_template()])
    assert list(unit._drain()) == []
