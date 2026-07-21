"""Integration tests for the ResampleUnit's event-driven publisher.

The publisher waits on an asyncio.Event set by both input handlers instead of
polling. Two wake paths need proving through a live graph:

* reference-driven mode: output flows on pushes alone (pure event wait), and
  the graph terminates once sources go quiet;
* prescribed-rate mode with a finite ``max_chunk_delay``: the wall-clock
  extrapolation in ``ResampleProcessor.__next__`` must still fire when input
  stops, which requires the timed wake.
"""

import os
import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTimeout, TerminateOnTimeoutSettings

from ezmsg.sigproc.resample import ResampleSettings, ResampleUnit
from tests.helpers.util import get_test_fn


def _mk(fs: float, offset: float, n: int, n_ch: int, key: str) -> AxisArray:
    """Build a [time, ch] message where ``data[:, j] == time + 1000*j``."""
    t = offset + np.arange(n) / fs
    data = t[:, None] + np.arange(n_ch)[None, :] * 1000.0
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": AxisArray.LinearAxis(gain=1 / fs, offset=offset, unit="s")},
        key=key,
    )


class DualSourceSettings(ez.Settings):
    n_msgs: int = 20
    chunk: int = 30
    fs_ref: float = 100.0
    fs_sig: float = 99.7
    emit_reference: bool = True


class DualSource(ez.Unit):
    """Emit interleaved reference/signal chunks, then go quiet."""

    SETTINGS = DualSourceSettings

    OUTPUT_REFERENCE = ez.OutputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.publisher(OUTPUT_REFERENCE)
    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        s = self.SETTINGS
        for i in range(s.n_msgs):
            if s.emit_reference:
                yield self.OUTPUT_REFERENCE, _mk(s.fs_ref, i * s.chunk / s.fs_ref, s.chunk, 1, "ref")
            yield self.OUTPUT_SIGNAL, _mk(s.fs_sig, i * s.chunk / s.fs_sig, s.chunk, 2, "sig")


def _run_graph(resample_settings: ResampleSettings, source_settings: DualSourceSettings, term_time: float) -> list:
    test_filename = get_test_fn(None)
    comps = {
        "SRC": DualSource(source_settings),
        "RESAMPLE": ResampleUnit(resample_settings),
        "LOG": MessageLogger(MessageLoggerSettings(output=test_filename)),
        "TERM": TerminateOnTimeout(TerminateOnTimeoutSettings(time=term_time)),
    }
    conns = (
        (comps["SRC"].OUTPUT_REFERENCE, comps["RESAMPLE"].INPUT_REFERENCE),
        (comps["SRC"].OUTPUT_SIGNAL, comps["RESAMPLE"].INPUT_SIGNAL),
        (comps["RESAMPLE"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT),
    )
    ez.run(components=comps, connections=conns)
    messages = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)
    return messages


def test_resample_system_reference_driven():
    """Pushes alone must wake the publisher; no polling loop exists to find output."""
    messages = _run_graph(
        ResampleSettings(axis="time", resample_rate=None, buffer_duration=4.0),
        DualSourceSettings(),
        term_time=1.0,
    )
    assert messages, "ResampleUnit published no output in reference-driven mode."
    cat = AxisArray.concatenate(*messages, dim="time")
    t = cat.axes["time"].value(np.arange(cat.data.shape[0]))
    assert np.all(np.diff(t) > 0), "Output time axis is not monotonic."
    # Signal was built with data[:, 0] == time, so resampling onto the
    # reference grid must reproduce the grid itself.
    assert np.max(np.abs(cat.data[:, 0] - t)) < 1e-6


def test_resample_system_prescribed_rate():
    """Prescribed-rate mode publishes on signal pushes alone (no reference input).

    Note: this does NOT assert the ``max_chunk_delay`` wall-clock extrapolation
    past the end of input. That path is currently unreachable at the processor
    level regardless of how the unit polls or wakes: after a drain the source
    buffer retains only ~2 samples, and ``ResampleProcessor.__next__`` returns
    early on ``src.available() < 3`` before evaluating ``b_project``. The
    unit's timed wake preserves the intended trigger; if the processor guard is
    reworked, extend this test to assert output beyond the input end.
    """
    n_msgs, chunk, fs = 5, 30, 100.0
    max_chunk_delay = 0.2
    messages = _run_graph(
        ResampleSettings(
            axis="time",
            resample_rate=90.0,
            buffer_duration=4.0,
            max_chunk_delay=max_chunk_delay,
        ),
        DualSourceSettings(n_msgs=n_msgs, chunk=chunk, fs_sig=fs, emit_reference=False),
        term_time=3 * max_chunk_delay,
    )
    assert messages, "ResampleUnit published no output in prescribed-rate mode."
    cat = AxisArray.concatenate(*messages, dim="time")
    t = cat.axes["time"].value(np.arange(cat.data.shape[0]))
    assert np.all(np.diff(t) > 0), "Output time axis is not monotonic."
    assert np.max(np.abs(cat.data[:, 0] - t)) < 1e-6
