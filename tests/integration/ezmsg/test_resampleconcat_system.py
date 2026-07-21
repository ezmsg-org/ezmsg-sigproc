"""Integration test for the ResampleConcat unit's event-driven publishing.

The unit publishes from its two subscriber handlers (no polling loop), so this
test verifies that outputs actually flow through a live pub/sub graph when
reference and signal chunks arrive interleaved, and that the graph terminates
once the sources go quiet (i.e., nothing depends on a background publisher).
"""

import os
import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTimeout, TerminateOnTimeoutSettings

from ezmsg.sigproc.resampleconcat import ResampleConcat, ResampleConcatSettings
from tests.helpers.util import get_test_fn


def _mk(fs: float, offset: float, n: int, n_ch: int, key: str) -> AxisArray:
    """Build a [time, ch] message where ``data[:, j] == time + 1000*j``."""
    t = offset + np.arange(n) / fs
    data = t[:, None] + np.arange(n_ch)[None, :] * 1000.0
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.LinearAxis(gain=1 / fs, offset=offset, unit="s"),
            "ch": AxisArray.CoordinateAxis(
                data=np.array([f"{key}{i}" for i in range(n_ch)]), dims=["ch"], unit="label"
            ),
        },
        key=key,
    )


class DualSourceSettings(ez.Settings):
    n_msgs: int = 20
    chunk: int = 30
    fs_ref: float = 100.0
    fs_sig: float = 99.7


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
            yield self.OUTPUT_REFERENCE, _mk(s.fs_ref, i * s.chunk / s.fs_ref, s.chunk, 4, "h1")
            yield self.OUTPUT_SIGNAL, _mk(s.fs_sig, i * s.chunk / s.fs_sig, s.chunk, 2, "h2")


def test_resampleconcat_system(test_name: str | None = None):
    test_filename = get_test_fn(test_name)

    comps = {
        "SRC": DualSource(DualSourceSettings()),
        "RSC": ResampleConcat(
            ResampleConcatSettings(
                axis="time",
                concat_axis="ch",
                buffer_duration=4.0,
                label_a="_h1",
                label_b="_h2",
            )
        ),
        "LOG": MessageLogger(MessageLoggerSettings(output=test_filename)),
        "TERM": TerminateOnTimeout(TerminateOnTimeoutSettings(time=1.0)),
    }
    conns = (
        (comps["SRC"].OUTPUT_REFERENCE, comps["RSC"].INPUT_REFERENCE),
        (comps["SRC"].OUTPUT_SIGNAL, comps["RSC"].INPUT_SIGNAL),
        (comps["RSC"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT),
    )
    ez.run(components=comps, connections=conns)

    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    assert messages, "ResampleConcat unit published no output."
    cat = AxisArray.concatenate(*messages, dim="time")
    assert cat.data.shape[1] == 6, "Expected 4 (reference) + 2 (signal) = 6 channels."

    t = cat.axes["time"].value(np.arange(cat.data.shape[0]))
    # Output time axis must be strictly monotonic even though both handlers publish.
    assert np.all(np.diff(t) > 0), "Output time axis is not monotonic."
    # Side A is the reference gathered on the output grid; side B is resampled onto it.
    assert np.max(np.abs(cat.data[:, 0] - t)) < 1e-9
    assert np.max(np.abs(cat.data[:, 4] - t)) < 1e-6
