"""Integration tests: combining two mismatched-rate streams onto one clock.

Models the real-world "two acquisition hubs, slightly different effective rates"
pipeline: HUB1 is the reference clock, HUB2 is resampled onto it, and the two are
concatenated along the channel axis (256 + 128 -> 384, here 4 + 2 -> 6).

Three wirings are exercised:

* ``raw``       -- HUB1 -> RESAMPLE.INPUT_REFERENCE, HUB1 -> MERGE.INPUT_SIGNAL_A,
                   RESAMPLE.OUTPUT_SIGNAL -> MERGE.INPUT_SIGNAL_B.  The original graph.
* ``ref``       -- as above but MERGE.INPUT_SIGNAL_A is fed from RESAMPLE.OUTPUT_REFERENCE
                   (the reference gathered on the exact output grid), so A and B stay
                   sample-aligned even when the resampler drops/holds back samples.
* ``composite`` -- a single :class:`ResampleConcat` unit doing both steps.

The ``glitch`` cases inject a sustained *backward* jump in HUB1's chunk offsets (a
non-monotonic reference clock, e.g. from a misbehaving simulator).
"""

import asyncio
import os
import time
import typing
from pathlib import Path

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTimeout, TerminateOnTimeoutSettings

from ezmsg.sigproc.merge import Merge, MergeSettings
from ezmsg.sigproc.resample import ResampleSettings, ResampleUnit
from ezmsg.sigproc.resampleconcat import ResampleConcat, ResampleConcatSettings
from tests.helpers.util import get_test_fn

CHUNK = 30
N_CHUNKS = 60
FS = 1000.0


class ArmedTerminateOnTimeout(TerminateOnTimeout):
    """TerminateOnTimeout arms its idle timer only on the first received message;
    a graph that never produces output therefore never terminates (the hubs stay
    alive by design, so there is no other exit path and CI hangs). Arm the timer
    at startup so an all-idle graph still tears down after ``time`` seconds and
    the test fails on its assertions instead of hanging."""

    async def initialize(self) -> None:
        self.STATE.last_msg_timestamp = time.time()


class HubSettings(ez.Settings):
    fs: float
    n_ch: int
    chunk: int = CHUNK
    n_chunks: int = N_CHUNKS
    glitch_at: int = -1
    """Chunk index at which to inject a sustained backward offset jump (-1 = never)."""
    glitch_back: float = 0.0
    """Seconds to jump the offset backward at (and after) ``glitch_at``."""
    key: str = "hub"
    dispatch_dt: float = 0.002


class Hub(ez.Unit):
    """Synthetic acquisition hub emitting [time, ch] AxisArrays on a LinearAxis."""

    SETTINGS = HubSettings
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        s = self.SETTINGS
        ch_ax = AxisArray.CoordinateAxis(
            data=np.array([f"{s.key}{i}" for i in range(s.n_ch)]), dims=["ch"], unit="label"
        )
        shift = 0.0
        for i in range(s.n_chunks):
            if i == s.glitch_at:
                shift = -s.glitch_back
            offset = i * s.chunk / s.fs + shift
            yield (
                self.OUTPUT_SIGNAL,
                AxisArray(
                    data=np.random.randn(s.chunk, s.n_ch).astype(np.float32),
                    dims=["time", "ch"],
                    axes={
                        "time": AxisArray.LinearAxis(gain=1 / s.fs, offset=offset, unit="s"),
                        "ch": ch_ax,
                    },
                    key=s.key,
                ),
            )
            await asyncio.sleep(s.dispatch_dt)
        # Stay alive; let TerminateOnTimeout decide when the graph is done so a seized
        # graph is distinguishable from one that simply ran out of input.
        while True:
            await asyncio.sleep(0.1)


def _hubs(glitch_at: int, glitch_back: float) -> dict:
    return {
        "HUB1": Hub(HubSettings(fs=FS, n_ch=4, glitch_at=glitch_at, glitch_back=glitch_back, key="h1")),
        # HUB2 runs 0.3% slow -- the realistic effective-rate mismatch.
        "HUB2": Hub(HubSettings(fs=FS * 0.997, n_ch=2, key="h2")),
    }


def _run(comps: dict, conns: tuple, output_stream, fn: Path) -> tuple[int, int, int, float]:
    log = MessageLogger(MessageLoggerSettings(output=fn))
    # Idle-gap terminator: end the graph once no message has reached the logger
    # for `time` seconds. `time` must comfortably exceed the worst-case scheduling
    # stall on a loaded CI runner -- if a transient stall opens an output gap
    # longer than `time` mid-stream, the graph is torn down early and the output
    # is truncated. 2.0 s was too tight on Windows CI; 4.0 s gives headroom.
    # (Armed at startup, so it also covers the zero-output case.)
    term = ArmedTerminateOnTimeout(TerminateOnTimeoutSettings(time=4.0))
    # Absolute-deadline watchdog: no input is connected, so its armed timer is
    # never refreshed and it unconditionally ends the graph after 30 s. A full
    # run terminates via `term` in well under 10 s, so this only fires if the
    # graph is wired in a way that defeats the idle terminator.
    watchdog = ArmedTerminateOnTimeout(TerminateOnTimeoutSettings(time=30.0))
    comps = {**comps, "LOG": log, "TERM": term, "WATCHDOG": watchdog}
    conns = (
        *conns,
        (output_stream, log.INPUT_MESSAGE),
        (log.OUTPUT_MESSAGE, term.INPUT),
    )
    ez.run(components=comps, connections=conns)
    msgs: list[AxisArray] = list(message_log(fn))
    os.remove(fn)
    total = sum(m.data.shape[0] for m in msgs)
    n_ch = msgs[0].data.shape[1] if msgs else 0
    # `last_t` is the stream-time of the final emitted sample -- i.e. how far
    # through the signal the output reached. Unlike message count (a coalescing
    # artifact) or total samples (which the reset discontinuity drops by a
    # timing-dependent amount), this tracks whether output progressed past the
    # glitch to the end, and is insensitive to mid-stream drops. It only falls if
    # the tail is truncated, which the generous idle timeout above guards against.
    if msgs:
        ax = msgs[-1].axes["time"]
        if hasattr(ax, "data") and len(ax.data):
            last_t = float(ax.data[-1])
        else:
            last_t = float(ax.offset + (msgs[-1].data.shape[0] - 1) * ax.gain)
    else:
        last_t = float("-inf")
    return len(msgs), total, n_ch, last_t


def _run_resample_merge(glitch_at, glitch_back, reset_after, use_output_reference, fn):
    comps = {
        **_hubs(glitch_at, glitch_back),
        "RESAMPLE": ResampleUnit(
            ResampleSettings(
                axis="time",
                resample_rate=None,
                buffer_duration=2.0,
                output_reference=use_output_reference,
                reference_reset_after_chunks=reset_after,
            )
        ),
        "MERGE": Merge(MergeSettings(axis="ch", align_axis="time", buffer_dur=5.0)),
    }
    a_src = comps["RESAMPLE"].OUTPUT_REFERENCE if use_output_reference else comps["HUB1"].OUTPUT_SIGNAL
    conns = (
        (comps["HUB1"].OUTPUT_SIGNAL, comps["RESAMPLE"].INPUT_REFERENCE),
        (comps["HUB2"].OUTPUT_SIGNAL, comps["RESAMPLE"].INPUT_SIGNAL),
        (a_src, comps["MERGE"].INPUT_SIGNAL_A),
        (comps["RESAMPLE"].OUTPUT_SIGNAL, comps["MERGE"].INPUT_SIGNAL_B),
    )
    return _run(comps, conns, comps["MERGE"].OUTPUT_SIGNAL, fn)


def _run_composite(glitch_at, glitch_back, reset_after, fn):
    comps = {
        **_hubs(glitch_at, glitch_back),
        "RC": ResampleConcat(
            ResampleConcatSettings(
                axis="time",
                concat_axis="ch",
                buffer_duration=2.0,
                reference_reset_after_chunks=reset_after,
            )
        ),
    }
    conns = (
        (comps["HUB1"].OUTPUT_SIGNAL, comps["RC"].INPUT_REFERENCE),
        (comps["HUB2"].OUTPUT_SIGNAL, comps["RC"].INPUT_SIGNAL),
    )
    return _run(comps, conns, comps["RC"].OUTPUT_SIGNAL, fn)


def test_resample_merge_healthy_system(test_name: str | None = None):
    """Monotonic reference clock: the graph streams to completion, 6 channels out."""
    _, _, n_ch, last_t = _run_resample_merge(
        glitch_at=-1,
        glitch_back=0.0,
        reset_after=3,
        use_output_reference=False,
        fn=get_test_fn(test_name),
    )
    # Assert output reached the end of the stream (last sample time), not a
    # message or sample count: under backpressure the same data arrives coalesced
    # into a variable number of messages (108 locally vs 29 on a loaded Windows
    # runner). A full run reaches ~1.799 s; 1.5 leaves margin while staying far
    # above the seized run's ~0.449 s (see the seize test).
    assert last_t > 1.5, f"Healthy graph should stream to completion (last sample ~1.8 s), got {last_t:.3f} s."
    assert n_ch == 6, f"Merged output should be 4+2=6 channels, got {n_ch}."


def test_resample_merge_seizes_when_recovery_disabled_system(test_name: str | None = None):
    """Documents the original bug: with recovery disabled, a backward reference jump stops output.

    Setting ``reference_reset_after_chunks=inf`` restores the pre-hardening behaviour, so a
    large sustained backward offset jump pushes the reference permanently below the
    resampler's high-water mark and the merged output ceases well before input is exhausted.
    """
    n_msgs, _, _, last_t = _run_resample_merge(
        glitch_at=15,
        glitch_back=100.0,
        reset_after=float("inf"),
        use_output_reference=False,
        fn=get_test_fn(test_name),
    )
    # Output ceases at the chunk-15 glitch, so the last emitted sample sits near
    # 15*30/1000 = 0.45 s; a recovered/healthy run reaches >= 0.799 s. Assert on
    # last sample time (see _run) -- it isolates "output stopped early" from the
    # variable amount of data dropped. n_msgs > 0 is a liveness sanity.
    assert n_msgs > 0 and last_t < 0.6, f"Expected output to cease near the chunk-15 glitch, got {last_t:.3f} s."


def test_resample_merge_recovers_with_output_reference_system(test_name: str | None = None):
    """Hardened resampler + OUTPUT_REFERENCE: the graph recovers from a reference reset.

    Reset recovery (default) keeps the resampler producing past the backward jump, and
    feeding MERGE.INPUT_SIGNAL_A from RESAMPLE.OUTPUT_REFERENCE keeps the two halves
    sample-aligned through the drops, so the merged 6-channel output continues.
    """
    _, _, n_ch, last_t = _run_resample_merge(
        glitch_at=15,
        glitch_back=1.0,
        reset_after=3,
        use_output_reference=True,
        fn=get_test_fn(test_name),
    )
    # Recovery re-anchors and keeps producing past the glitch to the (re-anchored)
    # end at ~0.799 s. Assert on last sample time, not sample count: the reset
    # drops a timing-dependent amount of data (observed 1710 samples locally but
    # 960 on a loaded runner), while the *time reached* stays ~0.799 s regardless.
    # 0.6 sits between the seized ~0.449 s and the recovered ~0.799 s.
    assert last_t > 0.6, f"Hardened graph should keep producing past the glitch (~0.8 s), got {last_t:.3f} s."
    assert n_ch == 6, f"Merged output should stay 4+2=6 channels, got {n_ch}."


def test_resampleconcat_recovers_system(test_name: str | None = None):
    """The fused ResampleConcat unit produces aligned 6-channel output and recovers from a reset."""
    _, _, n_ch, last_t = _run_composite(glitch_at=15, glitch_back=1.0, reset_after=3, fn=get_test_fn(test_name))
    # Last sample time, same rationale as the merge recovery test above.
    assert last_t > 0.6, f"Composite should keep producing past the glitch (~0.8 s), got {last_t:.3f} s."
    assert n_ch == 6, f"Composite output should be 4+2=6 channels, got {n_ch}."
