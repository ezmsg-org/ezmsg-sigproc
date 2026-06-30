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


def _run(comps: dict, conns: tuple, output_stream, fn: Path) -> tuple[int, int, int]:
    log = MessageLogger(MessageLoggerSettings(output=fn))
    term = TerminateOnTimeout(TerminateOnTimeoutSettings(time=2.0))
    comps = {**comps, "LOG": log, "TERM": term}
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
    return len(msgs), total, n_ch


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
    n_msgs, total, n_ch = _run_resample_merge(
        glitch_at=-1,
        glitch_back=0.0,
        reset_after=3,
        use_output_reference=False,
        fn=get_test_fn(test_name),
    )
    assert n_msgs >= 50, f"Healthy graph should produce ~60 merged messages, got {n_msgs}."
    assert total >= 1500 and n_ch == 6


def test_resample_merge_seizes_when_recovery_disabled_system(test_name: str | None = None):
    """Documents the original bug: with recovery disabled, a backward reference jump stops output.

    Setting ``reference_reset_after_chunks=inf`` restores the pre-hardening behaviour, so a
    large sustained backward offset jump pushes the reference permanently below the
    resampler's high-water mark and the merged output ceases well before input is exhausted.
    """
    n_msgs, _, _ = _run_resample_merge(
        glitch_at=15,
        glitch_back=100.0,
        reset_after=float("inf"),
        use_output_reference=False,
        fn=get_test_fn(test_name),
    )
    assert n_msgs <= 20, f"Expected output to cease near the chunk-15 glitch, got {n_msgs}."


def test_resample_merge_recovers_with_output_reference_system(test_name: str | None = None):
    """Hardened resampler + OUTPUT_REFERENCE: the graph recovers from a reference reset.

    Reset recovery (default) keeps the resampler producing past the backward jump, and
    feeding MERGE.INPUT_SIGNAL_A from RESAMPLE.OUTPUT_REFERENCE keeps the two halves
    sample-aligned through the drops, so the merged 6-channel output continues.
    """
    n_msgs, total, n_ch = _run_resample_merge(
        glitch_at=15,
        glitch_back=1.0,
        reset_after=3,
        use_output_reference=True,
        fn=get_test_fn(test_name),
    )
    assert n_msgs > 25, f"Hardened graph should keep producing after the glitch, got {n_msgs}."
    assert n_ch == 6, f"Merged output should stay 4+2=6 channels, got {n_ch}."


def test_resampleconcat_recovers_system(test_name: str | None = None):
    """The fused ResampleConcat unit produces aligned 6-channel output and recovers from a reset."""
    n_msgs, total, n_ch = _run_composite(glitch_at=15, glitch_back=1.0, reset_after=3, fn=get_test_fn(test_name))
    assert n_msgs > 25, f"Composite should keep producing after the glitch, got {n_msgs}."
    assert n_ch == 6, f"Composite output should be 4+2=6 channels, got {n_ch}."
