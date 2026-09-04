"""How often does the base-class state hash actually reset a live graph?

Most processors no longer implement ``_hash_message``: the base class decides,
folding in the message key, the dims, the length of every dimension except the
chunk dimension, the coordinate values on those dimensions, and the gain and
offset of any linear axis. This runs a real ``ez.run`` graph over simulated
256-channel data shaped like the intracranial feature pipeline and counts how
often each node rebuilds its state.

What to look for: a stream whose configuration never changes should reset each
node exactly *once*, no matter how much the chunk size jitters. Anything more
means the hash is folding in something per-message, and every extra reset is a
filter redesigned or a buffer reallocated mid-stream.

The source can rebuild its axis objects per message or reuse them, and the
graph can be split across a process boundary; neither should change the reset
counts, only the cost of arriving at them.

Run from the repository root::

    uv run python benchmarks/benchmark_memo_hit_rate.py
    uv run python benchmarks/benchmark_memo_hit_rate.py --n-messages 500 --n-ch 512
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import tempfile

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.sigproc.affinetransform import AffineTransform, AffineTransformSettings, AffineTransformTransformer
from ezmsg.sigproc.flatten import Flatten, FlattenSettings, FlattenTransformer

# Count state rebuilds per processor class, without touching the library.
# Patched on each concrete class rather than on Stateful: every processor
# overrides _reset_state, so a base-class patch would never be reached.
_RESETS: dict[str, int] = {}


def _count_resets(cls: type) -> None:
    original = cls._reset_state

    def counting(self, *args, **kwargs):
        _RESETS[type(self).__name__] = _RESETS.get(type(self).__name__, 0) + 1
        return original(self, *args, **kwargs)

    cls._reset_state = counting


for _cls in (AffineTransformTransformer, FlattenTransformer):
    _count_resets(_cls)

CHANNELMAP_DTYPE = np.dtype(
    [
        ("label", "U16"),
        ("x", "<f8"),
        ("y", "<f8"),
        ("size", "<f8"),
        ("array", "<i4"),
        ("bank", "U2"),
        ("elec", "<i4"),
        ("headstage", "<i4"),
    ]
)


def channelmap(n_ch: int) -> CoordinateAxis:
    d = np.zeros(n_ch, dtype=CHANNELMAP_DTYPE)
    d["label"] = [f"elec{i:04d}" for i in range(n_ch)]
    d["x"] = np.arange(n_ch, dtype=float)
    d["array"] = np.arange(n_ch) // 64
    d["bank"] = "A"
    d["elec"] = np.arange(n_ch, dtype=np.int32)
    return CoordinateAxis(data=d, dims=["ch"])


def feature_axis() -> CoordinateAxis:
    return CoordinateAxis(data=np.array(["spk", "sbp"]), dims=["feature"])


class SourceSettings(ez.Settings):
    n_ch: int = 256
    n_times: int = 30
    n_messages: int = 200
    fs: float = 30000.0
    rebuild_axes: bool = False
    """Build NEW axis objects per message, as a source that re-reads its channel
    metadata each chunk would. False reuses them, which is what a source that
    builds its channel map once at startup does."""


class SourceState(ez.State):
    ch_axis: CoordinateAxis | None = None
    feat_axis: CoordinateAxis | None = None


class SimSource(ez.Unit):
    """Emit AxisArrays shaped like one hub's broadband stream."""

    SETTINGS = SourceSettings
    STATE = SourceState
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self.STATE.ch_axis = channelmap(self.SETTINGS.n_ch)
        self.STATE.feat_axis = feature_axis()

    @ez.publisher(OUTPUT_SIGNAL)
    async def pub(self):
        n_ch, n_t = self.SETTINGS.n_ch, self.SETTINGS.n_times
        for i in range(self.SETTINGS.n_messages):
            if self.SETTINGS.rebuild_axes:
                ch, feat = channelmap(n_ch), feature_axis()
            else:
                ch, feat = self.STATE.ch_axis, self.STATE.feat_axis
            yield (
                self.OUTPUT_SIGNAL,
                AxisArray(
                    np.zeros((n_t, n_ch, 2), dtype=np.float32),
                    dims=["time", "ch", "feature"],
                    axes={
                        "time": AxisArray.TimeAxis(fs=self.SETTINGS.fs, offset=i * n_t / self.SETTINGS.fs),
                        "ch": ch,
                        "feature": feat,
                    },
                    key="sim",
                ),
            )
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.5)
        raise ez.NormalTermination


class ReporterSettings(ez.Settings):
    out_dir: str = ""
    tag: str = ""


class Reporter(ez.Unit):
    """Dump this process's fingerprint counters on shutdown."""

    SETTINGS = ReporterSettings
    INPUT_SIGNAL = ez.InputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL)
    async def sink(self, _: AxisArray) -> None:
        pass

    async def shutdown(self) -> None:
        path = pathlib.Path(self.SETTINGS.out_dir) / f"{self.SETTINGS.tag}-{os.getpid()}.json"
        path.write_text(json.dumps({"pid": os.getpid(), "tag": self.SETTINGS.tag, "stats": dict(_RESETS)}))


class DownstreamSettings(ez.Settings):
    flatten: FlattenSettings
    reporter: ReporterSettings


class Downstream(ez.Collection):
    """Flatten + its reporter, so both land in the same worker process."""

    SETTINGS = DownstreamSettings
    INPUT_SIGNAL = ez.InputStream(AxisArray)

    FLATTEN = Flatten()
    REPORT = Reporter()

    def configure(self) -> None:
        self.FLATTEN.apply_settings(self.SETTINGS.flatten)
        self.REPORT.apply_settings(self.SETTINGS.reporter)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.FLATTEN.INPUT_SIGNAL),
            (self.FLATTEN.OUTPUT_SIGNAL, self.REPORT.INPUT_SIGNAL),
        )


def build_and_run(n_ch, n_times, n_messages, rebuild_axes, out_dir, split) -> None:
    """SRC -> LRR(non-square) -> LRR(square) -> {reporter, Flatten -> reporter}."""
    _RESETS.clear()  # counters are process-global; isolate the runs
    n_out = n_ch // 2
    weights = np.zeros((n_ch, n_out))
    weights[np.arange(n_out) * 2, np.arange(n_out)] = 1.0  # keep every other channel

    downstream = Downstream(
        DownstreamSettings(
            flatten=FlattenSettings(preserve_axis="time", flatten_axes=("ch", "feature"), output_axis="ch"),
            reporter=ReporterSettings(out_dir=out_dir, tag="downstream"),
        )
    )
    comps = {
        "SRC": SimSource(SourceSettings(n_ch=n_ch, n_times=n_times, n_messages=n_messages, rebuild_axes=rebuild_axes)),
        "LRR1": AffineTransform(AffineTransformSettings(weights=weights, axis="ch")),
        "LRR2": AffineTransform(AffineTransformSettings(weights=np.eye(n_out), axis="ch")),
        "UP_STATS": Reporter(ReporterSettings(out_dir=out_dir, tag="upstream")),
        "DOWN": downstream,
    }
    conns = (
        (comps["SRC"].OUTPUT_SIGNAL, comps["LRR1"].INPUT_SIGNAL),
        (comps["LRR1"].OUTPUT_SIGNAL, comps["LRR2"].INPUT_SIGNAL),
        (comps["LRR2"].OUTPUT_SIGNAL, comps["UP_STATS"].INPUT_SIGNAL),
        (comps["LRR2"].OUTPUT_SIGNAL, downstream.INPUT_SIGNAL),
    )
    ez.run(
        components=comps,
        connections=conns,
        process_components=(downstream,) if split else (),
        force_single_process=not split,
    )


# How many instances of each class the graph below contains; a correctly
# behaving stream rebuilds each exactly once, on its first message.
INSTANCES = {"AffineTransformTransformer": 2, "FlattenTransformer": 1}


def report(out_dir: str, label: str) -> None:
    print(f"\n=== {label} ===")
    recs = [json.loads(f.read_text()) for f in sorted(pathlib.Path(out_dir).glob("*.json"))]
    if not recs:
        print("  (no counters written)")
        return
    print(f"  pids reporting: {sorted({r['pid'] for r in recs})}")
    for rec in sorted(recs, key=lambda r: r["tag"], reverse=True):
        where = "source process" if rec["tag"] == "upstream" else "downstream of the boundary"
        if not rec["stats"]:
            print(f"  {where} (pid {rec['pid']}): no stateful nodes here")
            continue
        print(f"  {where} (pid {rec['pid']}):")
        for name, n_resets in sorted(rec["stats"].items()):
            # Counters are keyed by class, so a class used twice in the graph
            # should show two rebuilds -- one each, on its first message.
            expected = INSTANCES.get(name, 1)
            flag = (
                "" if n_resets <= expected else f"   <-- expected {expected}; the hash is seeing something per-message"
            )
            print(f"    {name:<40} state rebuilds: {n_resets} (one per instance, {expected} expected){flag}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-ch", type=int, default=256)
    p.add_argument("--n-times", type=int, default=30)
    p.add_argument("--n-messages", type=int, default=200)
    args = p.parse_args()

    print(f"{args.n_messages} messages, {args.n_ch} ch x {args.n_times} samples, ChannelMap ch axis")
    for label, rebuild, split in (
        ("single process, source reuses its axis objects", False, False),
        ("single process, source rebuilds axes per message", True, False),
        ("split across processes, source reuses its axis objects", False, True),
    ):
        with tempfile.TemporaryDirectory() as d:
            build_and_run(args.n_ch, args.n_times, args.n_messages, rebuild, d, split)
            report(d, label)


if __name__ == "__main__":
    main()
