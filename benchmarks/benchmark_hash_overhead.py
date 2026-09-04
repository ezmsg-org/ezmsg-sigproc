"""What the safer default state hash costs, and what fingerprint caching gives back.

Every stateful processor decides, once per message, whether its cached state is
still valid. That decision used to be nearly free and frequently wrong: the base
class returned a constant, so a processor without its own ``_hash_message``
never reset, and a source that renamed its channels under a fixed key and
channel count kept being filtered through the previous channels' history.

The default now folds in the message key, the dims, the length of every
dimension except the chunk dimension, the *values* on the coordinate axes and
the gain and offset of any linear axis among them. That is strictly more work
per message. This script measures how much more, and how much of it
``CoordinateAxis.fingerprint`` hands back by computing the expensive part -- a
crc32 over the axis data -- once per axis object rather than once per consumer.

Three arms, so the two effects can be told apart:

``before``
    The processors as they were: constant-hash default, hand-written overrides
    on the few processors that had them. Cheap, and wrong in the ways above.
``after``
    What ships now. Correct everywhere, and the coordinate checksum is computed
    on first access and cached on the axis, so a fan-out of N consumers pays for
    it once.
``naive``
    The same correctness without the cache -- ``fingerprint`` recomputes on
    every access. This is what the fix would have cost implemented the obvious
    way; the gap between it and ``after`` is what the caching is worth.

Fingerprints are cached *on the axis object*, so a benchmark that reuses one
message list across timed runs measures a warm cache and reports the checksum as
free. Every timed run below is preceded by an untimed pass that strips those
caches, which is what ``--arm naive`` and the cold/warm split are there to make
visible.

``after`` and ``naive`` run against the working tree. ``before`` needs the
pre-sweep sources of *both* packages on ``PYTHONPATH``, which git worktrees
supply without disturbing either checkout::

    git worktree add --detach /tmp/sigproc-pre <commit before the hashing sweep>
    cp src/ezmsg/sigproc/__version__.py /tmp/sigproc-pre/src/ezmsg/sigproc/
    git -C ../ezmsg-baseproc worktree add --detach /tmp/baseproc-pre <commit before PR #13>
    cp ../ezmsg-baseproc/src/ezmsg/baseproc/__version__.py /tmp/baseproc-pre/src/ezmsg/baseproc/

(``__version__.py`` is generated at build time, so a fresh worktree lacks it.)
Then, from the repository root::

    uv run python benchmarks/benchmark_hash_overhead.py --arm after
    uv run python benchmarks/benchmark_hash_overhead.py --arm naive
    PYTHONPATH=/tmp/sigproc-pre/src:/tmp/baseproc-pre/src \\
        uv run python benchmarks/benchmark_hash_overhead.py --arm before

Pass ``--json`` to emit a machine-readable record for diffing arms.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import inspect
import json
import time
import timeit
import typing
import warnings

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

warnings.filterwarnings("ignore")

HAS_CHUNK_DIM = "chunk_dim" in AxisArray.__dataclass_fields__
DECLARE_CHUNK_DIM = HAS_CHUNK_DIM
"""Whether the simulated source declares its chunk dimension.

The ``before`` arm must not: nothing set the field then, and the pre-sweep
``Spectrum`` does not clear it when it consumes ``time``, so a declared source
trips the validation that ships with it."""

# ezmsg-blackrock's ChannelMap ``ch`` axis: the "full metadata" case, 108 B per
# channel. Kept byte-identical to benchmark_axis_fingerprint.py so the numbers
# there line up with the ones here.
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


def channelmap_data(n_ch: int) -> np.ndarray:
    data = np.zeros(n_ch, dtype=CHANNELMAP_DTYPE)
    data["label"] = [f"elec{i:04d}" for i in range(n_ch)]
    data["x"] = np.arange(n_ch, dtype=np.float64)
    data["y"] = np.arange(n_ch, dtype=np.float64)
    data["size"] = 1.0
    data["array"] = np.arange(n_ch) // 128
    data["bank"] = np.array([("A", "B", "C", "D")[i // 64 % 4] for i in range(n_ch)])
    data["elec"] = np.arange(n_ch, dtype=np.int32) + 1
    data["headstage"] = np.arange(n_ch) // 32
    return data


def label_data(n_ch: int) -> np.ndarray:
    """The plain label axis most sources emit."""
    return np.array([f"ch{i:04d}" for i in range(n_ch)])


def make_message(data: np.ndarray, ch_data: np.ndarray, fs: float, key: str = "dev") -> AxisArray:
    """A fresh message, with a *fresh* coordinate axis: a live source builds new
    axis objects per message, so the fingerprint cache starts cold."""
    kwargs = {"chunk_dim": "time"} if DECLARE_CHUNK_DIM else {}
    return AxisArray(
        data,
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs), "ch": CoordinateAxis(data=ch_data, dims=["ch"])},
        key=key,
        **kwargs,
    )


def chill(message: typing.Any) -> typing.Any:
    """Drop any cached fingerprints, returning the message to its as-received state."""
    axes = getattr(message, "axes", None)
    if axes:
        for axis in axes.values():
            axis.__dict__.pop("_fingerprint", None)
    return message


def uncache_fingerprint() -> bool:
    """Make ``fingerprint`` recompute on every access (the ``naive`` arm)."""
    if not hasattr(CoordinateAxis, "_compute_fingerprint"):
        return False
    CoordinateAxis.fingerprint = property(lambda self: self._compute_fingerprint())
    return True


# --------------------------------------------------------------------------- #
# Section 1: what one _hash_message call costs, per processor
# --------------------------------------------------------------------------- #


def discover_processors() -> dict[str, type]:
    import importlib
    import pkgutil

    from ezmsg.baseproc.stateful import Stateful

    import ezmsg.sigproc

    mods = []
    for m in pkgutil.walk_packages(ezmsg.sigproc.__path__, "ezmsg.sigproc."):
        try:
            mods.append(importlib.import_module(m.name))
        except Exception:
            pass
    names = {m.__name__ for m in mods}
    found: dict[str, type] = {}
    for mod in mods:
        for obj in vars(mod).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Stateful)
                and obj is not Stateful
                and obj.__module__ in names
                and not inspect.isabstract(obj)
            ):
                found[f"{obj.__module__.replace('ezmsg.sigproc.', '')}.{obj.__qualname__}"] = obj
    return found


def _settings_type(cls: type) -> type | None:
    for klass in cls.__mro__:
        for base in getattr(klass, "__orig_bases__", ()):
            for arg in typing.get_args(base):
                if dataclasses.is_dataclass(arg):
                    return arg
    return None


def _bare_instance(cls: type) -> typing.Any:
    """``_hash_message`` reads the message and the settings, never the state, so
    an uninitialised instance carrying default settings is enough to time it."""
    inst = object.__new__(cls)
    settings_t = _settings_type(cls)
    inst.settings = settings_t() if settings_t is not None else None
    return inst


TARGET_S = 0.02
"""Wall time to aim each timing loop at. ``timeit.autorange`` targets 0.2 s,
which is 40x more than sub-microsecond calls need to resolve and turns this
script into a ten-minute run across three arms."""


def _time(fn: typing.Callable[[], typing.Any], repeats: int) -> tuple[float, int]:
    timer = timeit.Timer(fn)
    probe = timer.timeit(64) / 64
    n = max(64, min(int(TARGET_S / max(probe, 1e-9)), 200_000))
    return min(timer.repeat(repeats, n)) / n, n


def hash_cost_per_processor(msg: AxisArray, repeats: int) -> dict[str, dict[str, float]]:
    """Cold and warm cost of one ``_hash_message`` call, per processor.

    *Warm* is what every consumer after the first pays: the fingerprint is
    already on the axis. *Cold* includes computing it, and is what the first
    consumer to touch a freshly built axis pays. The cache-clearing itself is
    timed separately and subtracted, so cold is comparable to warm.
    """
    baseline, _ = _time(lambda: chill(msg), repeats)
    out: dict[str, dict[str, float]] = {}
    for qualname, cls in sorted(discover_processors().items()):
        try:
            inst = _bare_instance(cls)
            inst._hash_message(msg)
        except Exception:
            continue
        warm, _ = _time(lambda: inst._hash_message(msg), repeats)
        cold, _ = _time(lambda: inst._hash_message(chill(msg)), repeats)
        out[qualname] = {"warm_us": warm * 1e6, "cold_us": max(cold - baseline, 0.0) * 1e6}
    chill(msg)
    return out


# --------------------------------------------------------------------------- #
# Section 2: a whole chain, end to end
# --------------------------------------------------------------------------- #


def build_chain(fs: float) -> list:
    """An intracranial-features-shaped chain: rereference, band-limit, window,
    spectrum, band-average, flatten to a feature vector."""
    from ezmsg.sigproc.affinetransform import CommonRereferenceSettings, CommonRereferenceTransformer
    from ezmsg.sigproc.aggregate import AggregateSettings, AggregateTransformer, AggregationFunction
    from ezmsg.sigproc.butterworthfilter import ButterworthFilterSettings, ButterworthFilterTransformer
    from ezmsg.sigproc.flatten import FlattenSettings, FlattenTransformer
    from ezmsg.sigproc.spectrum import SpectrumSettings, SpectrumTransformer
    from ezmsg.sigproc.window import WindowSettings, WindowTransformer

    return [
        CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch")),
        ButterworthFilterTransformer(ButterworthFilterSettings(axis="time", order=4, cuton=70.0, cutoff=150.0)),
        WindowTransformer(WindowSettings(axis="time", newaxis="win", window_dur=0.1, window_shift=0.02)),
        SpectrumTransformer(SpectrumSettings(axis="time")),
        AggregateTransformer(AggregateSettings(axis="freq", operation=AggregationFunction.MEAN)),
        FlattenTransformer(FlattenSettings(preserve_axis="win", sample_axis="win", flatten_axes=("ch",))),
    ]


CHAIN_LABELS = ("rereference", "butterworth", "window", "spectrum", "aggregate", "flatten")


def _empty(out: typing.Any) -> bool:
    return out is None or (hasattr(out, "data") and out.data.size == 0)


def run_chain(chain: list, messages: list[AxisArray]) -> tuple[float, int]:
    n_out = 0
    for msg in messages:
        chill(msg)
    gc.collect()
    t0 = time.perf_counter()
    for msg in messages:
        out: typing.Any = msg
        for stage in chain:
            out = stage(out)
            if _empty(out):
                break
        else:
            n_out += 1
    return time.perf_counter() - t0, n_out


def freeze_hashes(chain: list) -> int:
    """Neutralise ``_hash_message`` on every processor a stage owns, so the same
    chain measures pure compute. Returns how many were reached."""
    from ezmsg.baseproc.stateful import Stateful

    seen: set[int] = set()
    frozen = 0

    def visit(obj) -> None:
        nonlocal frozen
        if id(obj) in seen or isinstance(obj, (np.ndarray, AxisArray, type)):
            return
        seen.add(id(obj))
        if isinstance(obj, Stateful):
            obj._hash_message = lambda message: 0
            frozen += 1
        for value in list(getattr(obj, "__dict__", {}).values()):
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif hasattr(value, "__dict__"):
                visit(value)

    for stage in chain:
        visit(stage)
    return frozen


def chain_throughput(messages: list[AxisArray], fs: float, n_rounds: int) -> dict[str, typing.Any]:
    """Time the chain with hashing live and with it neutralised.

    The difference is what hashing costs end to end -- but read it with the
    per-stage numbers next to it. Hashing is a couple of microseconds against a
    chain that spends hundreds on filtering and FFTs, so the difference sits
    well inside run-to-run drift and can come out negative. The order of the two
    runs alternates so that drift at least does not accumulate in one direction;
    ``per_stage_hash_cost`` is the measurement that actually resolves it.
    """
    live_times, frozen_times = [], []
    n_out = 0
    n_frozen = 0

    def live_run() -> None:
        nonlocal n_out
        # A fresh chain each round, warmed on a few messages so the timed run
        # measures steady state rather than every stage's first-message setup.
        chain = build_chain(fs)
        run_chain(chain, messages[:8])
        dt, n_out = run_chain(chain, messages)
        live_times.append(dt)

    def frozen_run() -> None:
        nonlocal n_frozen
        chain = build_chain(fs)
        run_chain(chain, messages[:8])
        n_frozen = freeze_hashes(chain)
        dt, _ = run_chain(chain, messages)
        frozen_times.append(dt)

    for round_ix in range(n_rounds):
        for run in (live_run, frozen_run) if round_ix % 2 == 0 else (frozen_run, live_run):
            run()

    per_msg = lambda ts: np.array(ts) / len(messages) * 1e6  # noqa: E731
    live_us, frozen_us = per_msg(live_times), per_msg(frozen_times)
    live = float(live_us.min())
    frozen = float(frozen_us.min())
    # The spread across identical rounds. If it exceeds the live-minus-frozen
    # difference -- which it does on this chain -- the difference is noise and
    # the per-stage table is the number to quote.
    spread = float(np.percentile(live_us, 90) - live_us.min())
    return {
        "per_message_us": live,
        "per_message_us_no_hashing": frozen,
        "round_spread_us": spread,
        "hash_overhead_us": live - frozen,
        "hash_overhead_resolved": abs(live - frozen) > spread,
        "msgs_per_sec": 1e6 / live,
        "n_outputs": n_out,
        "n_processors_frozen": n_frozen,
    }


def per_stage_hash_cost(messages: list[AxisArray], fs: float, repeats: int) -> list[dict[str, typing.Any]]:
    """Time each stage's ``_hash_message`` on the message that stage really sees.

    The intermediate messages matter: after ``Window`` the chunk dimension is
    ``win``, after ``Spectrum`` the coordinate axes are different ones, and a
    stage that hashes cheaply on the source message may not downstream.
    """
    chain = build_chain(fs)
    run_chain(chain, messages[:8])

    inputs: list[typing.Any] = []
    out: typing.Any = messages[8]
    for stage in chain:
        inputs.append(out)
        out = stage(out)
        if _empty(out):
            break

    rows = []
    for label, stage, msg in zip(CHAIN_LABELS, chain, inputs):
        if not hasattr(stage, "_hash_message"):
            # Stateless stages hold nothing to invalidate and never hash.
            rows.append({"stage": label, "cls": type(stage).__name__, "dims": list(getattr(msg, "dims", []))})
            continue
        baseline, _ = _time(lambda: chill(msg), repeats)
        warm, _ = _time(lambda: stage._hash_message(msg), repeats)
        cold, _ = _time(lambda: stage._hash_message(chill(msg)), repeats)
        rows.append(
            {
                "stage": label,
                "cls": type(stage).__name__,
                "dims": list(getattr(msg, "dims", [])),
                "warm_us": warm * 1e6,
                "cold_us": max(cold - baseline, 0.0) * 1e6,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Section 3: how many checksums the chain actually computes
# --------------------------------------------------------------------------- #


def count_checksums(messages: list[AxisArray], fs: float) -> dict[str, float]:
    """Count crc32 calls over a run. This is the work ``fingerprint`` caches."""
    import zlib

    calls = 0
    real_crc32 = zlib.crc32

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_crc32(*args, **kwargs)

    chain = build_chain(fs)
    run_chain(chain, messages[:8])
    for msg in messages:
        chill(msg)
    zlib.crc32 = counting
    try:
        for msg in messages:
            out: typing.Any = msg
            for stage in chain:
                out = stage(out)
                if _empty(out):
                    break
    finally:
        zlib.crc32 = real_crc32
    return {"crc32_calls": float(calls), "crc32_per_message": calls / len(messages)}


# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("before", "after", "naive"), default="after")
    p.add_argument("--n-ch", type=int, default=256)
    p.add_argument("--fs", type=float, default=2000.0)
    p.add_argument("--chunk-ms", type=float, default=20.0)
    p.add_argument("--n-messages", type=int, default=200)
    p.add_argument("--n-rounds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    global DECLARE_CHUNK_DIM
    if args.arm == "before":
        DECLARE_CHUNK_DIM = False
    if args.arm == "naive" and not uncache_fingerprint():
        raise SystemExit("--arm naive needs a build of ezmsg that has CoordinateAxis.fingerprint")

    n_time = int(round(args.fs * args.chunk_ms / 1000.0))
    ch_full = channelmap_data(args.n_ch)
    ch_plain = label_data(args.n_ch)
    signal = np.random.default_rng(0).standard_normal((n_time, args.n_ch)).astype(np.float32)
    messages = [make_message(signal, ch_full, args.fs) for _ in range(args.n_messages)]

    record: dict[str, typing.Any] = {
        "arm": args.arm,
        "n_ch": args.n_ch,
        "fs": args.fs,
        "chunk_ms": args.chunk_ms,
        "n_time": n_time,
        "n_messages": args.n_messages,
        "declares_chunk_dim": DECLARE_CHUNK_DIM,
        "hash_us_full_metadata": hash_cost_per_processor(make_message(signal, ch_full, args.fs), args.repeats),
        "hash_us_plain_labels": hash_cost_per_processor(make_message(signal, ch_plain, args.fs), args.repeats),
        "per_stage": per_stage_hash_cost(messages, args.fs, args.repeats),
        "chain": chain_throughput(messages, args.fs, args.n_rounds),
        "checksums": count_checksums(messages, args.fs),
    }

    if args.json:
        print(json.dumps(record))
        return

    costs = record["hash_us_full_metadata"]
    warm = [v["warm_us"] for v in costs.values()]
    cold = [v["cold_us"] for v in costs.values()]
    print(f"arm={args.arm}   {args.n_ch} ch @ {args.fs:g} Hz, {n_time}-sample chunks\n")
    print(f"_hash_message across {len(costs)} processors, 256-ch ChannelMap (us/call):")
    print(f"  warm (fingerprint cached)   median {np.median(warm):6.3f}   sum {sum(warm):7.2f}")
    print(f"  cold (fingerprint computed) median {np.median(cold):6.3f}   sum {sum(cold):7.2f}\n")
    print("per stage, on the message that stage really receives:")
    hashing = 0.0
    for ix, row in enumerate(record["per_stage"]):
        if "warm_us" not in row:
            print(f"  {row['stage']:<13}{str(row['dims']):<26}stateless, never hashes")
            continue
        # Only the first consumer of a freshly built axis pays the checksum.
        hashing += row["cold_us"] if ix == 0 else row["warm_us"]
        print(f"  {row['stage']:<13}{str(row['dims']):<26}warm {row['warm_us']:6.3f}   cold {row['cold_us']:6.3f}")
    c = record["chain"]
    print(f"\n6-stage chain ({c['n_processors_frozen']} stateful processors, {c['n_outputs']} outputs):")
    print(f"  per message              {c['per_message_us']:8.2f} us   ({c['msgs_per_sec']:.0f} msg/s)")
    print(f"  hashing, from the table  {hashing:8.2f} us   ({hashing / c['per_message_us'] * 100:.2f}% of the chain)")
    print(f"  crc32 calls per message  {record['checksums']['crc32_per_message']:8.2f}")
    print(f"\n  cross-check by neutralising hashing: {c['hash_overhead_us']:+.2f} us")
    print(f"  round-to-round spread on this chain: {c['round_spread_us']:.2f} us", end="  ")
    print("-- resolved" if c["hash_overhead_resolved"] else "-- below the noise floor, use the table above")


if __name__ == "__main__":
    main()
