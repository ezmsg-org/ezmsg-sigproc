"""What the state-hash witness is worth in a graph that keeps growing.

``Stateful._message_hash`` walks the dims, reaches into the axes and builds a
tuple to hash, once per message per processor. In a steady stream the answer is
the same every time, and the work to prove it is the same every time too. The
witness records the objects the last answer was derived from and, when none of
them has changed identity, returns the cached answer.

The precondition is producer-side and every ezmsg source already satisfies it:
build the per-stream axes once, and replace only the chunk axis per message
(``replace(template, data=..., axes={**template.axes, "time": new_time_ax})``).
That hands every consumer the *same* coordinate axis object for the life of the
stream, and identity settles the question in one pointer comparison.

Identity is not available everywhere: unpickling hands out a new axis object per
message, so every processor downstream of a process boundary sees fresh objects
carrying identical values. That arm is measured here as ``rebuilt``, and is why
the witness falls back to comparing the axis *value* -- the fingerprint, which
rides along already computed -- rather than giving up when identity fails.

Run against the working tree for the witness arm, and against a checkout without
it for the baseline::

    git -C ../ezmsg-baseproc worktree add --detach /tmp/baseproc-nowitness <pre-witness commit>
    cp ../ezmsg-baseproc/src/ezmsg/baseproc/__version__.py /tmp/baseproc-nowitness/src/ezmsg/baseproc/

    uv run python benchmarks/benchmark_hash_witness.py --label witness
    PYTHONPATH=/tmp/baseproc-nowitness/src uv run python benchmarks/benchmark_hash_witness.py --label baseline

Pass ``--json`` to emit a record for diffing arms.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import timeit
import typing

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from ezmsg.util.messages.util import replace

from ezmsg.sigproc.ewma import EWMASettings, EWMATransformer

TARGET_S = 0.02


def _time(fn: typing.Callable[[], typing.Any], repeats: int = 7) -> float:
    timer = timeit.Timer(fn)
    probe = timer.timeit(64) / 64
    n = max(64, min(int(TARGET_S / max(probe, 1e-9)), 200_000))
    return min(timer.repeat(repeats, n)) / n * 1e6


def make_template(n_ch: int, fs: float, n_time: int, key: str = "dev") -> AxisArray:
    return AxisArray(
        np.zeros((n_time, n_ch), np.float32),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": CoordinateAxis(data=np.array([f"ch{i:03d}" for i in range(n_ch)]), dims=["ch"]),
        },
        key=key,
        chunk_dim="time",
    )


def stream(template: AxisArray, n: int, n_time: int, fs: float, *, hoist: bool) -> list[AxisArray]:
    """``hoist=True`` is the template idiom every ezmsg source uses, and gives
    every consumer in the producing process the same axis object. ``False`` is
    what the far side of a process boundary sees: a new object per message,
    carrying the same values and the same precomputed fingerprint."""
    out = []
    for i in range(n):
        axes = {**template.axes, "time": replace(template.axes["time"], offset=i * n_time / fs)}
        if not hoist:
            axes["ch"] = CoordinateAxis(data=template.axes["ch"].data, dims=["ch"])
        msg = replace(template, data=np.full((n_time, template.data.shape[1]), float(i), np.float32), axes=axes)
        for axis in msg.axes.values():
            getattr(axis, "fingerprint", None)  # a source that primes its fingerprints
        out.append(msg)
    return out


def build_chain(n_nodes: int) -> list:
    return [EWMATransformer(EWMASettings(axis="time", time_constant=0.5)) for _ in range(n_nodes)]


def run_chain(chain: list, messages: list[AxisArray]) -> float:
    gc.collect()
    t0 = time.perf_counter()
    for msg in messages:
        out: typing.Any = msg
        for stage in chain:
            out = stage(out)
    return time.perf_counter() - t0


def hit_rate(n_nodes: int, messages: list[AxisArray]) -> tuple[int, int]:
    """How often the fast path answered, by counting the validator's verdicts.

    Patched before the chain is built, since a witness compiles its validator at
    construction and a chain warmed beforehand would carry unpatched ones.
    """
    from ezmsg.baseproc import stateful as st

    if not hasattr(st, "_build_witness"):
        return (0, 0)
    hits = total = 0
    real = st._build_witness

    def counting_build(*args, **kwargs):
        witness = real(*args, **kwargs)
        if witness is None:
            return None
        validator = witness[0]

        def counted(msg, _v=validator):
            nonlocal hits, total
            out = _v(msg)
            total += 1
            hits += out
            return out

        return (counted,) + witness[1:]

    st._build_witness = counting_build
    try:
        chain = build_chain(n_nodes)
        run_chain(chain, messages[:4])
        hits = total = 0
        for msg in messages:
            out: typing.Any = msg
            for stage in chain:
                out = stage(out)
    finally:
        st._build_witness = real
    return hits, total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="witness")
    p.add_argument("--n-ch", type=int, default=16)
    p.add_argument("--n-time", type=int, default=8)
    p.add_argument("--fs", type=float, default=1000.0)
    p.add_argument("--nodes", type=int, nargs="+", default=[1, 10, 30, 100])
    p.add_argument("--n-messages", type=int, default=200)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    template = make_template(args.n_ch, args.fs, args.n_time)
    hoisted = stream(template, args.n_messages, args.n_time, args.fs, hoist=True)
    rebuilt = stream(template, args.n_messages, args.n_time, args.fs, hoist=False)

    record: dict[str, typing.Any] = {
        "label": args.label,
        "n_ch": args.n_ch,
        "n_time": args.n_time,
        "n_messages": args.n_messages,
        "per_call": {},
        "chains": {},
    }

    # --- one _hash_message call, in isolation
    for name, msgs in (("hoisted", hoisted), ("rebuilt", rebuilt)):
        proc = EWMATransformer(EWMASettings(axis="time", time_constant=0.5))
        for m in msgs[:4]:
            proc(m)
        cyc = iter(msgs * 100000)
        overhead = _time(lambda: next(iter([None])))
        record["per_call"][name] = _time(lambda: proc._hash_message(next(cyc))) - overhead

    # --- whole chains, at several depths
    for n_nodes in args.nodes:
        for name, msgs in (("hoisted", hoisted), ("rebuilt", rebuilt)):
            times = []
            for _ in range(args.rounds):
                chain = build_chain(n_nodes)
                run_chain(chain, msgs[:4])
                times.append(run_chain(chain, msgs))
            per_msg = min(times) / len(msgs) * 1e6
            hits, total = hit_rate(n_nodes, msgs)
            record["chains"].setdefault(str(n_nodes), {})[name] = {
                "us_per_message": per_msg,
                "us_per_node": per_msg / n_nodes,
                "msgs_per_sec": 1e6 / per_msg,
                "witness_hit_pct": (100.0 * hits / total) if total else None,
            }

    if args.json:
        print(json.dumps(record))
        return

    print(f"{args.label}: {args.n_ch} ch x {args.n_time} samples, {args.n_messages} messages\n")
    print("one _hash_message call:")
    for name, us in record["per_call"].items():
        print(f"  source {name:<9} {us:7.3f} us")
    print(
        f"\n{'nodes':>6}  {'hoisted us/msg':>15}{'/node':>8}{'hit%':>7}   {'rebuilt us/msg':>15}{'/node':>8}{'hit%':>7}"
    )
    for n_nodes in args.nodes:
        h = record["chains"][str(n_nodes)]["hoisted"]
        r = record["chains"][str(n_nodes)]["rebuilt"]
        hp = f"{h['witness_hit_pct']:.0f}" if h["witness_hit_pct"] is not None else "-"
        rp = f"{r['witness_hit_pct']:.0f}" if r["witness_hit_pct"] is not None else "-"
        print(
            f"{n_nodes:>6}  {h['us_per_message']:15.2f}{h['us_per_node']:8.3f}{hp:>7}   "
            f"{r['us_per_message']:15.2f}{r['us_per_node']:8.3f}{rp:>7}"
        )


if __name__ == "__main__":
    main()
