"""End-to-end streaming throughput of the MLX-capable transformers.

The A/B micro-benchmarks in ``benchmark_mlx_gist_lessons.py`` isolate single
rewrites; this one asks whether they still matter once a real node's other work
is in the frame. Run it on either side of a change (e.g. across a ``git stash``)
and compare.

Each node is fed a stream of messages whose length jitters, which is the case
that matters here: an upstream source with dynamic chunk sizing must not push
the node into recompiling or re-transferring per-message constants.

Run from the repository root::

    uv run python benchmarks/benchmark_mlx_end_to_end.py
    uv run python benchmarks/benchmark_mlx_end_to_end.py --channels 700 --messages 2000
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

FS = 30_000.0


def make_messages(n_msgs: int, n_ch: int, lengths, seed: int) -> list[AxisArray]:
    rng = np.random.default_rng(seed)
    msgs = []
    offset = 0.0
    for i in range(n_msgs):
        n = int(lengths[i % len(lengths)])
        data = mx.array(rng.standard_normal((n, n_ch)).astype(np.float32))
        msgs.append(
            AxisArray(
                data,
                dims=["time", "ch"],
                axes={"time": AxisArray.TimeAxis(fs=FS, offset=offset)},
                key="bench",
            )
        )
        offset += n / FS
    mx.eval([m.data for m in msgs])
    return msgs


def run(proc, msgs, *, warmup: int = 50) -> float:
    """µs per message, evaluating asynchronously like a streaming graph does."""
    for msg in msgs[:warmup]:
        out = proc(msg)
        if out.data.size:
            mx.async_eval(out.data)
    mx.synchronize()

    started = time.perf_counter()
    for msg in msgs:
        out = proc(msg)
        if out.data.size:
            mx.async_eval(out.data)
    mx.synchronize()
    return (time.perf_counter() - started) / len(msgs) * 1e6


def build_nodes(n_ch: int):
    from ezmsg.sigproc.butterworthfilter import butter
    from ezmsg.sigproc.downsample import downsample
    from ezmsg.sigproc.ewma import EWMATransformer
    from ezmsg.sigproc.math.log import LogSettings, LogTransformer
    from ezmsg.sigproc.scaler import scaler_np
    from ezmsg.sigproc.spectrum import spectrum

    return [
        (
            "butterworth(sos, mlx-metal)",
            lambda: butter(axis="time", order=4, cuton=10.0, cutoff=450.0, coef_type="sos"),
        ),
        ("ewma", lambda: EWMATransformer(time_constant=1.0, axis="time")),
        ("scaler", lambda: scaler_np(time_constant=1.0, axis="time")),
        ("downsample(q=4)", lambda: downsample(axis="time", factor=4)),
        ("log(clip_zero=True)", lambda: LogTransformer(LogSettings(base=10.0, clip_zero=True))),
        ("spectrum", lambda: spectrum(axis="time")),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--messages", type=int, default=1000)
    parser.add_argument("--lengths", type=int, nargs="+", default=[30, 31, 33, 30, 64, 30, 45, 30])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5, help="Runs per node; the minimum is reported")
    args = parser.parse_args()

    msgs = make_messages(args.messages, args.channels, args.lengths, args.seed)
    # spectrum keys off the chunk length, so give it a fixed one of its own.
    fixed = make_messages(args.messages, args.channels, [256], args.seed)

    print(f"channels={args.channels} messages={args.messages} lengths={args.lengths} (fixed=256 for spectrum)")
    # Report the minimum across repeats: run-to-run spread on an unquiesced
    # machine is ~6%, which is larger than several of the effects under test,
    # and noise here is strictly additive (scheduler, thermal, other processes).
    print(f"{'node':<32} {'min µs/msg':>12} {'median':>9} {'max':>9}")
    for name, factory in build_nodes(args.channels):
        stream = fixed if name == "spectrum" else msgs
        try:
            times = sorted(run(factory(), stream) for _ in range(args.repeats))
            print(f"{name:<32} {times[0]:>12.1f} {times[len(times) // 2]:>9.1f} {times[-1]:>9.1f}")
        except Exception as exc:  # a node that cannot run on MLX at all
            print(f"{name:<32} {type(exc).__name__ + ': ' + str(exc)[:48]:>12}")


if __name__ == "__main__":
    main()
