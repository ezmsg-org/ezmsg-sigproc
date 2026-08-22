"""Where to put a :obj:`~ezmsg.sigproc.materialize.Materialize` node, measured.

``mx.async_eval`` does not block -- not on an in-flight dependency, and not when
the queue is already deep -- so it is tempting to conclude that an evaluation
point can go anywhere, or everywhere. It cannot: each call costs a fixed slice
of *host* time to submit a command buffer, and in a streaming graph that host
time is the executor's, not the device's.

These four sections measure that trade. Together they say: one evaluation point,
at the end of the MLX segment, before whatever else the executor has to do.

Run from the repository root::

    uv run python benchmarks/benchmark_mlx_async_eval_placement.py
    uv run python benchmarks/benchmark_mlx_async_eval_placement.py --sections placement
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

FS = 30_000.0


def make_messages(n_msgs: int, n_ch: int, lengths, seed: int = 0) -> list[AxisArray]:
    rng = np.random.default_rng(seed)
    msgs, offset = [], 0.0
    for i in range(n_msgs):
        n = int(lengths[i % len(lengths)])
        msgs.append(
            AxisArray(
                mx.array(rng.standard_normal((n, n_ch)).astype(np.float32)),
                dims=["time", "ch"],
                axes={"time": AxisArray.TimeAxis(fs=FS, offset=offset)},
                key="bench",
            )
        )
        offset += n / FS
    mx.eval([m.data for m in msgs])
    mx.synchronize()
    return msgs


def build_chain():
    """A representative MLX-capable segment: Metal SOS filter, scaler, abs."""
    from ezmsg.sigproc.butterworthfilter import butter
    from ezmsg.sigproc.math.abs import AbsTransformer
    from ezmsg.sigproc.scaler import scaler_np

    return [
        butter(axis="time", order=4, cuton=10.0, cutoff=450.0, coef_type="sos"),
        scaler_np(time_constant=0.01, axis="time"),
        AbsTransformer(),
    ]


# ---------------------------------------------------------------------------
# 1. Does async_eval block on a dependency that is still in flight?
# ---------------------------------------------------------------------------


def section_blocking() -> None:
    print(f"\n{'=' * 78}\n1. async_eval on a dependent of in-flight work\n{'=' * 78}")

    def heavy(x):
        for _ in range(40):
            x = mx.matmul(x, x) * 1e-3
        return x

    x0 = mx.random.normal((1024, 1024))
    mx.eval(x0)
    mx.synchronize()
    t = time.perf_counter()
    mx.eval(heavy(x0))
    gpu_ms = (time.perf_counter() - t) * 1e3

    mx.synchronize()
    t0 = time.perf_counter()
    y = x0
    for _ in range(4):  # each stage consumes the previous stage's output
        y = heavy(y)
        mx.async_eval(y)
    host_ms = (time.perf_counter() - t0) * 1e3
    mx.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3

    print(f"  one stage is ~{gpu_ms:.0f} ms of GPU work; 4 chained stages enqueued")
    print(f"  host returned at {host_ms:.1f} ms, device finished at {wall_ms:.0f} ms")
    print(f"  -> async_eval does NOT wait on its inputs ({wall_ms - host_ms:.0f} ms of overlap)")


# ---------------------------------------------------------------------------
# 2. Is there back-pressure once many command buffers are outstanding?
# ---------------------------------------------------------------------------


def section_backpressure() -> None:
    print(f"\n{'=' * 78}\n2. Queue depth: does the host stall once the device falls behind?\n{'=' * 78}")

    def heavy(x):
        for _ in range(20):
            x = mx.matmul(x, x) * 1e-3
        return x

    chunks = [mx.random.normal((1024, 1024)) for _ in range(24)]
    mx.eval(chunks)
    mx.synchronize()

    t0 = time.perf_counter()
    marks = []
    for c in chunks:
        mx.async_eval(heavy(c))
        marks.append((time.perf_counter() - t0) * 1e3)
    host_ms = (time.perf_counter() - t0) * 1e3
    mx.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3

    deltas = [marks[0]] + [marks[i] - marks[i - 1] for i in range(1, len(marks))]
    print(f"  24 independent chunks enqueued in {host_ms:.1f} ms; device finished at {wall_ms:.0f} ms")
    print(f"  per-chunk host cost: min {min(deltas):.2f} ms, max {max(deltas):.2f} ms")
    print("  -> a flat spread means no back-pressure; work queues and runs back to back")


# ---------------------------------------------------------------------------
# 3. What does one async_eval actually cost the host?
# ---------------------------------------------------------------------------


def section_call_cost() -> None:
    print(f"\n{'=' * 78}\n3. Host cost of a single async_eval, small graph\n{'=' * 78}")
    x = mx.random.normal((64, 256))
    mx.eval(x)

    mx.synchronize()
    t = time.perf_counter()
    for _ in range(2000):
        mx.async_eval(mx.abs(x) + 1.0)
    with_eval = (time.perf_counter() - t) / 2000 * 1e6
    mx.synchronize()

    t = time.perf_counter()
    for _ in range(2000):
        y = mx.abs(x) + 1.0
    build = (time.perf_counter() - t) / 2000 * 1e6
    mx.eval(y)
    mx.synchronize()

    print(f"  build a 2-op graph:          {build:6.1f} µs")
    print(f"  build it and async_eval it:  {with_eval:6.1f} µs")
    print(f"  -> each evaluation point costs ~{with_eval - build:.0f} µs of host time")


# ---------------------------------------------------------------------------
# 4. Placement: per-node vs once at the tail, and the effect of concurrent work
# ---------------------------------------------------------------------------


def _run_chain(msgs, mode, cpu_reps=0, scratch=None):
    procs = build_chain()

    def once(msg):
        m = msg
        for p in procs:
            m = p(m)
            if mode == "per-node" and m.data.size:
                mx.async_eval(m.data)
        if mode == "blocking-tail":
            mx.eval(m.data)
        elif mode != "off" and m.data.size:
            # "off" leaves the graph lazy: nothing forces it until the NumPy
            # conversion below, which is the placement this is measured against.
            mx.async_eval(m.data)
        if scratch is None:
            return m.data
        for _ in range(cpu_reps):  # stand-in for whatever else the executor runs
            np.dot(scratch, scratch)
        return np.asarray(m.data)  # process boundary; forces completion either way

    for msg in msgs[:30]:
        once(msg)
    mx.synchronize()
    t0 = time.perf_counter()
    for msg in msgs:
        once(msg)
    mx.synchronize()
    return (time.perf_counter() - t0) / len(msgs) * 1e6


def section_placement() -> None:
    print(f"\n{'=' * 78}\n4. Placement in a 3-node MLX chain (µs/message, min of 3)\n{'=' * 78}")
    print(f"  {'shape':<14} {'blocking tail':>14} {'async tail':>12} {'per-node':>10} {'per-node vs tail':>18}")
    for n_ch, lengths in ((256, [30, 31, 33, 30, 64, 30, 45, 30]), (1024, [128]), (2048, [1024])):
        msgs = make_messages(150 if n_ch > 1024 else 400, n_ch, lengths)
        label = f"{lengths[0] if len(lengths) == 1 else 'jitter'}x{n_ch}"
        blocking = min(_run_chain(msgs, "blocking-tail") for _ in range(3))
        tail = min(_run_chain(msgs, "async-tail") for _ in range(3))
        per_node = min(_run_chain(msgs, "per-node") for _ in range(3))
        print(f"  {label:<14} {blocking:>14.1f} {tail:>12.1f} {per_node:>10.1f} {per_node / tail:>17.2f}x")


def section_concurrent_work() -> None:
    """The case that decides placement: an executor with something else to do."""
    print(f"\n{'=' * 78}\n5. Early async_eval vs deferring to the NumPy conversion\n{'=' * 78}")
    scratch = np.random.randn(400, 400)
    msgs = make_messages(300, 1024, [128])
    print(f"  {'other CPU work':<18} {'deferred':>10} {'early async_eval':>18} {'gain':>7}")
    for reps, label in ((0, "none"), (2, "~0.5 ms"), (8, "~2 ms"), (25, "~6 ms")):
        deferred = min(_run_chain(msgs, "off", reps, scratch) for _ in range(3))
        early = min(_run_chain(msgs, "async-tail", reps, scratch) for _ in range(3))
        print(f"  {label:<18} {deferred:>10.1f} {early:>18.1f} {deferred / early:>6.2f}x")


def section_runahead() -> None:
    """What unbounded run-ahead costs in transient memory."""
    print(f"\n{'=' * 78}\n6. Run-ahead: throughput bought with in-flight memory\n{'=' * 78}")
    msgs = make_messages(400, 1024, [512])
    baseline = mx.get_active_memory() / 2**20
    print(f"  resident input messages alone: {baseline:.0f} MiB")
    for mode, label in (("async-tail", "ASYNC (runs ahead)"), ("blocking-tail", "SYNC (lockstep)")):
        procs = build_chain()
        for msg in msgs[:30]:
            m = msg
            for p in procs:
                m = p(m)
            mx.eval(m.data)
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        for msg in msgs:
            m = msg
            for p in procs:
                m = p(m)
            if mode == "async-tail":
                mx.async_eval(m.data)
            else:
                mx.eval(m.data)
        host = time.perf_counter() - t0
        mx.synchronize()
        wall = time.perf_counter() - t0
        peak = mx.get_peak_memory() / 2**20 - baseline
        print(f"  {label:<20} wall={wall * 1e3:6.0f} ms  host={host * 1e3:6.0f} ms  in-flight peak={peak:7.1f} MiB")


SECTIONS = {
    "blocking": section_blocking,
    "backpressure": section_backpressure,
    "call_cost": section_call_cost,
    "placement": section_placement,
    "concurrent_work": section_concurrent_work,
    "runahead": section_runahead,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+", choices=list(SECTIONS), default=list(SECTIONS))
    args = parser.parse_args()
    mx.random.seed(0)
    for name in args.sections:
        SECTIONS[name]()


if __name__ == "__main__":
    main()
