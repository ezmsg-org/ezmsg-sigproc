"""Measure MLX Metal compile cost for full chunks and variable-length tails.

This is intentionally a single-shot benchmark rather than a pytest-benchmark
test: compilation is cached after the first specialization, so repeated rounds
would mostly measure the warm steady state and hide the startup compile storm.

Run from the repository root, for example::

    uv run python benchmarks/benchmark_mlx_metal_tail_specialization.py
    uv run python benchmarks/benchmark_mlx_metal_tail_specialization.py --operation ewma
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import scipy.signal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("sosfilt", "ewma"), default="sosfilt")
    parser.add_argument("--channels", type=int, default=256)
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument("--chunk-size", type=int, help="Benchmark one backward-compatible CS value")
    chunk_group.add_argument("--chunk-sizes", type=int, nargs="+", help="Allowable adaptive CS specializations")
    parser.add_argument("--base-chunks", type=int, default=2)
    parser.add_argument("--tail-start", type=int, default=1)
    parser.add_argument("--tail-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=187)
    return parser


def _build_runner(operation: str, channels: int, chunk_sizes: tuple[int, ...], seed: int) -> Callable[[int], float]:
    rng = np.random.default_rng(seed)
    inputs: dict[int, mx.array] = {}

    if operation == "sosfilt":
        from ezmsg.sigproc.util.sosfilt_mlx_metal import sosfilt_mlx_metal

        sos = mx.array(
            scipy.signal.butter(
                4,
                [10.0, 450.0],
                btype="bandpass",
                fs=30_000.0,
                output="sos",
            ).astype(np.float32)
        )

        def invoke(x: mx.array):
            return sosfilt_mlx_metal(sos, x, chunk_sizes=chunk_sizes)

    else:
        from ezmsg.sigproc.util.ewma_mlx_metal import ewma_mlx_metal

        zi = mx.zeros((channels, 1), dtype=mx.float32)

        def invoke(x: mx.array):
            return ewma_mlx_metal(x, alpha=0.1, zi=zi, chunk_sizes=chunk_sizes)

    def run(n_samples: int) -> float:
        if n_samples not in inputs:
            inputs[n_samples] = mx.array(rng.standard_normal((channels, n_samples), dtype=np.float32))
        started = time.perf_counter()
        y, zf = invoke(inputs[n_samples])
        mx.eval(y, zf)
        return (time.perf_counter() - started) * 1e3

    return run


def main() -> None:
    args = _parser().parse_args()
    if args.chunk_sizes is not None:
        chunk_sizes = tuple(args.chunk_sizes)
    elif args.chunk_size is not None:
        chunk_sizes = (args.chunk_size,)
    else:
        chunk_sizes = (512,) if args.operation == "sosfilt" else (32, 1024)
    max_chunk_size = max(chunk_sizes)
    if not 1 <= args.tail_start < max_chunk_size:
        raise ValueError("tail-start must be in [1, max(chunk-sizes))")
    if args.tail_count < 1 or args.tail_start + args.tail_count > max_chunk_size:
        raise ValueError("tail-count must select tails smaller than max(chunk-sizes)")

    run = _build_runner(args.operation, args.channels, chunk_sizes, args.seed)
    full_samples = args.base_chunks * max_chunk_size

    rows: list[tuple[str, int, int, float]] = []
    rows.append(("first full", full_samples, 0, run(full_samples)))
    rows.append(("repeat full", full_samples, 0, run(full_samples)))
    for tail in range(args.tail_start, args.tail_start + args.tail_count):
        n_samples = full_samples + tail
        rows.append(("new tail", n_samples, tail, run(n_samples)))

    repeat_tail = args.tail_start
    repeat_samples = full_samples + repeat_tail
    rows.append(("repeat tail", repeat_samples, repeat_tail, run(repeat_samples)))

    print(
        f"operation={args.operation} channels={args.channels} "
        f"chunk_sizes={chunk_sizes} base_chunks={args.base_chunks}"
    )
    print(f"{'phase':<12} {'samples':>8} {'tail':>6} {'elapsed (ms)':>14}")
    for phase, n_samples, tail, elapsed_ms in rows:
        print(f"{phase:<12} {n_samples:>8} {tail:>6} {elapsed_ms:>14.2f}")

    new_tail_times = [elapsed_ms for phase, _, _, elapsed_ms in rows if phase == "new tail"]
    print(f"new-tail median: {statistics.median(new_tail_times):.2f} ms")
    print(f"repeat-tail:     {rows[-1][3]:.2f} ms")


if __name__ == "__main__":
    main()
