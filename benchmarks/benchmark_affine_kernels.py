"""Compare AffineTransform matmul kernels, and check the planner's choices.

:mod:`ezmsg.sigproc.util.blockdiag` decides between a dense matmul and a
block-diagonal one using a cost model whose constants are ratios fit to
measurements like these. This script (a) times the candidate formulations
directly so those constants can be re-derived on a new machine, and (b) reports
which kernel the planner actually picks at each size, so a bad constant shows up
as a "picked the slower one" row.

Run from the repository root, for example::

    uv run python benchmarks/benchmark_affine_kernels.py
    uv run python benchmarks/benchmark_affine_kernels.py --dtype float64 --non-contiguous

Two rules of thumb the defaults encode, measured on an Apple M-series CPU with
float32 data:

* Contiguous blocks slice into views, so the block loop beats dense from a few
  hundred channels up -- but only once the chunk is long enough to amortize the
  per-block call overhead.
* Non-contiguous blocks need a gather to become contiguous, which costs more
  than the FLOPs it saves until the weight matrix gets very wide.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.affinetransform import AffineTransformSettings, AffineTransformTransformer
from ezmsg.sigproc.util.blockdiag import plan_block_matmul

CHANNEL_COUNTS = (64, 128, 256, 512, 1024, 2048)
BLOCK_SIZES = (32, 64)
CHUNK_LENGTHS = (30, 300, 3000)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--non-contiguous", action="store_true", help="Interleave each block's channels")
    parser.add_argument("--channels", type=int, nargs="+", default=CHANNEL_COUNTS)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=BLOCK_SIZES)
    parser.add_argument("--chunk-lengths", type=int, nargs="+", default=CHUNK_LENGTHS)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _weights(n_ch: int, block: int, non_contiguous: bool, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    n_blocks = n_ch // block
    if non_contiguous:
        groups = [np.arange(b, n_ch, n_blocks) for b in range(n_blocks)]
    else:
        groups = [np.arange(b * block, (b + 1) * block) for b in range(n_blocks)]
    weights = np.zeros((n_ch, n_ch))
    for group in groups:
        weights[np.ix_(group, group)] = rng.standard_normal((block, block))
    return weights, groups


def _best_of(fn, data, reps: int) -> float:
    fn(data)
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn(data)
        best = min(best, time.perf_counter() - start)
    return best * 1e6


def _kernels(weights: np.ndarray, groups: list[np.ndarray], dtype):
    """The formulations the planner chooses among, plus the one it replaced."""
    n_ch = weights.shape[0]
    weights = weights.astype(dtype)
    perm = np.concatenate(groups)
    inverse = np.empty(n_ch, dtype=np.intp)
    inverse[perm] = np.arange(n_ch)
    sub = [np.ascontiguousarray(weights[np.ix_(g, g)]) for g in groups]
    spans = []
    offset = 0
    for group in groups:
        spans.append(slice(offset, offset + group.size))
        offset += group.size

    def dense(data):
        return data @ weights

    def blocks(data):
        # Contiguous only: basic slicing on both sides, filled in place.
        out = np.empty(data.shape[:-1] + (n_ch,), dtype=data.dtype)
        for span, block in zip(spans, sub):
            np.matmul(data[..., span], block, out=out[..., span])
        return out

    def permuted_blocks(data):
        # One gather in, one gather out, contiguous blocks in between.
        gathered = data[..., perm]
        out = np.empty_like(gathered)
        for span, block in zip(spans, sub):
            np.matmul(gathered[..., span], block, out=out[..., span])
        return out[..., inverse]

    def per_block_gather(data):
        # What the block path used to do: a fancy-index gather and scatter each.
        out = np.zeros(data.shape[:-1] + (n_ch,), dtype=data.dtype)
        for group, block in zip(groups, sub):
            out[..., group] = np.take(data, group, axis=-1) @ block
        return out

    return dense, blocks, permuted_blocks, per_block_gather


def main() -> None:
    args = _parser().parse_args()
    dtype = np.dtype(args.dtype)
    contiguous = not args.non_contiguous

    print(f"dtype={dtype}, blocks {'contiguous' if contiguous else 'interleaved'} along the channel axis")
    print("times in microseconds, best of N; 'picked' is what kernel='auto' selects")
    header = f"{'n_ch':>6}{'blk':>5}{'n':>7}{'dense':>9}{'blocks':>9}{'perm':>9}{'gather':>9}  picked"
    print(header)
    print("-" * len(header))

    for n_ch in args.channels:
        for block in args.block_sizes:
            if block >= n_ch:
                continue
            weights, groups = _weights(n_ch, block, args.non_contiguous, args.seed)
            dense, blocks, permuted_blocks, per_block_gather = _kernels(weights, groups, dtype)
            for n in args.chunk_lengths:
                data = np.random.default_rng(args.seed).standard_normal((n, n_ch)).astype(dtype)
                reference = dense(data)
                for fn in (permuted_blocks, per_block_gather) + ((blocks,) if contiguous else ()):
                    assert np.allclose(fn(data), reference, atol=1e-3), fn.__name__

                reps = 200 if n <= 300 else 20
                timings = {
                    "dense": _best_of(dense, data, reps),
                    "blocks": _best_of(blocks, data, reps) if contiguous else float("nan"),
                    "perm": _best_of(permuted_blocks, data, reps),
                    "gather": _best_of(per_block_gather, data, reps),
                }

                plan = plan_block_matmul(weights, n)
                if plan is None:
                    picked = "dense"
                else:
                    picked = f"blocks({plan.n_blocks})" + ("+perm" if plan.in_perm is not None else "")

                # End-to-end through the transformer, so the reported time
                # includes message handling, not just the kernel.
                transformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
                message = AxisArray(data, dims=["time", "ch"], key="bench")
                end_to_end = _best_of(lambda _: transformer(message), data, reps)

                print(
                    f"{n_ch:6d}{block:5d}{n:7d}"
                    f"{timings['dense']:9.1f}{timings['blocks']:9.1f}{timings['perm']:9.1f}{timings['gather']:9.1f}"
                    f"  {picked} ({end_to_end:.1f} us end-to-end)"
                )


if __name__ == "__main__":
    main()
