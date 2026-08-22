"""A/B micro-benchmarks for the "Writing Fast MLX" guidance applied to this package.

Each section pairs the code we ship today ("current") against a candidate
rewrite drawn from one of the guide's rules, at streaming-shaped inputs. Nothing
here is a claim on its own -- the point is to measure before changing anything.

Run from the repository root::

    uv run python benchmarks/benchmark_mlx_gist_lessons.py
    uv run python benchmarks/benchmark_mlx_gist_lessons.py --sections log scaler
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import mlx.core as mx
import numpy as np

SHAPES = ((30, 256), (128, 512), (512, 1024))
"""(n_time, n_ch) -- acquisition-order chunks from small streaming to blocky."""


def bench(fn: Callable[[], object], *, iters: int = 500, warmup: int = 50) -> tuple[float, float]:
    """Per-message cost of ``fn``, measured the way a streaming graph pays it.

    A streaming node does not block on each op: it hands work to the device and
    goes back to building the next message. So we ``async_eval`` each iteration
    -- which detaches the graph without a round trip, exactly what
    :obj:`MaterializeMode.ASYNC` does -- and synchronize once at the end.

    Returns ``(throughput_us, cpu_us)``: wall clock per message including the
    device tail, and the host-side graph-construction time alone. A change that
    only removes MLX op dispatches moves the second number; a change that
    removes device work moves the first.
    """
    for _ in range(warmup):
        mx.async_eval(fn())
    mx.synchronize()

    started = time.perf_counter()
    for _ in range(iters):
        mx.async_eval(fn())
    cpu_done = time.perf_counter()
    mx.synchronize()
    ended = time.perf_counter()
    return (ended - started) / iters * 1e6, (cpu_done - started) / iters * 1e6


def report(title: str, rows: list[tuple[str, tuple[float, float], tuple[float, float]]]) -> None:
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")
    print(f"{'case':<28} {'current µs (cpu)':>22} {'candidate µs (cpu)':>22} {'speedup':>9} {'cpu':>7}")
    for name, cur, cand in rows:
        print(
            f"{name:<28} {cur[0]:>13.1f} ({cur[1]:>5.1f}) {cand[0]:>13.1f} ({cand[1]:>5.1f}) "
            f"{cur[0] / cand[0]:>8.2f}x {cur[1] / cand[1]:>6.2f}x"
        )


# ---------------------------------------------------------------------------
# 1. math/log.py clip_zero: bool(xp.any(...)) forces a device sync per message
#    Guide: "Graph Evaluation" -- avoid accidental frequent evaluation.
# ---------------------------------------------------------------------------


def section_log() -> None:
    rows = []
    for n_t, n_ch in SHAPES:
        data = mx.abs(mx.random.normal((n_t, n_ch))) + 1e-3
        tiny = float(np.finfo(np.float32).smallest_normal)
        log_base = float(np.log(10.0))

        def current():
            d = data
            if bool(mx.any(d <= 0)):  # <-- host sync
                d = mx.clip(d, tiny, None)
            return mx.log(d) / log_base

        def candidate():
            return mx.log(mx.clip(data, tiny, None)) / log_base

        rows.append((f"{n_t}x{n_ch}", bench(current), bench(candidate)))
    report("1. Log(clip_zero=True): host sync vs unconditional device clip", rows)


# ---------------------------------------------------------------------------
# 2. scaler.py / rollingscaler.py / digitize.py: xp.asarray(python_scalar)
#    Guide: "Type Promotion" -- pass Python scalars, not wrapped arrays.
# ---------------------------------------------------------------------------


def section_scaler() -> None:
    rows = []
    for n_t, n_ch in SHAPES:
        data = mx.random.normal((n_t, n_ch))
        mean = mx.random.normal((n_t, n_ch))
        var_sq = mx.abs(mx.random.normal((n_t, n_ch)))

        def current():
            varis = var_sq - mean**2
            mask = varis > 0
            safe_varis = mx.where(mask, varis, mx.array(0.0, dtype=varis.dtype))
            std = safe_varis**0.5
            safe_std = mx.where(mask, std, mx.array(1.0, dtype=std.dtype))
            return mx.where(mask, (data - mean) / safe_std, mx.array(0.0, dtype=data.dtype))

        def candidate():
            varis = var_sq - mean**2
            mask = varis > 0
            safe_varis = mx.where(mask, varis, 0.0)
            std = mx.sqrt(safe_varis)
            safe_std = mx.where(mask, std, 1.0)
            return mx.where(mask, (data - mean) / safe_std, 0.0)

        rows.append((f"{n_t}x{n_ch}", bench(current), bench(candidate)))
    report("2. AdaptiveStandardScaler: wrapped scalars + **0.5 vs Python scalars + sqrt", rows)


# ---------------------------------------------------------------------------
# 3. ewma.py bias correction: NumPy arange/pow + host->device copy per message
#    Guide: "Graph Evaluation" / "Memory Use" -- keep the work on device, and
#    do not recompute per-message constants.
# ---------------------------------------------------------------------------


def section_ewma_bias() -> None:
    alpha = 0.01
    rows = []
    for n_t, n_ch in SHAPES:
        expected = mx.random.normal((n_t, n_ch))
        n_seen_states = [0, 100_000]

        for n_seen in n_seen_states:

            def current():
                t = n_seen + np.arange(1, n_t + 1)
                corr = 1.0 - (1.0 - alpha) ** t
                corr = corr.reshape([n_t, 1])
                return expected / mx.array(corr, dtype=expected.dtype)

            # Candidate: cache the correction per (n_seen, n_t); and once
            # (1-alpha)**t underflows float32 the correction is exactly 1.
            cache: dict[tuple[int, int], mx.array | None] = {}

            def candidate():
                key = (n_seen, n_t)
                if key not in cache:
                    t = n_seen + np.arange(1, n_t + 1)
                    corr = (1.0 - (1.0 - alpha) ** t).astype(np.float32)
                    cache[key] = None if np.all(corr == 1.0) else mx.array(corr.reshape(n_t, 1))
                c = cache[key]
                return expected if c is None else expected / c

            rows.append((f"{n_t}x{n_ch} n_seen={n_seen}", bench(current), bench(candidate)))
    report("3. EWMA bias correction: per-message NumPy+transfer vs cached/elided", rows)


# ---------------------------------------------------------------------------
# 4. Per-launch scratch arrays: ewma coef and chunked_scan valid_length are
#    rebuilt on every call. Guide: "Memory Use" -- do not recreate constants.
# ---------------------------------------------------------------------------


def section_kernel_constants() -> None:
    from ezmsg.sigproc.util.ewma_mlx_metal import _kernel

    rows = []
    alpha = 0.1
    for n_t, n_ch in ((30, 256), (128, 512)):
        cs = 32 if n_t <= 32 else 1024
        x = mx.random.normal((n_ch, cs))
        zi = mx.zeros((n_ch,))

        def current():
            coef = mx.array([float(alpha), float(1.0 - alpha)], dtype=mx.float32)
            valid_length = mx.array([n_t], dtype=mx.uint32)
            return _kernel(
                inputs=[x, coef, zi, valid_length],
                template=[("CS", cs)],
                grid=(n_ch * cs, 1, 1),
                threadgroup=(cs, 1, 1),
                output_shapes=[(n_ch, cs), (n_ch,)],
                output_dtypes=[mx.float32, mx.float32],
            )

        coef_c = mx.array([float(alpha), float(1.0 - alpha)], dtype=mx.float32)
        vl_c = mx.array([n_t], dtype=mx.uint32)
        mx.eval(coef_c, vl_c)

        def candidate():
            return _kernel(
                inputs=[x, coef_c, zi, vl_c],
                template=[("CS", cs)],
                grid=(n_ch * cs, 1, 1),
                threadgroup=(cs, 1, 1),
                output_shapes=[(n_ch, cs), (n_ch,)],
                output_dtypes=[mx.float32, mx.float32],
            )

        rows.append((f"ch={n_ch} CS={cs}", bench(current), bench(candidate)))
    report("4. Metal kernel constants: rebuilt per launch vs cached", rows)


# ---------------------------------------------------------------------------
# 5. downsample.py: integer-array gather vs the strided slice it is equivalent to
#    Guide: "Operations" -- prefer take/slicing over fancy indexing.
# ---------------------------------------------------------------------------


def section_downsample() -> None:
    rows = []
    for n_t, n_ch in SHAPES:
        for q in (2, 8):
            data = mx.random.normal((n_t, n_ch))
            mx.eval(data)
            # The shipped code hands `slice_along_axis` a NumPy index array,
            # which MLX rejects outright -- so "current" here is the naive fix
            # (build the index array on device) rather than the shipped code.
            idx_mx = mx.array(np.arange(0, n_t, q))
            mx.eval(idx_mx)

            def current():
                return data[idx_mx]  # gather

            def cand_take():
                return mx.take(data, idx_mx, axis=0)

            def cand_slice():
                return data[0:n_t:q]

            cur = bench(current)
            rows.append((f"{n_t}x{n_ch} q={q} -> take", cur, bench(cand_take)))
            rows.append((f"{n_t}x{n_ch} q={q} -> slice", cur, bench(cand_slice)))
    report("5. Downsample selection: gather vs mx.take vs strided slice", rows)


# ---------------------------------------------------------------------------
# 6. affinetransform.py stacked A|B: concat a ones column then matmul
#    Guide: "Operations" -- broadcasting over concatenation; mx.addmm for a@b+c.
# ---------------------------------------------------------------------------


def section_affine_bias() -> None:
    rows = []
    for n_t, n_ch in SHAPES:
        n_out = n_ch
        stacked = mx.random.normal((n_ch + 1, n_out))
        weights, bias = stacked[:n_ch], stacked[n_ch : n_ch + 1]
        data = mx.random.normal((n_t, n_ch))
        mx.eval(weights, bias)

        def current():
            d = mx.concatenate((data, mx.ones((n_t, 1), dtype=data.dtype)), axis=-1)
            return mx.matmul(d, stacked)

        def cand_broadcast():
            return mx.matmul(data, weights) + bias

        def cand_addmm():
            return mx.addmm(bias, data, weights)

        cur = bench(current)
        rows.append((f"{n_t}x{n_ch} -> matmul+add", cur, bench(cand_broadcast)))
        rows.append((f"{n_t}x{n_ch} -> addmm", cur, bench(cand_addmm)))
    report("6. AffineTransform bias row: concat+matmul vs broadcast vs addmm", rows)


# ---------------------------------------------------------------------------
# 7. affinetransform.py grouped CAR: data - (data @ project) @ spread
#    Guide: "Operations" -- mx.addmm for a@b+c.
# ---------------------------------------------------------------------------


def section_rereference() -> None:
    rows = []
    for n_t, n_ch in SHAPES:
        n_groups = max(2, n_ch // 32)
        project = mx.random.normal((n_ch, n_groups))
        spread = mx.random.normal((n_groups, n_ch))
        data = mx.random.normal((n_t, n_ch))
        mx.eval(project, spread)

        def current():
            return data - mx.matmul(mx.matmul(data, project), spread)

        def candidate():
            return mx.addmm(data, mx.matmul(data, project), spread, alpha=-1.0, beta=1.0)

        rows.append((f"{n_t}x{n_ch} g={n_groups}", bench(current), bench(candidate)))
    report("7. Grouped rereference: subtract vs addmm(alpha=-1)", rows)


# ---------------------------------------------------------------------------
# 8. spectrum.py power transform: abs(x)**2 does a sqrt then squares it.
#    Guide: "Operations" -- do not pay for work you throw away.
# ---------------------------------------------------------------------------


def section_spectrum_power() -> None:
    rows = []
    scale = 3.7
    for n_t, n_ch in SHAPES:
        spec = mx.random.normal((n_t // 2 + 1, n_ch)) + 1j * mx.random.normal((n_t // 2 + 1, n_ch))
        mx.eval(spec)

        def current():
            return (mx.abs(spec) ** 2.0) / scale

        def candidate():
            return (spec.real * spec.real + spec.imag * spec.imag) / scale

        rows.append((f"{n_t}x{n_ch} power", bench(current), bench(candidate)))

        def current_db():
            return 10 * mx.log10((mx.abs(spec) ** 2.0) / scale)

        def candidate_db():
            return 10 * mx.log10((spec.real * spec.real + spec.imag * spec.imag) / scale)

        rows.append((f"{n_t}x{n_ch} rel_db", bench(current_db), bench(candidate_db)))
    report("8. Spectrum power: abs(x)**2 vs real^2+imag^2", rows)


# ---------------------------------------------------------------------------
# 9. Weight orientation. Guide: "Operations" -- x @ W.T beats x @ W.
# ---------------------------------------------------------------------------


def section_weight_orientation() -> None:
    rows = []
    for n_t, n_ch in SHAPES:
        w = mx.random.normal((n_ch, n_ch))
        wt = mx.array(np.ascontiguousarray(np.asarray(w).T))
        data = mx.random.normal((n_t, n_ch))
        mx.eval(w, wt, data)

        def current():
            return mx.matmul(data, w)

        def candidate():
            return mx.matmul(data, wt.T)

        rows.append((f"{n_t}x{n_ch}", bench(current), bench(candidate)))
    report("9. Dense matmul: x @ W vs x @ W_contig.T", rows)


# ---------------------------------------------------------------------------
# 10. mx.compile(shapeless=True) over a per-message chain, with the chunk length
#     varying between messages. Guide: "Compile".
# ---------------------------------------------------------------------------


def section_compile() -> None:
    """``mx.compile`` fuses elementwise chains; the scaler is the longest one we have.

    ``shapeless=True`` is the only usable form here: with dynamic chunk lengths
    a shape-specialized compile would recompile per length, which is the kernel
    inflation we already refuse to pay in the Metal kernels.
    """

    def chain(data, mean, var_sq):
        varis = var_sq - mean**2
        mask = varis > 0
        safe_std = mx.where(mask, mx.sqrt(mx.where(mask, varis, 0.0)), 1.0)
        return mx.where(mask, (data - mean) / safe_std, 0.0)

    compiled_shaped = mx.compile(chain)
    compiled_shapeless = mx.compile(chain, shapeless=True)

    rows = []
    for n_t, n_ch in SHAPES:
        data = mx.random.normal((n_t, n_ch))
        mean = mx.random.normal((n_t, n_ch))
        var_sq = mx.abs(mx.random.normal((n_t, n_ch)))
        mx.eval(data, mean, var_sq)
        base = bench(lambda: chain(data, mean, var_sq))
        rows.append((f"{n_t}x{n_ch} shaped", base, bench(lambda: compiled_shaped(data, mean, var_sq))))
        rows.append((f"{n_t}x{n_ch} shapeless", base, bench(lambda: compiled_shapeless(data, mean, var_sq))))

    # Dynamic chunk lengths: does shapeless compile survive shape churn?
    lengths = [30, 31, 33, 30, 64, 30, 45, 30]
    args = {n: tuple(mx.random.normal((n, 512)) for _ in range(3)) for n in set(lengths)}
    mx.eval([a for triple in args.values() for a in triple])
    cycles = {name: iter(lengths * 20_000) for name in ("eager", "shaped", "shapeless")}

    def dyn(fn, name):
        def run():
            return fn(*args[next(cycles[name])])

        return run

    base = bench(dyn(chain, "eager"))
    rows.append(("dynamic n x512, shaped", base, bench(dyn(compiled_shaped, "shaped"))))
    rows.append(("dynamic n x512, shapeless", base, bench(dyn(compiled_shapeless, "shapeless"))))
    report("10. mx.compile on the scaler's elementwise chain", rows)


# ---------------------------------------------------------------------------
# 11. Buffer cache growth under dynamic chunk sizing.
#     Guide: "Memory Use" -- the cache can grow large for variable shapes.
# ---------------------------------------------------------------------------


def section_cache_growth() -> None:
    import scipy.signal

    from ezmsg.sigproc.util.sosfilt_mlx_metal import sosfilt_mlx_metal

    sos = mx.array(
        scipy.signal.butter(4, [10.0, 450.0], btype="bandpass", fs=30_000.0, output="sos").astype(np.float32)
    )
    rng = np.random.default_rng(0)

    print(f"\n{'=' * 78}\n11. Buffer-cache growth under dynamic chunk lengths\n{'=' * 78}")
    for label, lengths in (
        ("fixed n=256", [256] * 400),
        ("jittered n in [200,320]", list(rng.integers(200, 320, size=400))),
    ):
        mx.clear_cache()
        mx.reset_peak_memory()
        zi = None
        for n in lengths:
            x = mx.array(rng.standard_normal((256, int(n))).astype(np.float32))
            y, zi = sosfilt_mlx_metal(sos, x, zi=zi, chunk_sizes=(512,))
            mx.eval(y, zi)
        print(
            f"  {label:<26} cache={mx.get_cache_memory() / 2**20:8.2f} MiB "
            f"active={mx.get_active_memory() / 2**20:7.2f} MiB peak={mx.get_peak_memory() / 2**20:7.2f} MiB"
        )


SECTIONS = {
    "log": section_log,
    "scaler": section_scaler,
    "ewma_bias": section_ewma_bias,
    "kernel_constants": section_kernel_constants,
    "downsample": section_downsample,
    "affine_bias": section_affine_bias,
    "rereference": section_rereference,
    "spectrum_power": section_spectrum_power,
    "weight_orientation": section_weight_orientation,
    "compile": section_compile,
    "cache_growth": section_cache_growth,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+", choices=sorted(SECTIONS), default=sorted(SECTIONS))
    args = parser.parse_args()
    mx.random.seed(0)
    for name in args.sections:
        SECTIONS[name]()


if __name__ == "__main__":
    main()
