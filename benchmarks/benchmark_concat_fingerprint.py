"""What Concat spends deciding whether its axis cache is still valid.

:class:`~ezmsg.sigproc.concat.ConcatProcessor` caches its merged output axes and
rebuilds them only when a *fingerprint* of each input changes. The fingerprint
runs on every message and the concatenate it guards does not: at 128 channels
with full ChannelMap metadata, fingerprinting used to be ~63% of ``_concat``'s
total time against ~10% for the ``xp.concat`` itself.

Two things make it cheap, both measured here:

* **Identity before content.** ``replace()`` carries ``axes``, the axis objects,
  their ``.data`` arrays and ``attrs`` by reference, so in-process these are the
  *same objects* message after message. An ``is`` check settles it in ~0.02 µs
  where a checksum costs ~1 µs. A miss (what a cross-process hop produces) just
  falls through to the digest, so it is a pure fast path.
* **crc32 over siphash.** See ``benchmark_axis_fingerprint.py``: the copy is not
  the bottleneck, ``hash(bytes)`` is, and ``zlib.crc32`` reads the array buffer
  directly at ~5x the throughput.

The legacy fingerprint is reproduced verbatim below so the comparison stays
honest as the real one changes. Note it is not merely slower -- it digests only
the *concat* axis, while the cache holds every coordinate axis, and it serves
``LinearAxis`` objects from that cache too. The ``correctness`` section shows
what that costs you.

Run from the repository root::

    uv run python benchmarks/benchmark_concat_fingerprint.py
    uv run python benchmarks/benchmark_concat_fingerprint.py --n-ch 512
    uv run python benchmarks/benchmark_concat_fingerprint.py --sections correctness
"""

from __future__ import annotations

import argparse
import timeit

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.sigproc.concat import ConcatProcessor, ConcatSettings

SECTIONS = ("throughput", "breakdown", "correctness")
POOL = 64

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
FEATURE_AXIS = CoordinateAxis(data=np.array(["spk", "sbp"]), dims=["feature"])


def ch_axis(n_ch: int, offset: int = 0) -> CoordinateAxis:
    d = np.zeros(n_ch, dtype=CHANNELMAP_DTYPE)
    d["label"] = [f"elec{i + offset:04d}" for i in range(n_ch)]
    d["x"] = np.arange(n_ch, dtype=float)
    d["bank"] = "A"
    d["elec"] = np.arange(n_ch, dtype=np.int32)
    return CoordinateAxis(data=d, dims=["ch"])


def make_msg(n_ch, n_times, key, ch, attrs, t=0.0) -> AxisArray:
    return AxisArray(
        np.zeros((n_times, n_ch, 2), dtype=np.float32),
        dims=["time", "ch", "feature"],
        axes={"time": AxisArray.TimeAxis(fs=30000.0, offset=t), "ch": ch, "feature": FEATURE_AXIS},
        key=key,
        attrs=attrs,
    )


def legacy_fingerprint(proc: ConcatProcessor, msg: AxisArray) -> tuple:
    """``ConcatProcessor._fingerprint`` as it stood before the rewrite."""
    ax = msg.axes.get(proc.settings.axis)
    ax_hash = hash(ax.data.tobytes()) if ax is not None and hasattr(ax, "data") else None
    attrs_fp = frozenset((k, type(v).__name__, repr(v)) for k, v in (msg.attrs or {}).items())
    return (tuple(msg.dims), msg.data.shape, ax_hash, attrs_fp)


def bench(label: str, fn, number: int = 5_000) -> float:
    fn()
    us = timeit.timeit(fn, number=number) / number * 1e6
    print(f"  {label:<54} {us:8.3f} us")
    return us


def _pools(n_ch, n_times, reuse_axis, n_attrs):
    attrs = {f"k{i}": f"v{i}" for i in range(n_attrs)}
    if reuse_axis:
        sa, sb = ch_axis(n_ch, 0), ch_axis(n_ch, 10_000)
        return (
            [make_msg(n_ch, n_times, "a", sa, attrs, i * 0.001) for i in range(POOL)],
            [make_msg(n_ch, n_times, "b", sb, attrs, i * 0.001) for i in range(POOL)],
        )
    return (
        [make_msg(n_ch, n_times, "a", ch_axis(n_ch, 0), dict(attrs), i * 0.001) for i in range(POOL)],
        [make_msg(n_ch, n_times, "b", ch_axis(n_ch, 10_000), dict(attrs), i * 0.001) for i in range(POOL)],
    )


def section_throughput(n_ch: int, n_times: int) -> None:
    print("\n== throughput: full _concat, per message ==")
    for label, reuse, n_attrs in (
        ("axis objects reused (in-process steady state)", True, 2),
        ("axis rebuilt per message (post-transport)", False, 2),
        ("axis reused, 12 scalar attrs", True, 12),
    ):
        print(f"\n-- {label} --")
        pa, pb = _pools(n_ch, n_times, reuse, n_attrs)
        proc = ConcatProcessor(ConcatSettings(axis="ch"))
        proc._concat(pa[0], pb[0])
        i = [0]

        def step():
            i[0] = (i[0] + 1) % POOL
            return proc._concat(pa[i[0]], pb[i[0]])

        total = bench("_concat, today", step)

        # Cycle the same pool the concat does: in the rebuilt case every message
        # is a fresh object, so this has to pay the memo misses too. Measuring a
        # single fixed message would hit the memo every time and flatter it.
        j = [0]

        def cur_fp():
            j[0] = (j[0] + 1) % POOL
            return (proc._fingerprint(pa[j[0]], proc.state.memo_a), proc._fingerprint(pb[j[0]], proc.state.memo_b))

        def old_fp():
            j[0] = (j[0] + 1) % POOL
            return (legacy_fingerprint(proc, pa[j[0]]), legacy_fingerprint(proc, pb[j[0]]))

        cur = bench("  its fingerprint (both inputs)", cur_fp)
        old = bench("  legacy fingerprint (both inputs)", old_fp)
        print(
            f"  => fingerprint {cur / total * 100:.0f}% of _concat; "
            f"{old / cur:.1f}x cheaper than legacy ({old:.2f} -> {cur:.2f} us)"
        )


def section_breakdown(n_ch: int, n_times: int) -> None:
    print("\n== breakdown: one input, memo hit vs miss ==\n")
    pa, pb = _pools(n_ch, n_times, True, 2)
    proc = ConcatProcessor(ConcatSettings(axis="ch"))
    proc._concat(pa[0], pb[0])
    m = pa[0]
    bench("memo hit   (same axes + attrs objects)", lambda: proc._fingerprint(m, proc.state.memo_a))
    bench("memo bypassed (full content digest)", lambda: proc._fingerprint(m, None))
    bench("legacy (concat axis only, siphash + repr)", lambda: legacy_fingerprint(proc, m))
    print("\n  the work being guarded, for scale:")
    bench("np.concat([a.data, b.data], axis=1)", lambda: np.concat([pa[0].data, pb[0].data], axis=1))


def section_correctness() -> None:
    """The legacy fingerprint's blind spots, as running code."""
    print("\n== correctness ==\n")

    def msg(band, t, fill):
        return AxisArray(
            np.full((4, 2, 3), fill, dtype=np.float32),
            dims=["time", "ch", "band"],
            axes={
                "time": AxisArray.TimeAxis(fs=100.0, offset=t),
                "ch": CoordinateAxis(data=np.array(["c0", "c1"]), dims=["ch"]),
                "band": CoordinateAxis(data=np.array(band), dims=["band"]),
            },
            key="dev",
        )

    proc = ConcatProcessor(ConcatSettings(axis="ch", relabel_axis=False))
    proc._concat(msg(["alpha", "beta", "gamma"], 0.0, 1.0), msg(["alpha", "beta", "gamma"], 0.0, 2.0))
    out = proc._concat(msg(["delta", "theta", "mu"], 0.04, 1.0), msg(["delta", "theta", "mu"], 0.04, 2.0))
    print(
        f"  non-concat axis relabelled -> band {[str(x) for x in out.axes['band'].data]} "
        f"(want ['delta', 'theta', 'mu'])"
    )
    print(f"  time advanced              -> offset {out.axes['time'].offset} (want 0.04)")

    a = msg(["alpha", "beta", "gamma"], 0.0, 1.0)
    b = msg(["delta", "theta", "mu"], 0.0, 1.0)
    print(
        f"\n  legacy fingerprint tells those two messages apart? "
        f"{legacy_fingerprint(proc, a) != legacy_fingerprint(proc, b)}"
    )
    print(
        f"  current fingerprint does?                          "
        f"{proc._fingerprint(a, None) != proc._fingerprint(b, None)}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-ch", type=int, default=128, help="channels per input (default: 128)")
    p.add_argument("--n-times", type=int, default=30, help="samples per message (default: 30)")
    p.add_argument("--sections", default=",".join(SECTIONS), help=f"subset of {SECTIONS}")
    args = p.parse_args()

    requested = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SECTIONS]
    if unknown:
        p.error(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")

    print(
        f"concat: 2 x ({args.n_times} x {args.n_ch} x 2) f32, ChannelMap ch axis "
        f"({args.n_ch * CHANNELMAP_DTYPE.itemsize} B)"
    )
    if "throughput" in requested:
        section_throughput(args.n_ch, args.n_times)
    if "breakdown" in requested:
        section_breakdown(args.n_ch, args.n_times)
    if "correctness" in requested:
        section_correctness()


if __name__ == "__main__":
    main()
