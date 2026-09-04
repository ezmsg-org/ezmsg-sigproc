"""What it costs to notice that a coordinate axis changed, measured.

A stateful transformer that resolves channel *labels* to array *indices* has to
decide how much of the message to fold into ``_hash_message``. Fold too little
and a source that renames or reorders channels under a fixed key and channel
count keeps getting the previous message's indices -- one channel's samples
emitted under another channel's label (ezmsg-org/ezmsg-sigproc#232). Fold too
much and every message pays for it.

:mod:`ezmsg.sigproc.util.channels` offers both answers --
``group_spec_fingerprint`` (O(1), field presence only) and
``coord_value_fingerprint`` (O(bytes), actual values) -- and this script is
where the numbers in their docstrings come from.

Three results drive how ``coord_value_fingerprint`` is written:

* **The checksum dominates, not the copy.** ``tobytes()`` runs at ~94 GB/s;
  CPython's siphash over the result manages ~5.5 GB/s. ``zlib.crc32`` reads the
  array buffer directly at ~29 GB/s, so it is ~5x cheaper end to end.
* **Restricting to one field is not reliably cheaper.** Extracting a field from
  a struct array is a strided gather; a wide field (U16 ``label``, 59% of the
  itemsize) costs more than checksumming the whole contiguous axis. The
  restriction is for invalidation *correctness* -- not resetting when an unread
  ``x``/``y`` field churns -- and is only sometimes also a speedup.
* **Never ask numpy for several fields at once.** ``arr[['array', 'bank']]``
  returns a view that keeps the *original* itemsize, so its ``tobytes()`` is the
  whole array. Two of eight fields would cost more than all eight.

Run from the repository root::

    uv run python benchmarks/benchmark_axis_fingerprint.py
    uv run python benchmarks/benchmark_axis_fingerprint.py --n-ch 1024
    uv run python benchmarks/benchmark_axis_fingerprint.py --sections mechanics,scaling
"""

from __future__ import annotations

import argparse
import timeit
import zlib

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.util.channels import coord_value_fingerprint, group_spec_fingerprint

# ezmsg-blackrock's ChannelMap ``ch`` axis: the "full metadata" case. 108 B per
# channel, of which the U16 label is 64 B.
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

SECTIONS = ("strategies", "mechanics", "shipped", "scaling")


def channelmap_axis(n_ch: int) -> AxisArray.CoordinateAxis:
    data = np.zeros(n_ch, dtype=CHANNELMAP_DTYPE)
    data["label"] = [f"elec{i:04d}" for i in range(n_ch)]
    data["x"] = np.arange(n_ch, dtype=np.float64)
    data["y"] = np.arange(n_ch, dtype=np.float64)
    data["size"] = 1.0
    data["array"] = np.arange(n_ch) // 128
    data["bank"] = np.array([("A", "B", "C", "D")[i // 64 % 4] for i in range(n_ch)])
    data["elec"] = np.arange(n_ch, dtype=np.int32) + 1
    data["headstage"] = np.arange(n_ch) // 32
    return AxisArray.CoordinateAxis(data=data, dims=["ch"])


def label_axis(n_ch: int) -> AxisArray.CoordinateAxis:
    """The plain (unstructured) label axis most sources emit."""
    return AxisArray.CoordinateAxis(data=np.array([f"ch{i:04d}" for i in range(n_ch)]), dims=["ch"])


def make_msg(ch_axis, n_ch: int, n_times: int) -> AxisArray:
    return AxisArray(
        np.zeros((n_times, n_ch), dtype=np.float32),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=30000.0), "ch": ch_axis},
        key="dev",
        attrs={"source": "bench", "session": 3},
    )


def bench(label: str, fn, number: int = 50_000, nbytes: int | None = None) -> float:
    fn()  # warm
    us = timeit.timeit(fn, number=number) / number * 1e6
    rate = f"{nbytes / us / 1000:8.2f} GB/s" if nbytes else ""
    print(f"  {label:<54} {us:8.3f} us {rate}")
    return us


def concat_fingerprint(msg: AxisArray, concat_dim: str) -> tuple:
    """Verbatim from ``ConcatTransformer._fingerprint`` (concat.py), for comparison.

    Note that concat calls this on *both* of its inputs, so its per-message cost
    is twice what is reported here.
    """
    ax = msg.axes.get(concat_dim)
    ax_hash = hash(ax.data.tobytes()) if ax is not None and hasattr(ax, "data") else None
    attrs_fp = frozenset((k, type(v).__name__, repr(v)) for k, v in (msg.attrs or {}).items())
    return (tuple(msg.dims), msg.data.shape, ax_hash, attrs_fp)


def section_strategies(struct_msg: AxisArray, plain_msg: AxisArray) -> None:
    """Every candidate answer, against the baseline hash it would replace."""
    struct = struct_msg.axes["ch"].data
    plain = plain_msg.axes["ch"].data
    n_ch = struct_msg.data.shape[1]

    print("\n== strategies ==")
    print("\n-- O(1) baselines (what transformers pay today) --")
    base = bench("hash((key, n_ch))  [SlicerTransformer today]", lambda: hash((struct_msg.key, n_ch)))
    bench("group_spec_fingerprint(msg, 'ch', None)  [default]", lambda: group_spec_fingerprint(struct_msg, "ch", None))
    bench("group_spec_fingerprint(msg, 'ch', 'bank')", lambda: group_spec_fingerprint(struct_msg, "ch", "bank"))

    print("\n-- fold the values in: hash(tobytes()) --")
    naive = bench("hash(struct.tobytes())  whole ChannelMap", lambda: hash(struct.tobytes()))
    bench("hash(labels.tobytes())  plain label axis", lambda: hash(plain.tobytes()))
    bench("hash(struct['bank'].tobytes())  one U2 field", lambda: hash(struct["bank"].tobytes()))
    bench("hash(struct['label'].tobytes())  one U16 field", lambda: hash(struct["label"].tobytes()))
    bench(
        "hash(struct[['array','bank']].tobytes())  TRAP: 2 fields",
        lambda: hash(struct[["array", "bank"]].tobytes()),
    )

    print("\n-- fold the values in: zlib.crc32 (what util.channels uses) --")
    crc = bench("zlib.crc32(struct)  whole ChannelMap", lambda: zlib.crc32(struct), nbytes=struct.nbytes)
    bench("zlib.crc32(labels)  plain label axis", lambda: zlib.crc32(plain), nbytes=plain.nbytes)

    print("\n-- concat.py's _fingerprint (per input; concat calls it twice) --")
    concat = bench("_fingerprint(msg)  struct axis + 2 attrs", lambda: concat_fingerprint(struct_msg, "ch"))
    bench("_fingerprint(msg)  plain labels + 2 attrs", lambda: concat_fingerprint(plain_msg, "ch"))
    bench(
        "  ...its attrs frozenset alone",
        lambda: frozenset((k, type(v).__name__, repr(v)) for k, v in (struct_msg.attrs or {}).items()),
    )

    print("\n-- reference points --")
    guarded = bench("the work being guarded: data[:, :n/2] copy", lambda: struct_msg.data[:, : n_ch // 2].copy())
    bench("np.mean(data, axis=1)  (a cheap real op)", lambda: np.mean(struct_msg.data, axis=1))

    print("\n-- verdict (added cost over the O(1) baseline) --")
    for name, cost in (
        ("hash(tobytes()), whole", naive),
        ("crc32, whole", crc),
        ("concat _fingerprint x2", concat * 2),
    ):
        delta = cost - base
        print(
            f"  {name:<26} +{delta:6.3f} us/msg"
            f"   = {delta / guarded * 100:6.1f}% of the guarded copy"
            # delta us/msg * 1000 msg/s = delta ms/s = delta/10 percent of a core.
            f"   {delta / 10:5.2f}% of a core @ 1 kHz"
        )


def section_mechanics(struct_msg: AxisArray) -> None:
    """Why crc32, and why the dtype goes in as an object rather than a string."""
    struct = struct_msg.axes["ch"].data
    n_ch = struct_msg.data.shape[1]

    print("\n== mechanics ==")
    print(f"\nstruct itemsize {struct.dtype.itemsize} B, total {struct.nbytes} B")
    for name in CHANNELMAP_DTYPE.names:
        view = struct[name]
        print(
            f"  field {name:<10} itemsize {view.dtype.itemsize:>3} B  "
            f"packed {view.dtype.itemsize * n_ch:>7} B  contiguous={view.flags['C_CONTIGUOUS']}"
        )
    two = struct[["array", "bank"]]
    print(f"  fields ('array','bank') -> view itemsize {two.dtype.itemsize} B, tobytes {len(two.tobytes())} B")
    print("    ^ the multi-field view keeps the full itemsize: asking for 2 of 8")
    print("      fields materializes all 8. Digest fields one at a time.")

    print("\n-- copy vs checksum: which dominates? --")
    bench("struct.tobytes()  (copy alone)", lambda: struct.tobytes(), nbytes=struct.nbytes)
    bench("hash(struct.tobytes())  (copy + siphash)", lambda: hash(struct.tobytes()), nbytes=struct.nbytes)
    bench("zlib.crc32(struct)  (no intermediate copy)", lambda: zlib.crc32(struct), nbytes=struct.nbytes)
    bench("zlib.adler32(struct)", lambda: zlib.adler32(struct), nbytes=struct.nbytes)

    print("\n-- carrying dtype alongside the checksum --")
    bench("str(struct.dtype)   structured repr, built field by field", lambda: str(struct.dtype), number=20_000)
    bench("hash(struct.dtype)  the object, hashable and value-equal", lambda: hash(struct.dtype))
    print("    ^ 'free' metadata is not free: str() of an 8-field dtype costs more")
    print("      than the checksum it annotates. util.channels stores the object.")

    print("\n-- object-dtype axes --")
    a = np.array([f"ch{i}" for i in range(n_ch)], dtype=object)
    b = np.array(["".join(("ch", str(i))) for i in range(n_ch)], dtype=object)
    print(f"  equal object arrays, raw buffer: same crc32? {zlib.crc32(a.tobytes()) == zlib.crc32(b.tobytes())}")
    print("    ^ the buffer is pointers, so a naive digest resets state every message.")
    print(f"  after .astype('U'):              same crc32? {zlib.crc32(a.astype('U')) == zlib.crc32(b.astype('U'))}")
    bench("a.astype('U') then crc32  (the widening path)", lambda: zlib.crc32(np.ascontiguousarray(a.astype("U"))))

    print("\n-- identity fast path, if a source reuses one axis object --")
    cached_obj, cached = struct, zlib.crc32(struct)
    other = channelmap_axis(n_ch).data
    bench(
        "hit:  same array object -> reuse cached digest",
        lambda: cached if struct is cached_obj else zlib.crc32(struct),
    )
    bench("miss: fresh array -> full crc32", lambda: cached if other is cached_obj else zlib.crc32(other))
    print("    ^ ~50x cheaper on a hit, but only a transformer holding state can")
    print("      do this; coord_value_fingerprint is a pure function.")


def section_shipped(struct_msg: AxisArray, plain_msg: AxisArray) -> None:
    """The function as shipped -- these are the numbers in its docstring."""
    print("\n== shipped: util.channels.coord_value_fingerprint ==\n")
    bench("cvf(struct, None)              whole ChannelMap", lambda: coord_value_fingerprint(struct_msg, "ch", None))
    bench(
        "cvf(struct, ('bank',))         U2, 7% of bytes",
        lambda: coord_value_fingerprint(struct_msg, "ch", ("bank",)),
    )
    bench(
        "cvf(struct, ('array','bank'))  11% of bytes",
        lambda: coord_value_fingerprint(struct_msg, "ch", ("array", "bank")),
    )
    bench(
        "cvf(struct, ('label',))        U16, 59% of bytes",
        lambda: coord_value_fingerprint(struct_msg, "ch", ("label",)),
    )
    bench("cvf(plain, None)               plain label axis", lambda: coord_value_fingerprint(plain_msg, "ch", None))
    print("\n    Restricting to a field is about invalidation correctness (an unread")
    print("    x/y field churning must not reset the state), and is only sometimes")
    print("    also a speedup -- a wide field costs more than the whole axis.")


def section_scaling(n_times: int) -> None:
    print("\n== scaling with channel count ==\n")
    for n in (32, 64, 128, 256, 512, 1024, 2048):
        msg = make_msg(channelmap_axis(n), n, n_times)
        arr = msg.axes["ch"].data
        bench(f"n_ch={n:<5} cvf(struct, None)", lambda m=msg: coord_value_fingerprint(m, "ch", None), number=20_000)
        bench(f"n_ch={n:<5}   hash(tobytes())", lambda a=arr: hash(a.tobytes()), number=20_000, nbytes=arr.nbytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-ch", type=int, default=256, help="channels on the coordinate axis (default: 256)")
    parser.add_argument("--n-times", type=int, default=30, help="samples per message (default: 30)")
    parser.add_argument("--sections", default=",".join(SECTIONS), help=f"comma-separated subset of {SECTIONS}")
    args = parser.parse_args()

    requested = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")

    struct_msg = make_msg(channelmap_axis(args.n_ch), args.n_ch, args.n_times)
    plain_msg = make_msg(label_axis(args.n_ch), args.n_ch, args.n_times)

    print(f"{args.n_ch} channels, {args.n_times} samples/message, float32 data")
    print(
        f"  coordinate axis: struct {struct_msg.axes['ch'].data.nbytes} B, "
        f"plain labels {plain_msg.axes['ch'].data.nbytes} B"
    )

    if "strategies" in requested:
        section_strategies(struct_msg, plain_msg)
    if "mechanics" in requested:
        section_mechanics(struct_msg)
    if "shipped" in requested:
        section_shipped(struct_msg, plain_msg)
    if "scaling" in requested:
        section_scaling(args.n_times)


if __name__ == "__main__":
    main()
