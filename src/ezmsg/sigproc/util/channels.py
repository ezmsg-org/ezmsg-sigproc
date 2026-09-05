"""One way to say "split these channels into groups".

Several sources attach a structured ``CoordinateAxis`` to the channel dimension
carrying per-channel fields — e.g. ezmsg-blackrock's ``ChannelMap`` emits a
``ch`` axis whose ``.data`` is a numpy struct array with
``x``/``y``/``size``/``label``/``array``/``bank``/``elec``/``headstage``.
Operations that treat channels in groups — per-bank rereferencing, block
spatial filters — need to turn that metadata into index groups.

:data:`ChannelGroupSpec` is the single spec type every such operation accepts,
and :func:`resolve_channel_groups` is the single resolver:

===============================  =============================================
spec                             meaning
===============================  =============================================
``None``                         no grouping (caller's default applies)
``"bank"``                       group by that struct-array field
``("array", "bank")``            group by the tuple of those fields
``[[0, 1, 2], [3, 4, 5]]``       explicit index groups
``fn(message, axis) -> groups``  anything else
===============================  =============================================

Groups are returned in first-appearance order along the channel axis so a
resolved grouping is reproducible and readable against the channel table.

This module also owns the two ways a transformer can fold a channel axis into
its per-message state hash: :func:`group_spec_fingerprint` (O(1), notices only
that the metadata *field* appeared or vanished) and
:func:`coord_value_fingerprint` (O(bytes), notices the *values* changing). Their
docstrings explain which failure mode each one is for.
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Callable, Sequence
from typing import Union

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from .message import resolve_feature_dim

# Whether AxisFingerprintMemo counts its hits and misses. Off unless the env var
# is set, because the answer it gives -- do axis objects survive, or is every
# message a fresh deserialization? -- is a property of a *deployed* graph's
# process layout, not something a unit test can tell you. See
# benchmarks/benchmark_memo_hit_rate.py.
_STATS_ENABLED = bool(os.environ.get("EZMSG_SIGPROC_FINGERPRINT_STATS"))
_STATS: dict[str, list[int]] = {}


def fingerprint_stats() -> dict[str, tuple[int, int, int]]:
    """``{label: (calls, digests_computed, mapping_hits)}`` for this process.

    Empty unless ``EZMSG_SIGPROC_FINGERPRINT_STATS=1``. Counters are
    per-process, so a multi-process graph has to collect them from each worker.

    ``digests_computed`` is the number that matters -- how often the memo failed
    to save the O(bytes) work. ``mapping_hits`` separates the two shortcuts: the
    whole-``axes``-mapping check, which only fires when an upstream node passed
    the mapping through untouched, from the per-array check that does the real
    work (most nodes rebuild the mapping via ``replace(..., axes={...})`` even
    when the axis objects inside it are unchanged).
    """
    return {label: (c[0], c[1], c[2]) for label, c in _STATS.items()}


def reset_fingerprint_stats() -> None:
    """Zero every counter in this process."""
    for counts in _STATS.values():
        counts[0] = counts[1] = counts[2] = 0


ChannelGroupSpec = Union[
    str,
    Sequence[str],
    Sequence[Sequence[int]],
    Callable[[AxisArray, str], "Sequence[Sequence[int]] | None"],
]
"""How to split a channel axis into groups. See the module docstring."""


def validate_channel_groups(groups: Sequence[Sequence[int]], n_channels: int) -> None:
    """Raise ``ValueError`` unless *groups* are in-range and pairwise disjoint.

    Disjointness matters because every consumer of a grouping assumes a channel
    belongs to at most one group — a channel listed twice would be rereferenced
    twice, or would have two weight blocks written to the same output.
    Not every channel has to appear: omitted channels are the caller's business
    (rereferencing passes them through unchanged).

    An empty *groups* list, or empty groups within it, validates trivially.
    """
    seen = np.zeros(n_channels, dtype=bool)
    for group in groups:
        idx = np.asarray(group, dtype=np.intp).reshape(-1)
        if idx.size == 0:
            continue
        if np.any(idx < 0) or np.any(idx >= n_channels):
            raise ValueError(f"channel groups contain out-of-range indices (valid range: 0..{n_channels - 1})")
        if np.any(seen[idx]) or np.unique(idx).size != idx.size:
            raise ValueError("channel groups overlap; each channel may belong to at most one group")
        seen[idx] = True


def _groups_by_value(values: np.ndarray) -> list[np.ndarray]:
    """Indices grouped by equal value, ordered by first appearance."""
    _, first, inverse = np.unique(values, return_index=True, return_inverse=True)
    inverse = np.reshape(inverse, -1)
    # np.unique orders by value; re-rank so group 0 is the one seen first.
    order = np.argsort(first, kind="stable")
    rank = np.empty(order.size, dtype=np.intp)
    rank[order] = np.arange(order.size)
    keys = rank[inverse]
    sorter = np.argsort(keys, kind="stable")
    bounds = np.searchsorted(keys[sorter], np.arange(order.size + 1))
    return [sorter[bounds[g] : bounds[g + 1]] for g in range(order.size)]


def channel_groups_from_field(
    message: AxisArray,
    axis: str | None = None,
    field: str | Sequence[str] = "bank",
) -> list[np.ndarray] | None:
    """Group channel indices by one or more fields of a structured coordinate axis.

    Args:
        message: Message whose ``axis`` coordinate is a structured
            ``CoordinateAxis`` (its ``.data`` is a structured numpy array).
        axis: Channel axis name. ``None`` defaults to the last dimension.
        field: Struct-array field to group by (e.g. ``"bank"``), or a sequence
            of fields to group by their tuple (e.g. ``("array", "bank")``).

    Returns:
        Index groups, one per distinct value, in first-appearance order.
        ``None`` when the axis carries no usable structured field (no such axis,
        no ``.data``, unstructured ``.data``, a field is absent, or the
        per-channel length doesn't match the data). Returning ``None`` rather
        than a single all-channel group lets callers distinguish "no metadata,
        fall back to my default" from "one bank".
    """
    axis = axis or resolve_feature_dim(message)
    ax = message.axes.get(axis)
    data = getattr(ax, "data", None)
    names = getattr(getattr(data, "dtype", None), "names", None)
    fields = (field,) if isinstance(field, str) else tuple(field)
    if not names or not fields or any(f not in names for f in fields):
        return None

    if data.shape[0] != message.data.shape[message.get_axis_idx(axis)]:
        return None

    values = data[fields[0]] if len(fields) == 1 else data[list(fields)]
    return _groups_by_value(values)


def resolve_channel_groups(
    message: AxisArray,
    axis: str | None,
    spec: ChannelGroupSpec | None,
) -> list[np.ndarray] | None:
    """Turn a :data:`ChannelGroupSpec` into validated index groups.

    Returns ``None`` when *spec* is ``None`` or when a metadata-derived spec
    finds nothing to group by — in both cases the caller applies its own
    default (typically "all channels in one group").
    """
    if spec is None:
        return None

    fields = group_spec_fields(spec)
    if fields is not None:
        # group_spec_fields discriminates on the first element so the hot path
        # stays O(1); a mixed spec is caught here, once, instead of silently
        # resolving to nothing.
        if not all(isinstance(item, str) for item in fields):
            raise ValueError(f"channel group spec mixes field names with index groups: {spec!r}")
        groups = channel_groups_from_field(message, axis, fields)
    elif callable(spec):
        groups = spec(message, axis or resolve_feature_dim(message))
    else:
        groups = spec

    if groups is None:
        return None
    axis = axis or resolve_feature_dim(message)
    out = [np.asarray(group, dtype=np.intp).reshape(-1) for group in groups]
    validate_channel_groups(out, message.data.shape[message.get_axis_idx(axis)])
    return out


def group_spec_fields(spec: ChannelGroupSpec | None) -> tuple[str, ...] | None:
    """The metadata field names *spec* groups by, or ``None`` if it needs none.

    Explicit index groups, callables and ``None`` all return ``None``: nothing
    about them can change with the message.

    Deliberately O(1) — a field spec is discriminated by its *first* element,
    never by scanning all of them, because this runs on the per-message hash
    path. :func:`resolve_channel_groups` does the full homogeneity check once,
    at state reset, so a malformed spec still fails loudly.
    """
    if isinstance(spec, str):
        return (spec,)
    if spec is None or callable(spec):
        return None
    return tuple(spec) if len(spec) and isinstance(spec[0], str) else None


def group_spec_fingerprint(
    message: AxisArray,
    axis: str | None,
    spec: ChannelGroupSpec | None,
) -> tuple:
    """O(1) summary of whether *spec* can resolve against this message.

    Transformers fold this into their per-message state hash so a stream that
    gains or loses the grouping field re-resolves its groups, without paying to
    hash the field's bytes on every message. Field *values* changing under a
    fixed key and channel count is deliberately not detected — a genuine
    remap arrives with a new key or channel count.

    That concession is safe for a *grouping* (a stale grouping is arithmetic on
    the wrong partition, which a changed key or channel count would have caught)
    but not for every consumer of channel metadata. An operation whose cached
    state is a set of resolved *indices* -- where a stale answer emits one
    channel's samples under another channel's label -- wants
    :func:`coord_value_fingerprint` instead, which costs O(bytes) but actually
    tracks the values. The two are a deliberate pair; pick by whether a silent
    stale answer is recoverable downstream.

    The two common specs -- ``None`` and a single field name -- are classified
    inline rather than through :func:`group_spec_fields`, because at this call
    rate the function call itself is a measurable share of the cost. Both
    shortcuts must agree with that function; everything less trivial defers
    to it.
    """
    if spec is None:
        return ()
    fields = (spec,) if isinstance(spec, str) else group_spec_fields(spec)
    if fields is None:
        return ()
    ax = message.axes.get(axis or resolve_feature_dim(message))
    names = getattr(getattr(getattr(ax, "data", None), "dtype", None), "names", None)
    return (bool(names) and all(field in names for field in fields),)


def array_value_fingerprint(arr: np.ndarray) -> tuple:
    """Content digest of one array: ``(dtype, shape, checksum)``.

    The shared primitive under :func:`coord_value_fingerprint`, also used
    directly by consumers that already hold the array (e.g.
    :class:`~ezmsg.sigproc.concat.ConcatProcessor`, fingerprinting each axis it
    caches).

    ``zlib.crc32`` rather than ``hash(arr.tobytes())`` because the bottleneck is
    the hash, not the copy. Measured on a 256-channel ChannelMap axis (27.6 kB,
    Apple M-series): the ``tobytes()`` copy is 0.30 µs (94 GB/s) while CPython's
    siphash over the result is 4.7 µs (5.5 GB/s); ``crc32`` reads the array's
    buffer directly at 29 GB/s for 0.94 µs total -- 5.3x cheaper.

    The tradeoff is a 32-bit checksum, so a collision means a missed state
    reset. ``dtype`` and ``shape`` ride along both because they are nearly free
    and because they carry most of the structural change a checksum could alias.

    The dtype goes in as the ``np.dtype`` object, not ``str(dtype)``: numpy
    builds a structured dtype's repr field by field, which costs 9.8 µs for the
    eight-field ChannelMap above -- ten times the checksum it was annotating.
    The object is hashable and compares by value, so it does the same job for
    0.02 µs.

    ``crc32`` needs a C-contiguous buffer, which a struct-array *field* view
    never is, so the gather is explicit here rather than left to fail.
    """
    arr = np.ascontiguousarray(arr)
    if arr.dtype.hasobject:
        # An object array's buffer is pointers: two equal arrays built from
        # distinct string objects have different bytes, so checksumming it
        # would reset the state on every message. Widen to a real dtype first.
        try:
            arr = np.ascontiguousarray(arr.astype("U"))
        except (TypeError, ValueError):
            # Elements with no string form -- vanishingly rare on a coordinate
            # axis. Correctness over speed: repr is content-based and stable.
            return (arr.dtype, arr.shape, repr(arr.tolist()))
    return (arr.dtype, arr.shape, zlib.crc32(arr))


def coord_value_fingerprint(
    message: AxisArray,
    axis: str | None,
    fields: Sequence[str] | None = None,
) -> tuple:
    """Digest of the coordinate *values* on *axis*, restricted to *fields*.

    The value-sensitive counterpart to :func:`group_spec_fingerprint`, for
    transformers that cache indices resolved against coordinate values (labels,
    regex matches, field matches). Folding this into ``_hash_message`` makes such
    a cache re-resolve when a source renames, reorders or swaps out channels
    without changing its key or channel count.

    Args:
        message: The message whose axis is being fingerprinted.
        axis: Coordinate axis name. ``None`` defaults to the last dimension.
        fields: Struct-array fields the consumer actually matches against.
            ``None`` digests the whole coordinate array, which is what an
            unstructured (plain label) axis needs. A named field absent from the
            dtype contributes ``None``, so gaining or losing it still registers.

    Returns:
        A hashable tuple, empty when the axis carries no coordinate data.

    Restricting to *fields* is about **invalidation correctness, not speed**: a
    source that recomputes float ``x``/``y`` positions each message would
    otherwise reset the state continuously, even for a selection that only ever
    reads ``label``. It is sometimes also cheaper and sometimes not. Measured on
    a 256-channel ChannelMap (eight fields, 108 B itemsize, 27.6 kB total):

    ==============================  ==========  ==========================
    ``fields``                      cost        vs. whole axis (1.15 µs)
    ==============================  ==========  ==========================
    ``('bank',)`` (U2, 7% of bytes)   0.95 µs   cheaper
    ``('array', 'bank')`` (11%)       1.39 µs   *more expensive*
    ``('label',)`` (U16, 59%)         2.65 µs   *more expensive*
    ==============================  ==========  ==========================

    A wide field loses because extracting it is a strided gather (7-20 GB/s)
    while the whole axis is one contiguous read (29 GB/s). Fields are digested
    one at a time for the same reason numpy makes multi-field indexing a trap:
    ``arr[['array', 'bank']]`` returns a view that keeps the *original*
    itemsize, so its ``tobytes()`` is the entire 27.6 kB -- asking for two of
    eight fields would otherwise cost more than asking for all of them.
    """
    ax = message.axes.get(axis or resolve_feature_dim(message))
    data = getattr(ax, "data", None)
    if data is None:
        return ()
    names = getattr(getattr(data, "dtype", None), "names", None)
    if not fields or names is None:
        return array_value_fingerprint(data)
    return tuple(array_value_fingerprint(data[f]) if f in names else None for f in fields)


class AxisFingerprintMemo:
    """Per-consumer, identity-first fingerprints of a message's coordinate axes.

    A transformer that caches anything derived from axis *values* -- resolved
    indices, output labels -- has to notice when those values change under a
    fixed key and shape, and the honest check is O(bytes). This makes it O(1)
    in the case that actually occurs.

    ``replace()`` carries ``axes``, the axis objects and their ``.data`` arrays
    by reference, so a message threaded through a chain of transformers presents
    the *same objects* every time. Two ``is`` checks -- first the whole ``axes``
    mapping, then each array -- settle it without touching the bytes. A miss
    just computes the digest, so this is a pure fast path: it can make the check
    cheaper, never wrong.

    Measured across a 20-node graph checking one 256-channel ChannelMap axis:
    99.7 µs to digest per node per message, 0.6 µs with this. After a
    cross-process hop every object is fresh, so it degrades to ~20 µs -- see
    ``benchmarks/benchmark_axis_fingerprint.py``.

    **The contract this assumes**: a coordinate array is never mutated in place.
    Messages fan out to multiple graph branches, so mutating one is already
    unsafe; this turns that into a requirement.

    One memo belongs to one consumer, which must pass the same *names* and
    *exclude* on every call -- the whole-mapping shortcut caches a single answer
    per ``axes`` object and cannot tell that the question changed.
    """

    __slots__ = ("_axes_obj", "_axes_fp", "_per_axis", "_counts")

    def __init__(self, label: str | None = None) -> None:
        self._axes_obj: object = None
        self._axes_fp: tuple = ()
        self._per_axis: dict[str, tuple] = {}
        # None unless stats are enabled, so the hot path is one `is not None`.
        # [calls, digests_computed, mapping_hits]; None unless stats are on,
        # so the hot path costs one `is not None`.
        self._counts: list[int] | None = _STATS.setdefault(label, [0, 0, 0]) if _STATS_ENABLED and label else None

    def fingerprint(
        self,
        message: AxisArray,
        names: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
    ) -> tuple:
        """Digest the coordinate axes' values, as ``((name, digest), ...)``.

        Args:
            message: Message whose axes are being fingerprinted.
            names: Restrict to these axes. ``None`` covers every coordinate
                axis, which is the safe default when the consumer's own axis
                selection is not known until state reset.
            exclude: Axes to skip even when *names* is ``None`` -- for an axis
                deliberately read live rather than cached.

        Returns:
            A hashable tuple; empty when no axis carries coordinate data. Axes
            without ``.data`` contribute nothing, since a ``LinearAxis``
            compares by value for free.
        """
        axes = message.axes
        if self._counts is not None:
            self._counts[0] += 1
        if axes is self._axes_obj:
            if self._counts is not None:
                self._counts[2] += 1
            return self._axes_fp

        computed = 0
        parts = []
        for name, ax in axes.items():
            if name in exclude or (names is not None and name not in names):
                continue
            data = getattr(ax, "data", None)
            if data is None:
                continue
            seen = self._per_axis.get(name)
            if seen is not None and seen[0] is data:
                parts.append((name, seen[1]))
                continue
            # A CoordinateAxis from ezmsg >= 3.10 derives and caches its own
            # fingerprint, which rides the pickle across a process boundary --
            # the one place this memo can never hit, since every message
            # deserializes fresh objects. Ask before digesting. Older ezmsg has
            # no such attribute and falls through, so this stays version-
            # agnostic; within one process the answer is the same for every
            # axis, so the tuple never mixes the two forms.
            fp = getattr(ax, "fingerprint", None)
            if fp is None:
                fp = array_value_fingerprint(data)
                computed += 1
            self._per_axis[name] = (data, fp)
            parts.append((name, fp))

        if self._counts is not None and computed:
            self._counts[1] += 1
        self._axes_obj, self._axes_fp = axes, tuple(parts)
        return self._axes_fp
