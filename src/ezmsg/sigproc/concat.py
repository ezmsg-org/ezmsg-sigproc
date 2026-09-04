"""Concatenate two AxisArray streams along an existing or new axis."""

from __future__ import annotations

import asyncio
import logging
import typing
from copy import deepcopy
from dataclasses import dataclass, field

import ezmsg.core as ez
import numpy as np
from array_api_compat import get_namespace
from ezmsg.util.messages.axisarray import AxisArray, AxisBase, CoordinateAxis
from ezmsg.util.messages.util import replace

from ezmsg.sigproc.util.channels import AxisFingerprintMemo
from ezmsg.sigproc.util.message import with_fingerprint

logger = logging.getLogger(__name__)

# Sentinel for "attr key was missing on this side". Distinct from any user value.
_MISSING = object()

# ---------------------------------------------------------------------------
# Shared helpers (also used by merge.py)
# ---------------------------------------------------------------------------


def _build_merged_coordinate_axis(
    axis_a: CoordinateAxis,
    axis_b: CoordinateAxis,
    relabel: bool,
    label_a: str,
    label_b: str,
) -> CoordinateAxis:
    """Build a merged CoordinateAxis from two per-input axes.

    Handles both simple (string/numeric) and structured (numpy struct) dtypes.
    When *relabel* is True and the dtype is structured, only the ``"label"``
    field is modified (or created if absent).
    """
    data_a = axis_a.data
    data_b = axis_b.data

    if data_a.dtype.names is not None or data_b.dtype.names is not None:
        return _merge_struct_axes(data_a, data_b, relabel, label_a, label_b, axis_a)

    # Simple (non-struct) path — current behaviour.
    if relabel:
        labels_a = np.array([str(lbl) + label_a for lbl in data_a])
        labels_b = np.array([str(lbl) + label_b for lbl in data_b])
    else:
        labels_a = data_a
        labels_b = data_b
    return CoordinateAxis(
        data=np.concatenate([labels_a, labels_b]),
        dims=axis_a.dims,
        unit=axis_a.unit,
    )


def _merge_struct_axes(
    data_a: np.ndarray,
    data_b: np.ndarray,
    relabel: bool,
    label_a: str,
    label_b: str,
    ref_axis: CoordinateAxis,
) -> CoordinateAxis:
    """Merge two structured-dtype coordinate arrays, preserving all fields."""
    names_a = set(data_a.dtype.names or ())
    names_b = set(data_b.dtype.names or ())

    # Build the union dtype.  Shared fields must have compatible sub-dtypes.
    union_fields: list[tuple[str, np.dtype]] = []
    seen: set[str] = set()

    for src_names, src_dtype in [
        (data_a.dtype.names or (), data_a.dtype),
        (data_b.dtype.names or (), data_b.dtype),
    ]:
        for name in src_names:
            if name in seen:
                continue
            seen.add(name)
            dt_a = data_a.dtype[name] if name in names_a else None
            dt_b = data_b.dtype[name] if name in names_b else None
            if dt_a is not None and dt_b is not None:
                resolved = _resolve_field_dtype(name, dt_a, dt_b)
            else:
                resolved = dt_a if dt_a is not None else dt_b
            union_fields.append((name, resolved))

    # If relabel and "label" is not already a field, add it.
    has_label = "label" in seen
    if relabel and not has_label:
        max_len = max(
            max((len(str(i)) for i in range(len(data_a))), default=1),
            max((len(str(i)) for i in range(len(data_b))), default=1),
        )
        suffix_len = max(len(label_a), len(label_b))
        union_fields.append(("label", np.dtype(f"U{max_len + suffix_len}")))
        has_label = True

    union_dtype = np.dtype(union_fields)
    merged = np.zeros(len(data_a) + len(data_b), dtype=union_dtype)

    # Copy values from A.
    for name in data_a.dtype.names or ():
        merged[name][: len(data_a)] = data_a[name]
    # Copy values from B.
    for name in data_b.dtype.names or ():
        merged[name][len(data_a) :] = data_b[name]

    # Relabel only the "label" field.
    if relabel and has_label:
        for i in range(len(data_a)):
            src = str(data_a[i]["label"]) if "label" in names_a else str(i)
            merged[i]["label"] = src + label_a
        for j in range(len(data_b)):
            src = str(data_b[j]["label"]) if "label" in names_b else str(j)
            merged[len(data_a) + j]["label"] = src + label_b

    return CoordinateAxis(data=merged, dims=ref_axis.dims, unit=ref_axis.unit)


def _resolve_field_dtype(name: str, dt_a: np.dtype, dt_b: np.dtype) -> np.dtype:
    """Resolve a shared struct field's dtype.  String fields use the wider width."""
    if dt_a == dt_b:
        return dt_a
    if dt_a.kind == "U" and dt_b.kind == "U":
        return np.dtype(f"U{max(dt_a.itemsize // 4, dt_b.itemsize // 4)}")
    raise ValueError(f"Incompatible dtypes for shared struct field {name!r}: {dt_a} vs {dt_b}")


# ---------------------------------------------------------------------------
# Attrs merging + promotion
# ---------------------------------------------------------------------------

_ALLOWED_ATTR_SCALARS = (str, int, float, bool, np.integer, np.floating)


def _check_attr_type(key: str, value: typing.Any) -> None:
    if not isinstance(value, _ALLOWED_ATTR_SCALARS):
        raise TypeError(
            f"Cannot merge/promote attrs key {key!r}: unsupported value type "
            f"{type(value).__name__}; only scalar str/int/float/bool are allowed."
        )


def _attrs_values_equal(a: typing.Any, b: typing.Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return a is b


def _classify_attrs(a_attrs: dict, b_attrs: dict) -> tuple[dict, dict, dict]:
    """Split two attrs dicts into equal-shared vs side-to-promote.

    Returns ``(equal, promote_a, promote_b)``:
      * ``equal[k] = v`` — present in both with equal value; kept on output ``.attrs``.
      * ``promote_a[k]``/``promote_b[k]`` — value to use on each side's concat-axis
        elements. Use the ``_MISSING`` sentinel when the key was absent on that side.
    """
    equal: dict = {}
    promote_a: dict = {}
    promote_b: dict = {}
    a_attrs = a_attrs or {}
    b_attrs = b_attrs or {}
    for k in set(a_attrs) | set(b_attrs):
        a_has, b_has = k in a_attrs, k in b_attrs
        if a_has and b_has and _attrs_values_equal(a_attrs[k], b_attrs[k]):
            _check_attr_type(k, a_attrs[k])
            equal[k] = a_attrs[k]
            continue
        if a_has:
            _check_attr_type(k, a_attrs[k])
            promote_a[k] = a_attrs[k]
        else:
            promote_a[k] = _MISSING
        if b_has:
            _check_attr_type(k, b_attrs[k])
            promote_b[k] = b_attrs[k]
        else:
            promote_b[k] = _MISSING
    return equal, promote_a, promote_b


def _promoted_field_dtype(values: list) -> np.dtype:
    """Pick a numpy dtype that can hold the supplied promoted values."""
    non_missing = [v for v in values if v is not _MISSING]
    if not non_missing:
        return np.dtype("U1")
    if any(isinstance(v, str) for v in non_missing):
        max_len = max(len(str(v)) for v in non_missing)
        return np.dtype(f"U{max(max_len, 1)}")
    if any(isinstance(v, (float, np.floating)) for v in non_missing):
        return np.dtype("f8")
    # Booleans are ints in Python; if everything is bool, prefer bool.
    if all(isinstance(v, (bool, np.bool_)) for v in non_missing):
        return np.dtype("?")
    if any(isinstance(v, (int, np.integer)) for v in non_missing):
        return np.dtype("i8")
    return np.dtype("U1")


def _sentinel_for_dtype(dt: np.dtype) -> typing.Any:
    if dt.kind == "U":
        return ""
    if dt.kind == "f":
        return float("nan")
    if dt.kind == "i":
        return 0
    if dt.kind == "b":
        return False
    return None


def _extend_struct_with_fields(
    existing: np.ndarray,
    new_fields: list[tuple[str, np.dtype, np.ndarray]],
) -> np.ndarray:
    """Append new columns to a structured array, preserving existing columns.

    ``new_fields`` is a list of ``(name, dtype, values)`` triples where
    ``values`` has length ``len(existing)``.
    """
    union: list[tuple[str, np.dtype]] = [(n, existing.dtype[n]) for n in (existing.dtype.names or ())]
    union.extend((n, dt) for (n, dt, _) in new_fields)
    union_dtype = np.dtype(union)
    out = np.zeros(len(existing), dtype=union_dtype)
    for n in existing.dtype.names or ():
        out[n] = existing[n]
    for n, _, vals in new_fields:
        out[n] = vals
    return out


def _apply_promoted_attrs(
    merged_axis: CoordinateAxis | None,
    n_a: int,
    n_b: int,
    promote_a: dict,
    promote_b: dict,
    ref_axis: CoordinateAxis | None,
    concat_dim: str,
) -> CoordinateAxis | None:
    """Inject promoted attrs as per-element fields on the concat axis.

    If ``merged_axis`` has simple (non-structured) ``.data``, it is first
    converted to a structured array with a single ``"label"`` field.
    Keys that collide with an existing struct field are dropped with a warning
    (the per-element field already in place wins).
    """
    if not promote_a and not promote_b:
        return merged_axis

    promoted_keys = sorted(set(promote_a) | set(promote_b))

    base_data: np.ndarray | None = None
    if merged_axis is not None and merged_axis.data is not None:
        if merged_axis.data.dtype.names is not None:
            base_data = merged_axis.data
        else:
            labels = merged_axis.data
            base_data = np.zeros(len(labels), dtype=np.dtype([("label", labels.dtype)]))
            base_data["label"] = labels

    existing_names = set(base_data.dtype.names or ()) if base_data is not None else set()

    new_fields: list[tuple[str, np.dtype, np.ndarray]] = []
    for k in promoted_keys:
        if k in existing_names:
            logger.warning(
                "concat: attrs key %r collides with existing struct field on %r "
                "axis; dropping promoted attr (per-element field is authoritative).",
                k,
                concat_dim,
            )
            continue
        a_val = promote_a.get(k, _MISSING)
        b_val = promote_b.get(k, _MISSING)
        all_values = [a_val] * n_a + [b_val] * n_b
        dt = _promoted_field_dtype(all_values)
        sentinel = _sentinel_for_dtype(dt)
        full = np.empty(n_a + n_b, dtype=dt)
        for i, v in enumerate(all_values):
            full[i] = sentinel if v is _MISSING else v
        new_fields.append((k, dt, full))

    if not new_fields:
        return merged_axis

    if base_data is not None:
        merged_data = _extend_struct_with_fields(base_data, new_fields)
        dims = ref_axis.dims if ref_axis is not None else [concat_dim]
        unit = ref_axis.unit if ref_axis is not None else None
        return CoordinateAxis(data=merged_data, dims=dims, unit=unit)

    # No pre-existing axis data — synthesize one solely from promoted fields.
    dtype = np.dtype([(n, dt) for (n, dt, _) in new_fields])
    out = np.zeros(n_a + n_b, dtype=dtype)
    for n, _, vals in new_fields:
        out[n] = vals
    return CoordinateAxis(data=out, dims=[concat_dim])


def _validate_shared_axes(
    a: AxisArray,
    b: AxisArray,
    concat_dim: str,
    align_dim: str | None,
    assert_flag: bool,
    chunk_dim: str | None = None,
) -> None:
    """Raise ValueError if a shared axis describes A and B differently.

    ``offset`` is compared everywhere except the chunk dimension, where the two
    inputs are separate streams whose elapsed-sample counts have no reason to
    agree bit for bit. Everywhere else it locates the axis: two spectra merged
    along ``ch`` whose ``freq`` axes start 70 Hz apart are not the same axis, and
    the output can only claim one of them.
    """
    if not assert_flag:
        return
    skip = {concat_dim, align_dim}
    for name in a.axes:
        if name in skip or name not in b.axes:
            continue
        ax_a, ax_b = a.axes[name], b.axes[name]
        if hasattr(ax_a, "data") and hasattr(ax_b, "data"):
            if not np.array_equal(ax_a.data, ax_b.data):
                raise ValueError(f"Shared axis {name!r} has different .data between inputs A and B")
        if hasattr(ax_a, "gain") and hasattr(ax_b, "gain"):
            if ax_a.gain != ax_b.gain:
                raise ValueError(f"Shared axis {name!r} has different gain: {ax_a.gain} vs {ax_b.gain}")
        if name != chunk_dim and hasattr(ax_a, "offset") and hasattr(ax_b, "offset"):
            if ax_a.offset != ax_b.offset:
                raise ValueError(f"Shared axis {name!r} has different offset: {ax_a.offset} vs {ax_b.offset}")


def _linear_axes_fingerprint(
    message: AxisArray,
    chunk_dim: str | None,
    exclude: tuple[str, ...],
) -> tuple:
    """Digest the axes that carry no coordinate data, as ``((name, ...), ...)``.

    :class:`AxisFingerprintMemo` skips these, on the reasoning that a
    ``LinearAxis`` compares by value for free -- true, but only if something
    compares it, and nothing did. So a ``freq`` axis whose gain diverged between
    A and B mid-stream never invalidated the cache, and
    :func:`_validate_shared_axes` -- which runs only on rebuild -- never re-ran
    to catch it. Caught on the first message, silent on every one after.

    ``offset`` is dropped for the chunk dimension alone, where it counts off
    elapsed samples and would rebuild the cache on every message. Its ``gain``
    is kept: a sample-rate change is a configuration change, and nothing else in
    the fingerprint would notice one.

    Returns empty when the message does not declare ``chunk_dim``. Without it
    there is no way to tell which linear axis advances, and folding in an
    advancing offset would rebuild the cache at the sample rate -- so the same
    condition that stops :func:`_build_cached_axes` caching these also stops
    them being watched. An axis read live is always current; what is lost is
    only the revalidation.
    """
    if chunk_dim is None:
        return ()
    parts = []
    for name, ax in message.axes.items():
        if name in exclude or getattr(ax, "data", None) is not None:
            continue
        gain = getattr(ax, "gain", None)
        if gain is None:
            continue
        parts.append((name, gain) if name == chunk_dim else (name, gain, ax.offset))
    return tuple(parts)


def _validate_new_axis_shapes(a: AxisArray, b: AxisArray, concat_dim: str) -> None:
    """Every dimension must match when stacking A and B along a *new* one.

    Checked per message rather than on cache rebuild. The chunk dimension is one
    of the dimensions that has to agree, and its length is deliberately not part
    of the cache fingerprint -- so a divergence appearing mid-stream would
    otherwise reach ``xp.concat`` and surface as a backend shape error naming
    neither input. The check is a zip over two or three dims.
    """
    for dim, size_a, size_b in zip(a.dims, a.data.shape, b.data.shape):
        if size_a != size_b:
            raise ValueError(
                f"Cannot concatenate along new axis {concat_dim!r}: "
                f"dimension {dim!r} has size {size_a} in A but {size_b} in B"
            )


def _build_cached_axes(
    a: AxisArray,
    concat_dim: str,
    align_dim: str | None,
    merged_concat_axis: CoordinateAxis | None,
    chunk_dim: str | None,
) -> dict[str, AxisBase]:
    """Build an owned output-axis cache of the axes that describe the stream.

    Input axes may be views into an ezmsg transport buffer whose lifetime ends
    after the current subscriber callback. Axes carrying ``.data`` are therefore
    copied into processor-owned memory.

    A ``LinearAxis`` has no buffer to own, but is still cached when it describes
    the *configuration* rather than the chunk -- its ``gain`` and ``offset`` say
    where the axis starts and how far it steps, and both belong to the same
    validated snapshot as the coordinate axes beside it. Reading half the output
    axes from a snapshot and half from the live message is what let a divergence
    on one of the live ones go unnoticed.

    Two are deliberately left live:

    * the chunk dimension, whose ``offset`` advances on every message -- caching
      it would freeze the output's time base at whatever the first message said;
    * ``align_dim``, for the same reason, which is what it was quietly working
      around before the chunk dimension could be named.

    A message that does not declare ``chunk_dim`` gives no way to tell which
    linear axis advances, so none of them is cached. That is the old behaviour,
    and it is the safe direction: an axis read live is always current.
    """
    stay_live = {align_dim, chunk_dim} if chunk_dim is not None else None
    axes: dict[str, AxisBase] = {}
    for name, ax in a.axes.items():
        if name == align_dim:
            continue
        if getattr(ax, "data", None) is None:
            # No coordinate data: a linear axis, cacheable only if we can be sure
            # it is not the one that advances.
            if stay_live is None or name in stay_live or getattr(ax, "gain", None) is None:
                continue
            axes[name] = ax  # scalars only -- nothing to copy, nothing to alias
            continue
        if name == concat_dim and merged_concat_axis is not None:
            axes[name] = merged_concat_axis
        else:
            axes[name] = deepcopy(ax)
    if concat_dim not in axes and merged_concat_axis is not None:
        axes[concat_dim] = merged_concat_axis
    return axes


# ---------------------------------------------------------------------------
# ConcatProcessor / Concat unit
# ---------------------------------------------------------------------------


class ConcatSettings(ez.Settings):
    axis: str = "ch"
    """Axis along which to concatenate the two signals."""

    align_axis: str | None = None
    """Axis along which to validate alignment between the two signals."""

    relabel_axis: bool = True
    """Whether to relabel coordinate axis labels to ensure uniqueness."""

    label_a: str = "_a"
    """Per-side label for signal A.

    Used in two distinct ways depending on whether ``axis`` is an existing
    or new dimension on the inputs:

    * **Existing axis** (``axis`` is in both inputs' ``.dims``):
      ``label_a`` is a *suffix* appended to each entry of A's existing
      coordinate-axis labels when ``relabel_axis`` is True.  Defaults to
      ``"_a"``.

    * **New axis** (``axis`` is not in either input's ``.dims``):
      ``label_a`` is used as the single ``data`` entry on the merged
      axis's CoordinateAxis at index 0.  E.g. setting
      ``label_a="spk", label_b="sbp"`` on a Merge of two
      ``(time, ch)`` streams produces a ``(time, ch, feature)`` output
      whose ``feature`` axis has ``data=["spk", "sbp"]``.
    """

    label_b: str = "_b"
    """Per-side label for signal B.

    See :attr:`label_a`.  Defaults to ``"_b"``; used as the new-axis
    label at index 1 in the new-axis case.
    """

    assert_identical_shared_axes: bool = False
    """If True, raise ValueError when shared CoordinateAxis .data arrays differ."""

    new_key: str | None = None
    """Output AxisArray key. If None, uses the key from signal A."""

    auto_coerce_backend: bool = False
    """If True, silently coerce signal B to signal A's array namespace when the
    two inputs are on mismatched backends (e.g. MLX vs numpy). Defaults to False
    (strict): a backend mismatch raises a clear error instead, since it is almost
    always an upstream bug and silent coercion hides device<->host copies."""


@dataclass
class _FingerprintMemo:
    """Last-seen objects and their digests, for one input side.

    The axis half is :class:`~ezmsg.sigproc.util.channels.AxisFingerprintMemo`,
    shared with the other transformers that cache axis-derived state; ``attrs``
    gets the same identity treatment here because concat is the only consumer
    that fingerprints them.

    The cost of *not* doing this: fingerprinting was 63% of ``_concat``'s total
    per-message time, against 10% for the concatenate it guards.
    """

    axes: AxisFingerprintMemo = field(default_factory=lambda: AxisFingerprintMemo(label="ConcatProcessor"))
    attrs_obj: object = None
    attrs_fp: frozenset | None = None


@dataclass
class ConcatState:
    queue_a: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)
    queue_b: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)
    merged_concat_axis: CoordinateAxis | None = None
    cached_axes: dict[str, AxisBase] | None = None
    merged_attrs: dict | None = None
    # Fingerprints for cache invalidation.
    a_fingerprint: tuple | None = None
    b_fingerprint: tuple | None = None
    memo_a: _FingerprintMemo = field(default_factory=_FingerprintMemo)
    memo_b: _FingerprintMemo = field(default_factory=_FingerprintMemo)


class ConcatProcessor:
    """Concatenate paired AxisArray messages from two input queues.

    Uses FIFO queue pairing (like :class:`~ezmsg.sigproc.math.add.AddProcessor`).
    No time-alignment or buffering — inputs are assumed pre-synchronized.
    """

    def __init__(self, settings: ConcatSettings):
        self.settings = settings
        self._state = ConcatState()

    @property
    def state(self) -> ConcatState:
        return self._state

    @state.setter
    def state(self, state: ConcatState | bytes | None) -> None:
        if state is not None:
            self._state = state

    def push_a(self, msg: AxisArray) -> None:
        self._state.queue_a.put_nowait(msg)

    def push_b(self, msg: AxisArray) -> None:
        self._state.queue_b.put_nowait(msg)

    async def __acall__(self) -> AxisArray:
        a = await self._state.queue_a.get()
        b = await self._state.queue_b.get()
        return self._concat(a, b)

    def _concat(self, a: AxisArray, b: AxisArray) -> AxisArray:
        """Concatenate *a* and *b* along the configured axis."""
        concat_dim = self.settings.axis
        fp_a = self._fingerprint(a, self._state.memo_a)
        fp_b = self._fingerprint(b, self._state.memo_b)
        if fp_a != self._state.a_fingerprint or fp_b != self._state.b_fingerprint:
            self._rebuild_cache(a, b)
            self._state.a_fingerprint = fp_a
            self._state.b_fingerprint = fp_b

        new_axis = concat_dim not in a.dims

        xp = get_namespace(a.data)
        xp_b = get_namespace(b.data)
        if xp_b is not xp:
            if self.settings.auto_coerce_backend:
                b = replace(b, data=xp.asarray(b.data))
            else:
                raise TypeError(
                    f"Concat received inputs on mismatched backends: "
                    f"a.data namespace={xp.__name__}, b.data namespace={xp_b.__name__}. "
                    f"Coerce both inputs to one backend upstream "
                    f"(e.g., via ezmsg.sigproc.asarray.AsArrayTransformer) before merging, "
                    f"or set ConcatSettings(auto_coerce_backend=True) to coerce B to A's backend."
                )

        # expand_dims for new-axis concatenation.
        if new_axis:
            _validate_new_axis_shapes(a, b, concat_dim)
            a = replace(a, data=xp.expand_dims(a.data, axis=-1), dims=[*a.dims, concat_dim])
            b = replace(b, data=xp.expand_dims(b.data, axis=-1), dims=[*b.dims, concat_dim])

        concat_idx = a.dims.index(concat_dim)
        data = xp.concat([a.data, b.data], axis=concat_idx)

        # Build axes: owned coordinate axes from the cache, everything else
        # (alignment axis, LinearAxes) live from a, in a's original order.
        cached = self._state.cached_axes
        if cached is None:
            axes = dict(a.axes)
        else:
            axes = {name: cached.get(name, ax) for name, ax in a.axes.items()}
            # A concat axis created for a *new* dimension is not in a.axes.
            for name, ax in cached.items():
                if name not in axes:
                    axes[name] = ax

        key = self.settings.new_key if self.settings.new_key is not None else a.key
        attrs = dict(self._state.merged_attrs) if self._state.merged_attrs else {}
        # Built fresh rather than by replace(), so the layout has to be carried
        # over explicitly. A concat along a *new* dimension leaves A's chunk
        # dimension intact; concatenating along the chunk dimension itself would
        # not, hence the membership check.
        chunk_dim = a.chunk_dim if a.chunk_dim in a.dims else None
        return AxisArray(data, dims=list(a.dims), axes=axes, key=key, attrs=attrs, chunk_dim=chunk_dim)

    def _fingerprint(self, msg: AxisArray, memo: _FingerprintMemo | None = None) -> tuple:
        """Summarize everything ``_rebuild_cache`` reads, so the cache invalidates.

        Every coordinate axis, not just the concat axis: ``_build_cached_axes``
        copies all of them into ``cached_axes``, so a *different* axis changing
        its values (e.g. a band axis relabelled mid-stream) leaves the output
        carrying the first message's copy. Axes without ``.data`` are read live
        rather than cached, so they contribute nothing here.

        *memo* short-circuits the content digests on object identity; see
        :class:`_FingerprintMemo`. Passing ``None`` computes everything from
        scratch, which is what the tests compare against.
        """
        exclude = () if self.settings.align_axis is None else (self.settings.align_axis,)
        axes_memo = memo.axes if memo is not None else AxisFingerprintMemo()
        axes_fp = axes_memo.fingerprint(msg, exclude=exclude)

        attrs = msg.attrs
        if memo is not None and attrs is memo.attrs_obj:
            attrs_fp = memo.attrs_fp
        else:
            # _check_attr_type restricts merged attrs to str/int/float/bool, all
            # hashable, so the value goes in as itself -- repr() would be both
            # slower and, for anything numpy summarizes past 1000 elements,
            # unable to tell two different values apart. But this runs *before*
            # that validation, so an unsupported value must not raise here or it
            # would mask _check_attr_type's much clearer error.
            items = (attrs or {}).items()
            try:
                attrs_fp = frozenset((k, type(v).__name__, v) for k, v in items)
            except TypeError:
                attrs_fp = frozenset((k, type(v).__name__, repr(v)) for k, v in items)
            if memo is not None:
                memo.attrs_obj, memo.attrs_fp = attrs, attrs_fp

        linear_fp = _linear_axes_fingerprint(msg, msg.chunk_dim, exclude)
        # The chunk dimension's length is however much arrived, not a property of
        # the stream. Leaving it in rebuilt the cache on every chunk-size jitter
        # -- a deepcopy of every coordinate axis per message, which is the exact
        # cost `cached_axes` exists to avoid. Measured on a jittering source: 4
        # rebuilds in 4 messages before, 0 after.
        shape = msg.data.shape
        if msg.chunk_dim is not None and msg.chunk_dim in msg.dims:
            chunk_ix = msg.dims.index(msg.chunk_dim)
            shape = shape[:chunk_ix] + shape[chunk_ix + 1 :]
        return (tuple(msg.dims), shape, axes_fp, linear_fp, attrs_fp)

    def _rebuild_cache(self, a: AxisArray, b: AxisArray) -> None:
        concat_dim = self.settings.axis

        # Validate shared axes.
        _validate_shared_axes(
            a,
            b,
            concat_dim,
            align_dim=self.settings.align_axis,
            assert_flag=self.settings.assert_identical_shared_axes,
            chunk_dim=a.chunk_dim,
        )

        # Build merged concat axis.
        ax_a = a.axes.get(concat_dim)
        ax_b = b.axes.get(concat_dim)
        if ax_a is not None and ax_b is not None and hasattr(ax_a, "data") and hasattr(ax_b, "data"):
            self._state.merged_concat_axis = _build_merged_coordinate_axis(
                ax_a,
                ax_b,
                relabel=self.settings.relabel_axis,
                label_a=self.settings.label_a,
                label_b=self.settings.label_b,
            )
        elif concat_dim not in a.dims and concat_dim not in b.dims:
            self._state.merged_concat_axis = CoordinateAxis(
                data=np.asarray([self.settings.label_a, self.settings.label_b]),
                dims=[concat_dim],
            )
        else:
            self._state.merged_concat_axis = None

        # Merge .attrs across A and B. Equal-shared keys stay in attrs; differing
        # or partially-present keys are promoted to per-element fields on the
        # concat axis.
        equal_attrs, promote_a, promote_b = _classify_attrs(a.attrs, b.attrs)
        if promote_a or promote_b:
            if concat_dim in a.dims:
                n_a = a.data.shape[a.dims.index(concat_dim)]
            else:
                n_a = 1
            if concat_dim in b.dims:
                n_b = b.data.shape[b.dims.index(concat_dim)]
            else:
                n_b = 1
            ref_axis = ax_a if ax_a is not None else ax_b
            self._state.merged_concat_axis = _apply_promoted_attrs(
                self._state.merged_concat_axis,
                n_a,
                n_b,
                promote_a,
                promote_b,
                ref_axis,
                concat_dim,
            )
        self._state.merged_attrs = equal_attrs

        if self._state.merged_concat_axis is not None:
            with_fingerprint(self._state.merged_concat_axis)

        self._state.cached_axes = _build_cached_axes(
            a,
            concat_dim,
            align_dim=self.settings.align_axis,
            merged_concat_axis=self._state.merged_concat_axis,
            chunk_dim=a.chunk_dim,
        )


class Concat(ez.Unit):
    """Concatenate two AxisArray streams along an axis.

    Pairs messages by arrival order (FIFO). No time-alignment.
    """

    SETTINGS = ConcatSettings

    INPUT_SIGNAL_A = ez.InputStream(AxisArray)
    INPUT_SIGNAL_B = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self.processor = ConcatProcessor(self.SETTINGS)

    @ez.subscriber(INPUT_SIGNAL_A)
    async def on_a(self, msg: AxisArray) -> None:
        self.processor.push_a(msg)

    @ez.subscriber(INPUT_SIGNAL_B)
    async def on_b(self, msg: AxisArray) -> None:
        self.processor.push_b(msg)

    @ez.publisher(OUTPUT_SIGNAL)
    async def output(self) -> typing.AsyncGenerator:
        while True:
            yield self.OUTPUT_SIGNAL, await self.processor.__acall__()
