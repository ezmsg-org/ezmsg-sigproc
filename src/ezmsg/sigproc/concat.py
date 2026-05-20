"""Concatenate two AxisArray streams along an existing or new axis."""

from __future__ import annotations

import asyncio
import logging
import typing
from dataclasses import dataclass, field

import ezmsg.core as ez
import numpy as np
from array_api_compat import get_namespace
from ezmsg.util.messages.axisarray import AxisArray, AxisBase, CoordinateAxis
from ezmsg.util.messages.util import replace

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
) -> None:
    """Raise ValueError if shared CoordinateAxis .data arrays differ."""
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


def _build_cached_axes(
    a: AxisArray,
    concat_dim: str,
    align_dim: str | None,
    merged_concat_axis: CoordinateAxis | None,
) -> dict[str, AxisBase]:
    """Build the output axes dict (everything except the alignment axis)."""
    axes: dict[str, AxisBase] = {}
    for name, ax in a.axes.items():
        if name == align_dim:
            continue
        if name == concat_dim and merged_concat_axis is not None:
            axes[name] = merged_concat_axis
        else:
            axes[name] = ax
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
        fp_a = self._fingerprint(a)
        fp_b = self._fingerprint(b)
        if fp_a != self._state.a_fingerprint or fp_b != self._state.b_fingerprint:
            self._rebuild_cache(a, b)
            self._state.a_fingerprint = fp_a
            self._state.b_fingerprint = fp_b

        new_axis = concat_dim not in a.dims

        xp = get_namespace(a.data)

        # expand_dims for new-axis concatenation.
        if new_axis:
            a = replace(a, data=xp.expand_dims(a.data, axis=-1), dims=[*a.dims, concat_dim])
            b = replace(b, data=xp.expand_dims(b.data, axis=-1), dims=[*b.dims, concat_dim])

        concat_idx = a.dims.index(concat_dim)
        data = xp.concat([a.data, b.data], axis=concat_idx)

        # Build axes: use cached axes + live alignment axis from a.
        axes = dict(self._state.cached_axes) if self._state.cached_axes is not None else dict(a.axes)
        # Re-insert any axis that changes per-message (e.g. time offset).
        for name, ax in a.axes.items():
            if name not in axes:
                axes[name] = ax

        key = self.settings.new_key if self.settings.new_key is not None else a.key
        attrs = dict(self._state.merged_attrs) if self._state.merged_attrs else {}
        return AxisArray(data, dims=list(a.dims), axes=axes, key=key, attrs=attrs)

    def _fingerprint(self, msg: AxisArray) -> tuple:
        concat_dim = self.settings.axis
        ax = msg.axes.get(concat_dim)
        ax_hash = hash(ax.data.tobytes()) if ax is not None and hasattr(ax, "data") else None
        attrs_fp = frozenset((k, type(v).__name__, repr(v)) for k, v in (msg.attrs or {}).items())
        return (tuple(msg.dims), msg.data.shape, ax_hash, attrs_fp)

    def _rebuild_cache(self, a: AxisArray, b: AxisArray) -> None:
        concat_dim = self.settings.axis

        # Validate shared axes.
        _validate_shared_axes(
            a,
            b,
            concat_dim,
            align_dim=self.settings.align_axis,
            assert_flag=self.settings.assert_identical_shared_axes,
        )

        # New-axis validation: all other dims must match.
        if concat_dim not in a.dims or concat_dim not in b.dims:
            for i, (d, sa, sb) in enumerate(zip(a.dims, a.data.shape, b.data.shape)):
                if sa != sb:
                    raise ValueError(
                        f"Cannot concatenate along new axis {concat_dim!r}: "
                        f"dimension {d!r} has size {sa} in A but {sb} in B"
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

        self._state.cached_axes = _build_cached_axes(
            a,
            concat_dim,
            align_dim=self.settings.align_axis,
            merged_concat_axis=self._state.merged_concat_axis,
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
