"""Concatenate two AxisArray streams along an existing or new axis."""

from __future__ import annotations

import asyncio
import typing
from dataclasses import dataclass, field

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, AxisBase, CoordinateAxis
from ezmsg.util.messages.util import replace

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
    """Suffix appended to signal A labels when relabel_axis is True."""

    label_b: str = "_b"
    """Suffix appended to signal B labels when relabel_axis is True."""

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

        # expand_dims for new-axis concatenation.
        if new_axis:
            a = replace(a, data=np.expand_dims(a.data, axis=-1), dims=[*a.dims, concat_dim])
            b = replace(b, data=np.expand_dims(b.data, axis=-1), dims=[*b.dims, concat_dim])

        concat_idx = a.dims.index(concat_dim)
        data = np.concatenate([a.data, b.data], axis=concat_idx)

        # Build axes: use cached axes + live alignment axis from a.
        axes = dict(self._state.cached_axes) if self._state.cached_axes is not None else dict(a.axes)
        # Re-insert any axis that changes per-message (e.g. time offset).
        for name, ax in a.axes.items():
            if name not in axes:
                axes[name] = ax

        key = self.settings.new_key if self.settings.new_key is not None else a.key
        return AxisArray(data, dims=list(a.dims), axes=axes, key=key)

    def _fingerprint(self, msg: AxisArray) -> tuple:
        concat_dim = self.settings.axis
        ax = msg.axes.get(concat_dim)
        ax_hash = hash(ax.data.tobytes()) if ax is not None and hasattr(ax, "data") else None
        return (tuple(msg.dims), msg.data.shape, ax_hash)

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
        else:
            self._state.merged_concat_axis = None

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
