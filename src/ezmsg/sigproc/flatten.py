"""Flatten non-time dimensions of an AxisArray into a single axis.

The most common use is collapsing a ``(time, ch, feature)`` stream into
``(time, ch_x_feature)`` for transports that only support 2-D data (e.g.
LSL).

The merged axis is always emitted as a numpy *structured* CoordinateAxis
whose rows are the cartesian product of the source-axis entries.  Per
cartesian combination, the merge rule is **dict-union with later-wins on
conflicts**:

* Source axes with a structured CoordinateAxis contribute *all* their
  fields to the merged row (so an upstream ``ch`` axis carrying
  ``(x, y, label, bank, elec, device)`` propagates every field through).
* Source axes with a simple CoordinateAxis contribute one virtual field
  named after the axis (e.g. ``"feature"``) whose value is the axis
  label at that position.
* Source axes without a CoordinateAxis contribute one virtual uint32
  field with a 1-based position index.

A canonical ``"label"`` field is *always* present on the output struct.
Its value is the cartesian-product join of each source axis's primary
label (the axis's own ``"label"`` field if struct, else the axis value
or position index), separated by :attr:`FlattenSettings.label_separator`
(default ``"/"``).  This ``"label"`` overwrites any inherited label
field from the source axes — the original per-axis labels are still
available via the named fields (``ch``, ``feature``, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ezmsg.core as ez
import numpy as np
from array_api_compat import get_namespace
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, replace


def normalize_axis_label(label):
    """Return a hashable string-or-tuple representation of a coord label.

    Handles structured-dtype entries (with a ``"label"`` field, or any
    other named fields) and numpy scalar types.  Public so other ezmsg
    packages — notably :mod:`ezmsg.learn.process.flatten` — can share the
    same convention.
    """
    dtype_names = getattr(getattr(label, "dtype", None), "names", None)
    if dtype_names is not None:
        if "label" in dtype_names:
            return str(label["label"])
        return tuple((name, normalize_axis_label(label[name])) for name in dtype_names)

    if isinstance(label, np.generic):
        return label.item()

    try:
        hash(label)
        return label
    except TypeError:
        return str(label)


def axis_labels(axis_data) -> list:
    """Normalize a CoordinateAxis ``.data`` array to a list of labels."""
    return [normalize_axis_label(label) for label in axis_data]


class FlattenSettings(ez.Settings):
    """
    Settings for :obj:`Flatten`.
    """

    preserve_axis: str | None = None
    """Axis kept as the leading dim of the output (typically
        ``"time"``).  Defaults to ``message.dims[0]``."""

    sample_axis: str | None = None
    """Optional rename for ``preserve_axis`` on the output
        (e.g. set to ``"time"`` when the input's preserved dim is named
        ``"win"``).  Defaults to ``preserve_axis`` (no rename)."""

    flatten_axes: tuple[str, ...] | None = None
    """Tuple of axes to collapse, in fastest-varying-last
       order.  Defaults to *all non-preserve dims, in input order*, so a
       ``(time, ch, feature)`` input flattens with ``ch`` slow / ``feature``
       fast — matching numpy's C-order reshape."""

    output_axis: str = "ch"
    """Name of the merged axis on the output."""

    label_separator: str = "/"
    """Separator for cartesian-product labels in the canonical
        ``"label"`` field of the output struct.  Default ``"/"``.  Set
        to ``""`` to concatenate without a delimiter."""


@processor_state
class FlattenState:
    hash: int = -1

    preserve_axis: str = ""
    sample_axis: str = ""
    output_axis: str = ""
    flatten_axes: tuple[str, ...] = ()
    rest_axes: tuple[str, ...] = ()

    # None means input dims already match (preserve, *flatten, *rest).
    perm: tuple[int, ...] | None = None

    # Final shape after permute_dims + reshape: (n_preserve, n_flat, *rest_shape).
    target_shape: tuple[int, ...] = ()

    # Precomputed merged-axis CoordinateAxis (structured array).
    output_axis_obj: CoordinateAxis | None = None

    output_dims: tuple[str, ...] = ()


@dataclass
class _AxisContribution:
    """How one source axis fills the merged struct.

    ``rows`` is a structured array (length = source axis size)
    contributing this axis's fields to the merged dtype.  ``primary``
    is the per-position string column used in the canonical
    cartesian-product ``"label"`` join.
    """

    rows: np.ndarray
    primary: np.ndarray


def _axis_contribution(message: AxisArray, ax_name: str, ax_size: int) -> _AxisContribution:
    """Build the per-axis contribution to the merged struct."""
    ax_obj = message.axes.get(ax_name)
    ax_data = getattr(ax_obj, "data", None) if ax_obj is not None else None

    if ax_data is not None and getattr(ax_data, "dtype", None) is not None and ax_data.dtype.names is not None:
        # Structured axis: pass every field through.  Primary label =
        # the "label" field if present, else a 1-based index so the
        # cartesian-product label stays human-readable.
        rows = np.asarray(ax_data)
        if "label" in rows.dtype.names:
            primary = rows["label"].astype(str)
        else:
            primary = np.arange(1, ax_size + 1).astype(str)
        return _AxisContribution(rows=rows, primary=primary)

    if ax_data is not None:
        # Simple CoordinateAxis: one virtual field named after the axis.
        str_labels = [str(x) for x in axis_labels(np.asarray(ax_data)[:ax_size])]
        max_len = max((len(s) for s in str_labels), default=1)
        labels = np.asarray(str_labels, dtype=f"U{max_len}")
        rows = np.empty(ax_size, dtype=[(ax_name, labels.dtype)])
        rows[ax_name] = labels
        return _AxisContribution(rows=rows, primary=labels)

    # No CoordinateAxis: 1-based uint32 positions.
    positions = np.arange(1, ax_size + 1, dtype=np.uint32)
    rows = np.empty(ax_size, dtype=[(ax_name, np.uint32)])
    rows[ax_name] = positions
    return _AxisContribution(rows=rows, primary=positions.astype(str))


def _merge_field_dtype(name: str, a: np.dtype, b: np.dtype) -> np.dtype:
    """Resolve a shared field's dtype across two contributing axes.

    Equal dtypes pass through; unicode pairs widen to the larger
    width; anything else raises — silently coercing mismatched
    numeric/string fields would corrupt values.
    """
    if a == b:
        return a
    if a.kind == "U" and b.kind == "U":
        return np.dtype(f"U{max(a.itemsize, b.itemsize) // 4}")
    raise ValueError(
        f"Cannot merge incompatible dtypes {a!r} and {b!r} on shared "
        f"struct field {name!r}; widening is only defined for unicode strings."
    )


def _build_merged_axis(
    message: AxisArray,
    flatten_axes: tuple[str, ...],
    flatten_sizes: tuple[int, ...],
    output_axis: str,
    separator: str,
) -> CoordinateAxis:
    """Build the merged-axis structured CoordinateAxis."""
    if not flatten_axes:
        # Degenerate: nothing to flatten → one row, single canonical field.
        dtype = np.dtype([("label", "U1")])
        return CoordinateAxis(data=np.zeros(1, dtype=dtype), dims=[output_axis])

    contribs = [_axis_contribution(message, name, size) for name, size in zip(flatten_axes, flatten_sizes)]
    n_flat = int(math.prod(flatten_sizes))

    # Expand each axis's per-position arrays to the cartesian-product
    # length via tile+repeat — slowest-changing first mirrors numpy's
    # C-order reshape so output row k describes data[:, k, ...].
    def _expand(arr: np.ndarray, axis_idx: int) -> np.ndarray:
        inner = math.prod(flatten_sizes[axis_idx + 1 :])
        outer = math.prod(flatten_sizes[:axis_idx])
        return np.tile(np.repeat(arr, inner), outer)

    expanded_rows = [_expand(c.rows, i) for i, c in enumerate(contribs)]
    expanded_primary = [_expand(c.primary, i) for i, c in enumerate(contribs)]

    # Canonical "label" column: cartesian-product join of primaries.
    label_column = np.asarray([separator.join(parts) for parts in zip(*expanded_primary)])

    # Merge struct dtype: dict-union with later-wins (widening shared
    # string fields).  ``"label"`` is always overridden by the join
    # column, so source-struct labels survive only via their named
    # fields (``ch``, etc.).
    fields: dict[str, np.dtype] = {}
    for c in contribs:
        for name in c.rows.dtype.names:
            dt = c.rows.dtype[name]
            fields[name] = _merge_field_dtype(name, fields[name], dt) if name in fields else dt
    fields["label"] = (
        _merge_field_dtype("label", fields["label"], label_column.dtype) if "label" in fields else label_column.dtype
    )

    struct_data = np.zeros(n_flat, dtype=np.dtype(list(fields.items())))
    for c, rows in zip(contribs, expanded_rows):
        for name in c.rows.dtype.names:
            struct_data[name] = rows[name]
    struct_data["label"] = label_column

    return CoordinateAxis(data=struct_data, dims=[output_axis])


class FlattenTransformer(BaseStatefulTransformer[FlattenSettings, AxisArray, AxisArray, FlattenState]):
    def _hash_message(self, message: AxisArray) -> int:
        return hash((tuple(message.dims), tuple(message.data.shape)))

    def _reset_state(self, message: AxisArray) -> None:
        preserve_axis = self.settings.preserve_axis or message.dims[0]
        if preserve_axis not in message.dims:
            raise ValueError(f"preserve_axis {preserve_axis!r} not found in dims {message.dims}")

        sample_axis = self.settings.sample_axis or preserve_axis

        flatten_axes = self.settings.flatten_axes
        if flatten_axes is None:
            flatten_axes = tuple(d for d in message.dims if d != preserve_axis)
        for ax in flatten_axes:
            if ax not in message.dims:
                raise ValueError(f"flatten_axes entry {ax!r} not found in dims {message.dims}")
        if preserve_axis in flatten_axes:
            raise ValueError(f"preserve_axis {preserve_axis!r} cannot also be in flatten_axes {flatten_axes}")

        output_axis = self.settings.output_axis
        if output_axis == sample_axis:
            raise ValueError(f"sample_axis and output_axis must differ; both are {sample_axis!r}")

        rest_axes = tuple(d for d in message.dims if d != preserve_axis and d not in flatten_axes)
        target_order = (preserve_axis, *flatten_axes, *rest_axes)
        if target_order != tuple(message.dims):
            perm: tuple[int, ...] | None = tuple(message.dims.index(d) for d in target_order)
            permuted_shape = tuple(message.data.shape[i] for i in perm)
        else:
            perm = None
            permuted_shape = tuple(message.data.shape)

        n_preserve = permuted_shape[0]
        flatten_sizes = permuted_shape[1 : 1 + len(flatten_axes)]
        n_flat = int(math.prod(flatten_sizes)) if flatten_sizes else 1
        rest_shape = permuted_shape[1 + len(flatten_axes) :]
        target_shape = (n_preserve, n_flat, *rest_shape)

        output_axis_obj = _build_merged_axis(
            message,
            flatten_axes,
            flatten_sizes,
            output_axis,
            self.settings.label_separator,
        )

        st = self._state
        st.preserve_axis = preserve_axis
        st.sample_axis = sample_axis
        st.output_axis = output_axis
        st.flatten_axes = flatten_axes
        st.rest_axes = rest_axes
        st.perm = perm
        st.target_shape = target_shape
        st.output_axis_obj = output_axis_obj
        st.output_dims = (sample_axis, output_axis, *rest_axes)

    def _process(self, message: AxisArray) -> AxisArray:
        st = self._state
        xp = get_namespace(message.data)

        if st.perm is not None:
            data = xp.permute_dims(message.data, st.perm)
        else:
            data = message.data
        data = xp.reshape(data, st.target_shape)

        # Carry the live preserve axis through (its gain/offset/data may
        # advance per message).  Rename to sample_axis on the output if
        # they differ.
        axes: dict = {st.output_axis: st.output_axis_obj}
        preserve_ax = message.axes.get(st.preserve_axis)
        if preserve_ax is not None:
            if st.sample_axis != st.preserve_axis and isinstance(preserve_ax, CoordinateAxis):
                preserve_ax = replace(preserve_ax, dims=[st.sample_axis])
            axes[st.sample_axis] = preserve_ax
        for ax in st.rest_axes:
            entry = message.axes.get(ax)
            if entry is not None:
                axes[ax] = entry

        return replace(message, data=data, dims=list(st.output_dims), axes=axes)


class Flatten(BaseTransformerUnit[FlattenSettings, AxisArray, AxisArray, FlattenTransformer]):
    SETTINGS = FlattenSettings
