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

This is the canonical Flatten in the ezmsg ecosystem.
``ezmsg.learn.process.flatten`` is a thin wrapper that post-processes
the merged-axis labels for time-lag windowing
(``"<ch>__t-<lag>"`` semantics).
"""

from __future__ import annotations

import math

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


def _axis_contribution(message: AxisArray, ax_name: str, ax_size: int) -> dict:
    """Per-axis contribution to the merged struct.

    Returns a dict with keys:
      * ``"fields"``: ordered tuple of (field_name, numpy dtype) pairs
        this axis contributes to the merged struct dtype.
      * ``"get_row(i)``: callable mapping a 0-based axis position to a
        ``{field_name: value}`` dict.
      * ``"primary(i)``: callable mapping a 0-based axis position to the
        string used for that axis's slot in the cartesian-product
        ``"label"`` join.
    """
    ax_obj = message.axes.get(ax_name)
    ax_data = getattr(ax_obj, "data", None) if ax_obj is not None else None

    if ax_data is not None and getattr(ax_data, "dtype", None) is not None and ax_data.dtype.names is not None:
        # Structured axis: contribute every field; primary label =
        # the "label" field if present, else fall back to a 1-based
        # index so the cartesian-product label stays human-readable.
        names = tuple(ax_data.dtype.names)
        fields = tuple((n, ax_data.dtype[n]) for n in names)
        has_label = "label" in names
        return {
            "fields": fields,
            "get_row": lambda i: {n: ax_data[i][n] for n in names},
            "primary": (lambda i: str(ax_data[i]["label"])) if has_label else (lambda i: str(i + 1)),
        }

    if ax_data is not None:
        # Simple CoordinateAxis: one virtual field named after the
        # axis whose value is the per-position label.
        labels = [str(x) for x in axis_labels(np.asarray(ax_data[:ax_size]))]
        max_len = max((len(s) for s in labels), default=1)
        dtype = np.dtype(f"U{max_len}")
        return {
            "fields": ((ax_name, dtype),),
            "get_row": lambda i: {ax_name: labels[i]},
            "primary": lambda i: labels[i],
        }

    # No CoordinateAxis at all: 1-based uint32 indices.
    dtype = np.dtype(np.uint32)
    return {
        "fields": ((ax_name, dtype),),
        "get_row": lambda i: {ax_name: np.uint32(i + 1)},
        "primary": lambda i: str(i + 1),
    }


def _wider_string_dtype(a: np.dtype, b: np.dtype) -> np.dtype:
    """Return the wider unicode dtype, or raise if not both unicode."""
    if a.kind != "U" or b.kind != "U":
        raise ValueError(
            f"Cannot merge incompatible dtypes {a!r} and {b!r} on shared "
            "struct field; widening is only defined for unicode strings."
        )
    return np.dtype(f"U{max(a.itemsize, b.itemsize) // 4}")


def _resolve_field_dtype(name: str, dt_a: np.dtype, dt_b: np.dtype) -> np.dtype:
    """Resolve a shared field's dtype across two contributing axes."""
    if dt_a == dt_b:
        return dt_a
    return _wider_string_dtype(dt_a, dt_b)


def _build_merged_axis(
    message: AxisArray,
    flatten_axes: tuple[str, ...],
    flatten_sizes: tuple[int, ...],
    output_axis: str,
    separator: str,
) -> CoordinateAxis:
    """Build the merged-axis structured CoordinateAxis."""
    if not flatten_axes:
        # Degenerate case: nothing to flatten → empty struct with a
        # single uint32 ``label`` placeholder.
        dtype = np.dtype([("label", np.dtype("U1"))])
        return CoordinateAxis(data=np.zeros(1, dtype=dtype), dims=[output_axis])

    contribs = [_axis_contribution(message, ax_name, ax_size) for ax_name, ax_size in zip(flatten_axes, flatten_sizes)]

    # Build the merged dtype: union of all contributing fields, later
    # axes overwriting (widening) on conflict.  ``"label"`` is always
    # appended last and overrides any inherited label field.
    field_order: list[str] = []
    field_dtype: dict[str, np.dtype] = {}
    for c in contribs:
        for name, dt in c["fields"]:
            if name in field_dtype:
                field_dtype[name] = _resolve_field_dtype(name, field_dtype[name], dt)
            else:
                field_order.append(name)
                field_dtype[name] = dt

    # Cartesian product, slowest-changing axis first — mirrors the
    # C-order reshape so output row k describes data[:, k, ...].
    n_flat = int(math.prod(flatten_sizes))
    combos: list[tuple[int, ...]] = [tuple()]
    for ax_size in flatten_sizes:
        combos = [(*prefix, i) for prefix in combos for i in range(ax_size)]
    assert len(combos) == n_flat

    label_column = [
        separator.join(c["primary"](pos) for c, pos in zip(contribs, combo))
        if separator
        else "".join(c["primary"](pos) for c, pos in zip(contribs, combo))
        for combo in combos
    ]
    label_dtype = np.dtype(f"U{max((len(s) for s in label_column), default=1)}")
    if "label" in field_dtype:
        field_dtype["label"] = _resolve_field_dtype("label", field_dtype["label"], label_dtype)
    else:
        field_order.append("label")
        field_dtype["label"] = label_dtype

    struct_dtype = np.dtype([(n, field_dtype[n]) for n in field_order])
    struct_data = np.zeros(n_flat, dtype=struct_dtype)

    for row_idx, combo in enumerate(combos):
        # Apply per-axis contributions left-to-right; later axes
        # overwrite shared fields (dict-union with later-wins).
        for c, pos in zip(contribs, combo):
            for name, value in c["get_row"](pos).items():
                struct_data[row_idx][name] = value
        # The canonical label always wins.
        struct_data[row_idx]["label"] = label_column[row_idx]

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
            raise ValueError(f"preserve_axis {preserve_axis!r} cannot also be in " f"flatten_axes {flatten_axes}")

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
            if st.sample_axis != st.preserve_axis and hasattr(preserve_ax, "dims"):
                preserve_ax = replace(preserve_ax, dims=[st.sample_axis])
            axes[st.sample_axis] = preserve_ax
        for ax in st.rest_axes:
            entry = message.axes.get(ax)
            if entry is not None:
                axes[ax] = entry

        return replace(message, data=data, dims=list(st.output_dims), axes=axes)


class Flatten(BaseTransformerUnit[FlattenSettings, AxisArray, AxisArray, FlattenTransformer]):
    SETTINGS = FlattenSettings
