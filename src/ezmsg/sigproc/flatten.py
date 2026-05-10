"""Flatten non-time dimensions of an AxisArray into a single axis.

The most common use is collapsing a ``(time, ch, feature)`` stream into
``(time, ch_x_feature)`` for transports that only support 2-D data (e.g.
LSL).  When both flatten axes carry CoordinateAxis labels the output gets
a cartesian-product CoordinateAxis like
``["ch1-spk", "ch1-sbp", "ch2-spk", ...]`` whose ordering matches numpy's
C-order reshape — so consumers downstream can recover ``(channel, feature)``
identity from the labels alone.

This is the canonical Flatten in the ezmsg ecosystem.
``ezmsg.learn.process.flatten`` is a thin wrapper around it that adds
time-lag-windowing label semantics (``"<ch>__t-<lag>"``).
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


def _coord_labels(axis) -> list[str] | None:
    """Return string labels from a CoordinateAxis, or None if unavailable."""
    data = getattr(axis, "data", None)
    if data is None:
        return None
    return [str(x) for x in axis_labels(np.asarray(data))]


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
       fast — matching numpy's C-order reshape and the LSL convention of
       interleaving features within each channel."""

    output_axis: str = "ch"
    """Name of the merged axis on the output (default
       ``"ch"`` so that downstream LSL outlets see a familiar dim name)."""

    label_separator: str = "-"
    """Separator for cartesian-product labels.  Default
        ``"-"``.  Set to ``""`` to concatenate without a delimiter."""

    flat_labels: tuple[str, ...] | None = None
    """Optional caller-supplied labels for the merged axis,
       overriding the auto-generated cartesian product.  Length must
       equal ``prod(flattened axis sizes)``.  Used by
       :mod:`ezmsg.learn.process.flatten` to inject time-lag-style
       labels (``"spk__t-1"``)."""


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

    # Precomputed merged-axis CoordinateAxis (cartesian product / arange / flat_labels).
    output_axis_obj: CoordinateAxis | None = None

    output_dims: tuple[str, ...] = ()


class FlattenTransformer(BaseStatefulTransformer[FlattenSettings, AxisArray, AxisArray, FlattenState]):
    def _hash_message(self, message: AxisArray) -> int:
        return hash((tuple(message.dims), tuple(message.data.shape)))

    def _reset_state(self, message: AxisArray) -> None:
        preserve_axis = self.settings.preserve_axis or message.dims[0]
        if preserve_axis not in message.dims:
            raise ValueError(f"preserve_axis {preserve_axis!r} not found in dims " f"{message.dims}")

        sample_axis = self.settings.sample_axis or preserve_axis

        flatten_axes = self.settings.flatten_axes
        if flatten_axes is None:
            flatten_axes = tuple(d for d in message.dims if d != preserve_axis)
        for ax in flatten_axes:
            if ax not in message.dims:
                raise ValueError(f"flatten_axes entry {ax!r} not found in dims " f"{message.dims}")
        if preserve_axis in flatten_axes:
            raise ValueError(f"preserve_axis {preserve_axis!r} cannot also be in " f"flatten_axes {flatten_axes}")

        output_axis = self.settings.output_axis
        if output_axis == sample_axis:
            raise ValueError(f"sample_axis and output_axis must differ; both are " f"{sample_axis!r}")

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

        # Build the merged-axis CoordinateAxis once.  Precedence:
        #   1. ``flat_labels`` override — the caller fully specifies the
        #      labels (used by the time-lag-aware learn-side wrapper).
        #   2. Cartesian product of input axis labels when *every* flatten
        #      axis carries CoordinateAxis labels matching its size.
        #   3. Integer positions ``np.arange(n_flat)`` as a last resort.
        if self.settings.flat_labels is not None:
            if len(self.settings.flat_labels) != n_flat:
                raise ValueError(
                    f"flat_labels has length {len(self.settings.flat_labels)} " f"but flattened axis size is {n_flat}"
                )
            output_axis_obj = CoordinateAxis(
                data=np.asarray(self.settings.flat_labels),
                dims=[output_axis],
            )
        else:
            per_axis_labels: list[list[str] | None] = [_coord_labels(message.axes.get(ax)) for ax in flatten_axes]
            all_labeled = all(
                labels is not None and len(labels) == size for labels, size in zip(per_axis_labels, flatten_sizes)
            )
            if all_labeled and flatten_axes:
                sep = self.settings.label_separator
                # Cartesian product, slowest-changing first — matches the
                # C-order reshape so labels[k] describes data[:, k, ...].
                combined = [""]
                for labels in per_axis_labels:
                    assert labels is not None  # guarded by all_labeled
                    combined = [f"{c}{sep}{lbl}" if c else lbl for c in combined for lbl in labels]
                output_axis_obj = CoordinateAxis(data=np.asarray(combined), dims=[output_axis])
            else:
                output_axis_obj = CoordinateAxis(data=np.arange(n_flat), dims=[output_axis])

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
