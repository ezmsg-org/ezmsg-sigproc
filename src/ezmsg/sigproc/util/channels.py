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
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Union

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

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
    axis = axis or message.dims[-1]
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
        groups = spec(message, axis or message.dims[-1])
    else:
        groups = spec

    if groups is None:
        return None
    axis = axis or message.dims[-1]
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
    ax = message.axes.get(axis or message.dims[-1])
    names = getattr(getattr(getattr(ax, "data", None), "dtype", None), "names", None)
    return (bool(names) and all(field in names for field in fields),)
