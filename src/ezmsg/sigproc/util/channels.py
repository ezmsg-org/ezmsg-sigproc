"""Helpers for deriving channel clusters from structured coordinate-axis metadata.

Several sources attach a structured ``CoordinateAxis`` to the channel dimension
carrying per-channel fields — e.g. ezmsg-blackrock's ``ChannelMap`` emits a
``ch`` axis with ``x``/``y``/``size``/``label``/``bank``/``elec``/``headstage``.
``channel_clusters_from_field`` turns one such field (``bank`` by default) into
the ``list[list[int]]`` cluster spec consumed by per-cluster operations like
common-average rereferencing and linear-regression rereferencing, so those units
can become "bank aware" by reading metadata that already rides on the stream.
"""

from __future__ import annotations

from ezmsg.util.messages.axisarray import AxisArray


def channel_clusters_from_field(
    message: AxisArray,
    axis: str | None = None,
    field: str = "bank",
) -> list[list[int]] | None:
    """Group channel indices by a field of a structured coordinate axis.

    Args:
        message: Message whose ``axis`` coordinate is a structured
            ``CoordinateAxis`` (its ``.data`` is a structured numpy array).
        axis: Channel axis name. ``None`` defaults to the last dimension.
        field: Structured-array field to group by (e.g. ``"bank"``).

    Returns:
        Clusters as a list of index lists, one per distinct ``field`` value in
        first-appearance order. Returns ``None`` when the axis carries no usable
        structured ``field`` (no such axis, no ``.data``, unstructured ``.data``,
        the field is absent, or the per-channel length doesn't match the data).
        Returning ``None`` rather than a single all-channel cluster lets callers
        distinguish "no metadata, fall back to my default" from "one bank".
    """
    axis = axis or message.dims[-1]
    ax = message.axes.get(axis)
    if ax is None or not hasattr(ax, "data"):
        return None

    data = ax.data
    names = getattr(getattr(data, "dtype", None), "names", None)
    if not names or field not in names:
        return None

    axis_idx = message.get_axis_idx(axis)
    n = message.data.shape[axis_idx]
    if data.shape[0] != n:
        return None

    values = data[field]
    clusters: dict[object, list[int]] = {}
    order: list[object] = []
    for i in range(n):
        v = values[i]
        key = v.item() if hasattr(v, "item") else v
        if key not in clusters:
            clusters[key] = []
            order.append(key)
        clusters[key].append(i)

    return [clusters[k] for k in order]
