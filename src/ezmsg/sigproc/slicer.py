"""Select a subset of data along a named axis using slice notation."""

import re

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import (
    AxisArray,
    AxisBase,
    replace,
    slice_along_axis,
)

"""
Slicer:Select a subset of data along a particular axis.
"""


def _axis_labels(
    axinfo: AxisArray.CoordinateAxis | None,
    field: str | None = None,
) -> npt.NDArray | None:
    """Return the per-index label values of a coordinate axis, or None if it has none.

    A structured ``.data`` (e.g. ezmsg-blackrock's ChannelMap ``ch`` axis, with
    fields like ``x``/``y``/``label``/``bank``) contributes one of its fields,
    stringified so selection tokens can match non-string fields like ``bank``.
    Comparing a structured array against a string raises a TypeError in numpy,
    so it must never be matched against directly.

    ``field=None`` means "the ``label`` field, if present" and degrades to None
    (no usable labels) otherwise. An explicit ``field`` is a user commitment:
    a missing field or non-structured axis raises instead of silently falling
    back to positional indexing.
    """
    data = getattr(axinfo, "data", None)
    names = getattr(getattr(data, "dtype", None), "names", None)
    if field is not None:
        if names is None:
            raise ValueError(
                f"Selection field {field!r} requires the target axis to have structured "
                f"coordinate data, but it has {'no coordinate data' if data is None else 'unstructured data'}."
            )
        if field not in names:
            raise ValueError(f"Axis data has no field {field!r}; available fields: {list(names)}.")
        return data[field].astype(str)
    if names is not None:
        return data["label"].astype(str) if "label" in names else None
    return data


def parse_slice(
    s: str,
    axinfo: AxisArray.CoordinateAxis | None = None,
    field: str | None = None,
) -> tuple[slice | int, ...]:
    """
    Parses a string representation of a slice and returns a tuple of slice objects.

    - "" -> slice(None, None, None)  (take all)
    - ":" -> slice(None, None, None)
    - '"none"` (case-insensitive) -> slice(None, None, None)
    - "{start}:{stop}" or {start}:{stop}:{step} -> slice(start, stop, step)
    - "5" (or any integer) -> (5,). Take only that item.
        applying this to a ndarray or AxisArray will drop the dimension.
    - A comma-separated list of the above -> a tuple of slices | ints
    - A comma-separated list of values and axinfo is provided and is a CoordinateAxis -> a tuple of ints.
      Each value is first compared against the axis labels for an exact match; failing
      that, it is treated as a regular expression and full-matched against the labels
      (e.g. "C[34]" or "Ch.*"). Note: tokens containing ":" are parsed as slices, so
      regexes may not contain ":". If the axis ``.data`` is a structured array, its
      "label" field supplies the labels; a structured array without a "label" field
      supports only integer/slice selections.
    - If `field` is provided, tokens are matched against that field of the structured
      axis data (stringified, so numeric fields like "bank" work). This disables the
      bare-integer positional fallback: "3" means field value "3", not index 3.
      Positional selection remains available via slice syntax (e.g. "3:4").

    Args:
        s: The string representation of the slice.
        axinfo: (Optional) If provided, and of type CoordinateAxis,
          and `s` is a comma-separated list of values, then the values
          in s will be matched (exactly, then as regex) against the values in axinfo.data.
        field: (Optional) Which field of a structured `axinfo.data` to match tokens
          against. None uses the "label" field when present. An explicit field raises
          ValueError if the axis data is missing, unstructured, or lacks that field.

    Returns:
        A tuple of slice objects and/or ints.
    """
    if s.lower() in ["", ":", "none"]:
        return (slice(None),)
    if "," not in s:
        parts = [part.strip() for part in s.split(":")]
        if len(parts) == 1:
            labels = _axis_labels(axinfo, field=field)
            if labels is not None and parts[0] in labels:
                return tuple(np.where(labels == parts[0])[0])
            if field is None:
                try:
                    return (int(parts[0]),)
                except ValueError:
                    if labels is None:
                        raise
            pattern = re.compile(parts[0])
            hits = tuple(ix for ix, label in enumerate(labels) if pattern.fullmatch(str(label)))
            if hits:
                return hits
            raise ValueError(
                f"Selection {parts[0]!r} matched no "
                f"{'labels' if field is None else f'values in field {field!r}'} "
                f"on the target axis (neither exactly nor as a regex)."
            ) from None
        return (slice(*(int(part.strip()) if part else None for part in parts)),)
    suplist = [parse_slice(_, axinfo=axinfo, field=field) for _ in s.split(",")]
    return tuple([item for sublist in suplist for item in sublist])


class SlicerSettings(ez.Settings):
    selection: str = ""
    """selection: See :obj:`ezmsg.sigproc.slicer.parse_slice` for details."""

    axis: str | None = None
    """The name of the axis to slice along. If None, the last axis is used."""

    field: str | None = None
    """Which field of a structured coordinate axis to match selection values against
    (e.g. "bank", "elec"). If None, the "label" field is used when present. Setting
    this explicitly makes selection tokens mean field values only — bare integers are
    no longer positional indices (use slice syntax like "3:4" for positions) — and
    raises an error if the axis has no such field."""


@processor_state
class SlicerState:
    slice_: slice | int | npt.NDArray | None = None
    new_axis: AxisBase | None = None
    b_change_dims: bool = False


class SlicerTransformer(BaseStatefulTransformer[SlicerSettings, AxisArray, AxisArray, SlicerState]):
    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        return hash((message.key, message.data.shape[axis_idx]))

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        self._state.new_axis = None
        self._state.b_change_dims = False

        # Calculate the slice
        _slices = parse_slice(self.settings.selection, message.axes.get(axis, None), field=self.settings.field)
        if len(_slices) == 1:
            self._state.slice_ = _slices[0]
            self._state.b_change_dims = isinstance(self._state.slice_, int)
        else:
            indices = np.arange(message.data.shape[axis_idx])
            indices = np.hstack([indices[_] for _ in _slices])
            self._state.slice_ = np.s_[indices]

        # Create the output axis
        if axis in message.axes and hasattr(message.axes[axis], "data") and len(message.axes[axis].data) > 0:
            in_data = np.array(message.axes[axis].data)
            if self._state.b_change_dims:
                out_data = in_data[self._state.slice_ : self._state.slice_ + 1]
            else:
                out_data = in_data[self._state.slice_]
            self._state.new_axis = replace(message.axes[axis], data=out_data)

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)

        replace_kwargs = {}
        if self._state.b_change_dims:
            replace_kwargs["dims"] = [_ for dim_ix, _ in enumerate(message.dims) if dim_ix != axis_idx]
            replace_kwargs["axes"] = {k: v for k, v in message.axes.items() if k != axis}
        elif self._state.new_axis is not None:
            replace_kwargs["axes"] = {k: (v if k != axis else self._state.new_axis) for k, v in message.axes.items()}

        return replace(
            message,
            data=slice_along_axis(message.data, self._state.slice_, axis_idx),
            **replace_kwargs,
        )


class Slicer(BaseTransformerUnit[SlicerSettings, AxisArray, AxisArray, SlicerTransformer]):
    SETTINGS = SlicerSettings


def slicer(selection: str = "", axis: str | None = None, field: str | None = None) -> SlicerTransformer:
    """
    Slice along a particular axis.

    Args:
        selection: See :obj:`ezmsg.sigproc.slicer.parse_slice` for details.
        axis: The name of the axis to slice along. If None, the last axis is used.
        field: Which field of a structured coordinate axis to match selection values
          against. See :obj:`SlicerSettings` for details.

    Returns:
        :obj:`SlicerTransformer`
    """
    return SlicerTransformer(SlicerSettings(selection=selection, axis=axis, field=field))
