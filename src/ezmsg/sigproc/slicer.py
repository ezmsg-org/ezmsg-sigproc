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
    allow_empty: bool = False,
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
        allow_empty: (Optional) If True, a label/regex token that matches nothing
          returns no indices instead of raising. In a comma-separated selection,
          non-matching tokens are dropped and the matching ones kept; if every
          token matches nothing the result is an empty tuple.

    Returns:
        A tuple of slice objects and/or ints. May be empty when allow_empty is
        True and nothing matched.
    """
    if s.lower() in ["", ":", "none"]:
        return (slice(None),)
    if "," not in s:
        parts = [part.strip() for part in s.split(":")]
        if len(parts) == 1:
            labels = _axis_labels(axinfo, field=field)
            if labels is not None and parts[0] in labels:
                # Cast to Python ints so the return type is tuple[slice | int, ...]
                # as documented (np.where yields np.int64).
                return tuple(int(ix) for ix in np.where(labels == parts[0])[0])
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
            if allow_empty:
                return ()
            raise ValueError(
                f"Selection {parts[0]!r} matched no "
                f"{'labels' if field is None else f'values in field {field!r}'} "
                f"on the target axis (neither exactly nor as a regex)."
            ) from None
        return (slice(*(int(part.strip()) if part else None for part in parts)),)
    suplist = [parse_slice(_, axinfo=axinfo, field=field, allow_empty=allow_empty) for _ in s.split(",")]
    return tuple([item for sublist in suplist for item in sublist])


class SlicerSettings(ez.Settings):
    selection: str = ""
    """selection: See :obj:`ezmsg.sigproc.slicer.parse_slice` for details.
    Label/regex selections always preserve the sliced axis — a single matching
    entry yields a length-1 axis. Only a bare-integer positional selection
    (e.g. "5") drops the dimension."""

    axis: str | None = None
    """The name of the axis to slice along. If None, the last axis is used."""

    field: str | None = None
    """Which field of a structured coordinate axis to match selection values against
    (e.g. "bank", "elec"). If None, the "label" field is used when present. Setting
    this explicitly makes selection tokens mean field values only — bare integers are
    no longer positional indices (use slice syntax like "3:4" for positions) — and
    raises an error if the axis has no such field."""

    on_empty: str = "warn"
    """What to do when a label/regex selection matches nothing on the target axis.

    - "warn" (default): non-matching tokens are dropped (logged at info level); if
      the whole selection matches nothing, the output is empty (0-length along
      ``axis``) and a warning is logged once per stream configuration. This lets a
      selection be broadcast to streams that may legitimately contain none of the
      selected entries (e.g. a per-source region selection where a given source
      carries none of the requested regions).
    - "raise": raise a ValueError when any token matches nothing, which catches
      typos and wrong-axis mistakes at first message."""


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

    def _selects_positional_int(self, axinfo: AxisArray.CoordinateAxis | None) -> bool:
        """True iff the selection is a single bare-integer token that parse_slice
        resolved positionally (its int() path) rather than via a label match."""
        sel = self.settings.selection.strip()
        if self.settings.field is not None or "," in sel or ":" in sel:
            return False
        labels = _axis_labels(axinfo)
        if labels is not None and sel in labels:
            return False
        try:
            int(sel)
        except ValueError:
            return False
        return True

    def _reset_state(self, message: AxisArray) -> None:
        if self.settings.on_empty not in ("raise", "warn"):
            raise ValueError(f"on_empty must be 'raise' or 'warn', got {self.settings.on_empty!r}")
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        axinfo = message.axes.get(axis, None)
        self._state.new_axis = None
        self._state.b_change_dims = False

        # Calculate the slice
        allow_empty = self.settings.on_empty == "warn"
        _slices = parse_slice(
            self.settings.selection,
            axinfo,
            field=self.settings.field,
            allow_empty=allow_empty,
        )

        if allow_empty:
            tokens = [t.strip() for t in self.settings.selection.split(",")]
            dropped = [t for t in tokens if parse_slice(t, axinfo, field=self.settings.field, allow_empty=True) == ()]
            if len(dropped) == len(tokens):
                ez.logger.warning(
                    "Slicer: selection %r matched no entries on axis %r; emitting "
                    "an empty (0-length) result (on_empty='warn').",
                    self.settings.selection,
                    axis,
                )
            elif dropped:
                ez.logger.info(
                    "Slicer: dropped non-matching selection tokens %r on axis %r (on_empty='warn').",
                    dropped,
                    axis,
                )

        # Only a bare-integer positional selection ("5") drops the dimension. A
        # label/regex selection resolving to a single entry takes the indices-array
        # path so the axis is preserved and output rank does not depend on how many
        # entries matched.
        if len(_slices) == 1 and (isinstance(_slices[0], slice) or self._selects_positional_int(axinfo)):
            self._state.slice_ = _slices[0]
            self._state.b_change_dims = isinstance(self._state.slice_, int)
        else:
            indices = np.arange(message.data.shape[axis_idx])
            # Empty _slices (nothing matched) -> select no entries (0-length),
            # rather than np.hstack([]) which would raise.
            indices = np.hstack([indices[_] for _ in _slices]) if _slices else indices[:0]
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


def slicer(
    selection: str = "",
    axis: str | None = None,
    field: str | None = None,
    on_empty: str = "warn",
) -> SlicerTransformer:
    """
    Slice along a particular axis.

    Args:
        selection: See :obj:`ezmsg.sigproc.slicer.parse_slice` for details.
        axis: The name of the axis to slice along. If None, the last axis is used.
        field: Which field of a structured coordinate axis to match selection values
          against. See :obj:`SlicerSettings` for details.
        on_empty: "warn" (default) or "raise" — what to do when a label/regex
          selection matches nothing. See :obj:`SlicerSettings` for details.

    Returns:
        :obj:`SlicerTransformer`
    """
    return SlicerTransformer(SlicerSettings(selection=selection, axis=axis, field=field, on_empty=on_empty))
