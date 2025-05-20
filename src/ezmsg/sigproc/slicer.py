import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
    AxisBase,
    replace,
)

from .base import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)

"""
Slicer:Select a subset of data along a particular axis.
"""


def parse_slice(
    s: str,
    axinfo: AxisArray.CoordinateAxis | None = None,
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
    - A comma-separated list of values and axinfo is provided and is a CoordinateAxis -> a tuple of ints

    Args:
        s: The string representation of the slice.
        axinfo: (Optional) If provided, and of type CoordinateAxis,
          and `s` is a comma-separated list of values, then the values
          in s will be checked against the values in axinfo.data.

    Returns:
        A tuple of slice objects and/or ints.
    """
    if s.lower() in ["", ":", "none"]:
        return (slice(None),)
    if "," not in s:
        parts = [part.strip() for part in s.split(":")]
        if len(parts) == 1:
            if (
                axinfo is not None
                and hasattr(axinfo, "data")
                and parts[0] in axinfo.data
            ):
                return tuple(np.where(axinfo.data == parts[0])[0])
            return (int(parts[0]),)
        return (slice(*(int(part.strip()) if part else None for part in parts)),)
    suplist = [parse_slice(_, axinfo=axinfo) for _ in s.split(",")]
    return tuple([item for sublist in suplist for item in sublist])


class SlicerSettings(ez.Settings):
    selection: str = ""
    """selection: See :obj:`ezmsg.sigproc.slicer.parse_slice` for details."""

    axis: str | None = None
    """The name of the axis to slice along. If None, the last axis is used."""


@processor_state
class SlicerState:
    slice_: slice | int | npt.NDArray | None = None
    new_axis: AxisBase | None = None
    b_change_dims: bool = False


class SlicerTransformer(
    BaseStatefulTransformer[SlicerSettings, AxisArray, AxisArray, SlicerState]
):
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
        _slices = parse_slice(self.settings.selection, message.axes.get(axis, None))
        if len(_slices) == 1:
            self._state.slice_ = _slices[0]
            self._state.b_change_dims = isinstance(self._state.slice_, int)
        else:
            indices = np.arange(message.data.shape[axis_idx])
            indices = np.hstack([indices[_] for _ in _slices])
            self._state.slice_ = np.s_[indices]

        # Create the output axis
        if (
            axis in message.axes
            and hasattr(message.axes[axis], "data")
            and len(message.axes[axis].data) > 0
        ):
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
            replace_kwargs["dims"] = [
                _ for dim_ix, _ in enumerate(message.dims) if dim_ix != axis_idx
            ]
            replace_kwargs["axes"] = {
                k: v for k, v in message.axes.items() if k != axis
            }
        elif self._state.new_axis is not None:
            replace_kwargs["axes"] = {
                k: (v if k != axis else self._state.new_axis)
                for k, v in message.axes.items()
            }

        return replace(
            message,
            data=slice_along_axis(message.data, self._state.slice_, axis_idx),
            **replace_kwargs,
        )


class Slicer(
    BaseTransformerUnit[SlicerSettings, AxisArray, AxisArray, SlicerTransformer]
):
    SETTINGS = SlicerSettings


def slicer(selection: str = "", axis: str | None = None) -> SlicerTransformer:
    """
    Slice along a particular axis.

    Args:
        selection: See :obj:`ezmsg.sigproc.slicer.parse_slice` for details.
        axis: The name of the axis to slice along. If None, the last axis is used.

    Returns:
        :obj:`SlicerTransformer`
    """
    return SlicerTransformer(SlicerSettings(selection=selection, axis=axis))
