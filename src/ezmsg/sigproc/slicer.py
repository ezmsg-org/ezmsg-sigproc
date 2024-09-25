from dataclasses import replace
import typing

import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.generator import consumer

from .base import GenAxisArray


"""
Slicer:Select a subset of data along a particular axis.
"""


def parse_slice(s: str) -> typing.Tuple[typing.Union[slice, int], ...]:
    """
    Parses a string representation of a slice and returns a tuple of slice objects.

    - "" -> slice(None, None, None)  (take all)
    - ":" -> slice(None, None, None)
    - '"none"` (case-insensitive) -> slice(None, None, None)
    - "{start}:{stop}" or {start}:{stop}:{step} -> slice(start, stop, step)
    - "5" (or any integer) -> (5,). Take only that item.
        applying this to a ndarray or AxisArray will drop the dimension.
    - A comma-separated list of the above -> a tuple of slices | ints

    Args:
        s: The string representation of the slice.

    Returns:
        A tuple of slice objects and/or ints.
    """
    if s.lower() in ["", ":", "none"]:
        return (slice(None),)
    if "," not in s:
        parts = [part.strip() for part in s.split(":")]
        if len(parts) == 1:
            return (int(parts[0]),)
        return (slice(*(int(part.strip()) if part else None for part in parts)),)
    suplist = [parse_slice(_) for _ in s.split(",")]
    return tuple([item for sublist in suplist for item in sublist])


@consumer
def slicer(
    selection: str = "", axis: typing.Optional[str] = None
) -> typing.Generator[AxisArray, AxisArray, None]:
    msg_out = AxisArray(np.array([]), dims=[""])

    # State variables
    _slice: typing.Optional[typing.Union[slice, npt.NDArray]] = None
    new_axis: typing.Optional[AxisArray.Axis] = None
    b_change_dims: bool = False  # If number of dimensions changes when slicing

    # Reset if input changes
    check_input = {
        "key": None,  # key change used as proxy for label change, which we don't check explicitly
        "len": None,
    }

    while True:
        msg_in: AxisArray = yield msg_out

        axis = axis or msg_in.dims[-1]
        axis_idx = msg_in.get_axis_idx(axis)

        b_reset = _slice is None  # or new_axis is None
        b_reset = b_reset or msg_in.key != check_input["key"]
        b_reset = b_reset or (
            (msg_in.data.shape[axis_idx] != check_input["len"])
            and (type(_slice) is np.ndarray)
        )
        if b_reset:
            check_input["key"] = msg_in.key
            check_input["len"] = msg_in.data.shape[axis_idx]
            new_axis = None  # Will hold updated metadata
            b_change_dims = False

            # Calculate the slice
            _slices = parse_slice(selection)
            if len(_slices) == 1:
                _slice = _slices[0]
                # Do we drop the sliced dimension?
                b_change_dims = isinstance(_slice, int)
            else:
                # Multiple slices, but this cannot be done in a single step, so we convert the slices
                #  to a discontinuous set of integer indexes.
                indices = np.arange(msg_in.data.shape[axis_idx])
                indices = np.hstack([indices[_] for _ in _slices])
                _slice = np.s_[indices]  # Integer scalar array

            # Create the output axis.
            if (
                axis in msg_in.axes
                and hasattr(msg_in.axes[axis], "labels")
                and len(msg_in.axes[axis].labels) > 0
            ):
                new_labels = msg_in.axes[axis].labels[_slice]
                new_axis = replace(msg_in.axes[axis], labels=new_labels)

        replace_kwargs = {}
        if b_change_dims:
            # Dropping the target axis
            replace_kwargs["dims"] = [
                _ for dim_ix, _ in enumerate(msg_in.dims) if dim_ix != axis_idx
            ]
            replace_kwargs["axes"] = {k: v for k, v in msg_in.axes.items() if k != axis}
        elif new_axis is not None:
            replace_kwargs["axes"] = {
                k: (v if k != axis else new_axis) for k, v in msg_in.axes.items()
            }
        msg_out = replace(
            msg_in,
            data=slice_along_axis(msg_in.data, _slice, axis_idx),
            **replace_kwargs,
        )


class SlicerSettings(ez.Settings):
    selection: str = ""
    axis: typing.Optional[str] = None


class Slicer(GenAxisArray):
    SETTINGS = SlicerSettings

    def construct_generator(self):
        self.STATE.gen = slicer(
            selection=self.SETTINGS.selection, axis=self.SETTINGS.axis
        )
