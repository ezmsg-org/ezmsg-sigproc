"""AxisArray support for .buffer.HybridBuffer."""

import math
import typing

import numpy as np
from array_api_compat import get_namespace
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, LinearAxis
from ezmsg.util.messages.util import replace

from .buffer import HybridBuffer

Array = typing.TypeVar("Array")


class HybridAxisBuffer:
    """
    A buffer that intelligently handles ezmsg.util.messages.AxisArray _axes_ objects.
     LinearAxis is maintained internally by tracking its offset, gain, and the number
     of samples that have passed through.
     CoordinateAxis has its data values maintained in a `HybridBuffer`.

    Args:
        duration: The desired duration of the buffer in seconds. This is non-limiting
         when managing a LinearAxis.
        **kwargs: Additional keyword arguments to pass to the underlying HybridBuffer
            (e.g., `update_strategy`, `threshold`, `overflow_strategy`, `max_size`).
    """

    _coords_buffer: HybridBuffer | None
    _coords_template: CoordinateAxis | None
    _coords_gain_estimate: float | None = None
    _linear_axis: LinearAxis | None
    _linear_n_available: int

    def __init__(self, duration: float, **kwargs):
        self.duration = duration
        self.buffer_kwargs = kwargs
        # Delay initialization until the first message arrives
        self._coords_buffer = None
        self._coords_template = None
        self._linear_axis = None
        self._linear_n_available = 0

    @property
    def capacity(self) -> int:
        """The maximum number of samples that can be stored in the buffer."""
        if self._coords_buffer is not None:
            return self._coords_buffer.capacity
        elif self._linear_axis is not None:
            return int(math.ceil(self.duration / self._linear_axis.gain))
        else:
            return 0

    def available(self) -> int:
        if self._coords_buffer is None:
            return self._linear_n_available
        return self._coords_buffer.available()

    def is_empty(self) -> bool:
        return self.available() == 0

    def is_full(self) -> bool:
        if self._coords_buffer is not None:
            return self._coords_buffer.is_full()
        return 0 < self.capacity == self.available()

    def _initialize(self, first_axis: LinearAxis | CoordinateAxis) -> None:
        if hasattr(first_axis, "data"):
            # Initialize a CoordinateAxis buffer
            if len(first_axis.data) > 1:
                _axis_gain = (first_axis.data[-1] - first_axis.data[0]) / (len(first_axis.data) - 1)
            else:
                _axis_gain = 1.0
            self._coords_gain_estimate = _axis_gain
            capacity = int(self.duration / _axis_gain)
            self._coords_buffer = HybridBuffer(
                get_namespace(first_axis.data),
                capacity,
                other_shape=(),
                dtype=first_axis.data.dtype,
                **self.buffer_kwargs,
            )
            self._coords_template = replace(first_axis, data=first_axis.data[:0].copy())
        else:
            # Initialize a LinearAxis buffer
            self._linear_axis = replace(first_axis, offset=first_axis.offset)
            self._linear_n_available = 0

    def write(self, axis: LinearAxis | CoordinateAxis, n_samples: int) -> None:
        if self._linear_axis is None and self._coords_buffer is None:
            self._initialize(axis)

        if self._coords_buffer is not None:
            if axis.__class__ is not self._coords_template.__class__:
                raise TypeError(
                    f"Buffer initialized with {self._coords_template.__class__.__name__}, "
                    f"but received {axis.__class__.__name__}."
                )
            self._coords_buffer.write(axis.data)
        else:
            if axis.__class__ is not self._linear_axis.__class__:
                raise TypeError(
                    f"Buffer initialized with {self._linear_axis.__class__.__name__}, "
                    f"but received {axis.__class__.__name__}."
                )
            if axis.gain != self._linear_axis.gain:
                raise ValueError(
                    f"Buffer initialized with gain={self._linear_axis.gain}, but received gain={axis.gain}."
                )
            if self._linear_n_available + n_samples > self.capacity:
                # Simulate overflow by advancing the offset and decreasing
                # the number of available samples.
                n_to_discard = self._linear_n_available + n_samples - self.capacity
                self.seek(n_to_discard)
            # Update the offset corresponding to the oldest sample in the buffer
            #  by anchoring on the new offset and accounting for the samples already available.
            self._linear_axis.offset = axis.offset - self._linear_n_available * axis.gain
            self._linear_n_available += n_samples

    def peek(self, n_samples: int | None = None) -> LinearAxis | CoordinateAxis:
        if self._coords_buffer is not None:
            return replace(self._coords_template, data=self._coords_buffer.peek(n_samples))
        else:
            # Return a shallow copy.
            return replace(self._linear_axis, offset=self._linear_axis.offset)

    def seek(self, n_samples: int) -> int:
        if self._coords_buffer is not None:
            return self._coords_buffer.seek(n_samples)
        else:
            n_to_seek = min(n_samples, self._linear_n_available)
            self._linear_n_available -= n_to_seek
            self._linear_axis.offset += n_to_seek * self._linear_axis.gain
            return n_to_seek

    def prune(self, n_samples: int) -> int:
        """Discards all but the last n_samples from the buffer."""
        n_to_discard = self.available() - n_samples
        if n_to_discard <= 0:
            return 0
        return self.seek(n_to_discard)

    @property
    def final_value(self) -> float | None:
        """
        The axis-value (timestamp, typically) of the last sample in the buffer.
        This does not advance the read head.
        """
        if self._coords_buffer is not None:
            return self._coords_buffer.peek_last()[0]
        elif self._linear_axis is not None:
            return self._linear_axis.value(self._linear_n_available - 1)
        else:
            return None

    @property
    def first_value(self) -> float | None:
        """
        The axis-value (timestamp, typically) of the first sample in the buffer.
        This does not advance the read head.
        """
        if self.available() == 0:
            return None
        if self._coords_buffer is not None:
            return self._coords_buffer.peek_at(0)[0]
        elif self._linear_axis is not None:
            return self._linear_axis.value(0)
        else:
            return None

    @property
    def gain(self) -> float | None:
        if self._coords_buffer is not None:
            return self._coords_gain_estimate
        elif self._linear_axis is not None:
            return self._linear_axis.gain
        else:
            return None

    def searchsorted(self, values: typing.Union[float, Array], side: str = "left") -> typing.Union[int, Array]:
        if self._coords_buffer is not None:
            return self._coords_buffer.xp.searchsorted(self._coords_buffer.peek(self.available()), values, side=side)
        else:
            if self.available() == 0:
                if isinstance(values, float):
                    return 0
                else:
                    _xp = get_namespace(values)
                    return _xp.zeros_like(values, dtype=int)

            f_inds = (values - self._linear_axis.offset) / self._linear_axis.gain
            res = np.ceil(f_inds)
            if side == "right":
                res[np.isclose(f_inds, res)] += 1
            return res.astype(int)


class HybridAxisArrayBuffer:
    """A buffer that intelligently handles ezmsg.util.messages.AxisArray objects.

    This buffer defers its own initialization until the first message arrives,
    allowing it to automatically configure its size, shape, dtype, and array backend
    (e.g., NumPy, CuPy) based on the message content and a desired buffer duration.

    Args:
        duration: The desired duration of the buffer in seconds.
        axis: The name of the axis to buffer along.
        **kwargs: Additional keyword arguments to pass to the underlying HybridBuffer
            (e.g., `update_strategy`, `threshold`, `overflow_strategy`, `max_size`).
    """

    _data_buffer: HybridBuffer | None
    _axis_buffer: HybridAxisBuffer
    _template_msg: AxisArray | None

    def __init__(self, duration: float, axis: str = "time", **kwargs):
        self.duration = duration
        self._axis = axis
        self.buffer_kwargs = kwargs
        self._axis_buffer = HybridAxisBuffer(duration=duration, **kwargs)
        # Delay initialization until the first message arrives
        self._data_buffer = None
        self._template_msg = None

    def available(self) -> int:
        """The total number of unread samples currently available in the buffer."""
        if self._data_buffer is None:
            return 0
        return self._data_buffer.available()

    def is_empty(self) -> bool:
        return self.available() == 0

    def is_full(self) -> bool:
        return 0 < self._data_buffer.capacity == self.available()

    @property
    def axis_first_value(self) -> float | None:
        """The axis-value (timestamp, typically) of the first sample in the buffer."""
        return self._axis_buffer.first_value

    @property
    def axis_final_value(self) -> float | None:
        """The axis-value (timestamp, typically) of the last sample in the buffer."""
        return self._axis_buffer.final_value

    def _initialize(self, first_msg: AxisArray) -> None:
        # Create a template message that has everything except the data are length 0
        #  and the target axis is missing.
        self._template_msg = replace(
            first_msg,
            data=first_msg.data[:0],
            axes={k: v for k, v in first_msg.axes.items() if k != self._axis},
        )

        in_axis = first_msg.axes[self._axis]
        self._axis_buffer._initialize(in_axis)

        capacity = int(self.duration / self._axis_buffer.gain)
        self._data_buffer = HybridBuffer(
            get_namespace(first_msg.data),
            capacity,
            other_shape=first_msg.data.shape[1:],
            dtype=first_msg.data.dtype,
            **self.buffer_kwargs,
        )

    def write(self, msg: AxisArray) -> None:
        """Adds an AxisArray message to the buffer, initializing on the first call."""
        in_axis_idx = msg.get_axis_idx(self._axis)
        if in_axis_idx > 0:
            # This class assumes that the target axis is the first axis.
            # If it is not, we move it to the front.
            dims = list(msg.dims)
            dims.insert(0, dims.pop(in_axis_idx))
            _xp = get_namespace(msg.data)
            msg = replace(msg, data=_xp.moveaxis(msg.data, in_axis_idx, 0), dims=dims)

        if self._data_buffer is None:
            self._initialize(msg)

        self._data_buffer.write(msg.data)
        self._axis_buffer.write(msg.axes[self._axis], msg.shape[0])

    def peek(self, n_samples: int | None = None) -> AxisArray | None:
        """Retrieves the oldest unread data as a new AxisArray without advancing the read head."""

        if self._data_buffer is None:
            return None

        data_array = self._data_buffer.peek(n_samples)

        if data_array is None:
            return None

        out_axis = self._axis_buffer.peek(n_samples)

        return replace(
            self._template_msg,
            data=data_array,
            axes={**self._template_msg.axes, self._axis: out_axis},
        )

    def peek_axis(self, n_samples: int | None = None) -> LinearAxis | CoordinateAxis | None:
        """Retrieves the axis data without advancing the read head."""
        if self._data_buffer is None:
            return None

        out_axis = self._axis_buffer.peek(n_samples)

        if out_axis is None:
            return None

        return out_axis

    def seek(self, n_samples: int) -> int:
        """Advances the read pointer by n_samples."""
        if self._data_buffer is None:
            return 0

        skipped_data_count = self._data_buffer.seek(n_samples)
        axis_skipped = self._axis_buffer.seek(skipped_data_count)
        assert (
            axis_skipped == skipped_data_count
        ), f"Axis buffer skipped {axis_skipped} samples, but data buffer skipped {skipped_data_count}."

        return skipped_data_count

    def read(self, n_samples: int | None = None) -> AxisArray | None:
        """Retrieves the oldest unread data as a new AxisArray and advances the read head."""
        retrieved_axis_array = self.peek(n_samples)

        if retrieved_axis_array is None or retrieved_axis_array.shape[0] == 0:
            return None

        self.seek(retrieved_axis_array.shape[0])

        return retrieved_axis_array

    def prune(self, n_samples: int) -> int:
        """Discards all but the last n_samples from the buffer."""
        if self._data_buffer is None:
            return 0

        n_to_discard = self.available() - n_samples
        if n_to_discard <= 0:
            return 0

        return self.seek(n_to_discard)

    @property
    def axis_gain(self) -> float | None:
        """
        The gain of the target axis, which is the time step between samples.
        This is typically the sampling rate (e.g., 1 / fs).
        """
        return self._axis_buffer.gain

    def axis_searchsorted(self, values: typing.Union[float, Array], side: str = "left") -> typing.Union[int, Array]:
        """
        Find the indices into which the given values would be inserted
        into the target axis data to maintain order.
        """
        return self._axis_buffer.searchsorted(values, side=side)
