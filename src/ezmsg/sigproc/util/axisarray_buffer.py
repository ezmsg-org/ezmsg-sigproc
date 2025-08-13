import math
import time
import typing

from ezmsg.util.messages.axisarray import AxisArray, LinearAxis, CoordinateAxis
from ezmsg.util.messages.util import replace
from array_api_compat import get_namespace

from .buffer import HybridBuffer


Array = typing.TypeVar("Array")


class HybridAxisBuffer:
    """
    A buffer that intelligently handles ezmsg.util.messages.AxisArray _axes_ objects.
    LinearAxis and CoordinateAxis are supported.
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
        return self.capacity > 0 and self.available() == self.capacity

    def _initialize(self, first_axis: LinearAxis | CoordinateAxis) -> None:
        if hasattr(first_axis, "data"):
            # Initialize a CoordinateAxis buffer
            if len(first_axis.data) > 1:
                _axis_gain = (first_axis.data[-1] - first_axis.data[0]) / (
                    len(first_axis.data) - 1
                )
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
            self._coords_template = replace(first_axis, data=first_axis.data[:0])
        else:
            # Initialize a LinearAxis buffer
            self._linear_axis = first_axis
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
                    f"Buffer initialized with gain={self._linear_axis.gain}, "
                    f"but received gain={axis.gain}."
                )
            # Update the offset corresponding to the oldest sample in the buffer
            #  by anchoring on the new offset and accounting for the samples already available.
            self._linear_axis.offset = (
                axis.offset - self._linear_n_available * axis.gain
            )
            self._linear_n_available += n_samples

    def peek(self, n_samples: int | None = None) -> LinearAxis | CoordinateAxis:
        if self._coords_buffer is not None:
            return replace(
                self._coords_template, data=self._coords_buffer.peek(n_samples)
            )
        else:
            return self._linear_axis

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
            return self._coords_buffer.peek(self.available())[-1]
        elif self._linear_axis is not None:
            return self._linear_axis.value(self._linear_n_available - 1)
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

    def searchsorted(
        self, values: typing.Union[float, Array]
    ) -> typing.Union[int, Array]:
        if self._coords_buffer is not None:
            return self._coords_buffer.xp.searchsorted(
                self._coords_buffer.peek(self.available()), values, side="left"
            )
        else:
            if self.available() == 0:
                if isinstance(values, float):
                    return 0
                else:
                    _xp = get_namespace(values)
                    return _xp.zeros_like(values, dtype=int)

            # _linear_axis.index(values) uses np.rint
            return self._linear_axis.index(values)


class HybridAxisArrayBuffer:
    """A buffer that intelligently handles ezmsg.util.messages.AxisArray objects.

    This buffer defers its own initialization until the first message arrives,
    allowing it to automatically configure its size, shape, dtype, and array backend
    (e.g., NumPy, CuPy) based on the message content and a desired buffer duration.

    Args:
        duration: The desired duration of the buffer in seconds.
        **kwargs: Additional keyword arguments to pass to the underlying HybridBuffer
            (e.g., `update_strategy`, `threshold`).
    """

    _data_buffer: HybridBuffer | None
    _axis_offset: float
    _axis_buffer: HybridBuffer | None
    _template_msg: AxisArray | None
    _last_update: float | None

    def __init__(self, duration: float, axis: str = "time", **kwargs):
        self.duration = duration
        self._axis = axis
        self.buffer_kwargs = kwargs
        self._data_buffer = None
        self._axis_offset = 0.0
        self._axis_buffer = None
        self._template_msg = None
        self._last_update = None

    @property
    def n_unread(self) -> int:
        """The total number of unread samples currently available in the buffer."""
        if self._data_buffer is None:
            return 0
        return self._data_buffer.n_unread

    @property
    def last_update(self) -> float | None:
        """The timestamp of the last update to the buffer."""
        return self._last_update

    def add_message(self, msg: AxisArray) -> None:
        """Adds an AxisArray message to the buffer, initializing on the first call."""
        if self._data_buffer is None:
            self._initialize(msg)

        in_axis_idx = msg.get_axis_idx(self._axis)
        if in_axis_idx > 0:
            # TODO: Maybe we can support non-first axes in the future?
            raise ValueError(
                f"Axis '{self._axis}' must be the first axis in the message, "
                f"but found at index {in_axis_idx}."
            )
        in_axis = msg.axes[self._axis]
        if in_axis.__class__ is not self._template_msg.axes[self._axis].__class__:
            raise TypeError(
                f"Buffer initialized with {self._template_msg.axes[self._axis].__class__.__name__}, "
                f"but received message with {in_axis.__class__.__name__}."
            )

        self._data_buffer.add_message(msg.data)
        if hasattr(in_axis, "data"):
            self._axis_buffer.add_message(in_axis.data)
        else:
            self._axis_offset = in_axis.offset + (msg.shape[0] - 1) * in_axis.gain
        self._last_update = time.time()

    def _initialize(self, first_msg: AxisArray) -> None:
        in_axis_idx = first_msg.get_axis_idx(self._axis)
        if in_axis_idx > 0:
            # TODO: Maybe we can support non-first axes in the future?
            raise ValueError(
                f"Axis '{self._axis}' must be the first axis in the message, "
                f"but found at index {in_axis_idx}."
            )

        self._template_msg = replace(first_msg, data=first_msg.data[:0])

        in_axis = first_msg.axes[self._axis]
        if hasattr(in_axis, "data"):
            if len(in_axis.data) > 1:
                _axis_gain = (in_axis.data[-1] - in_axis.data[0]) / (
                    len(in_axis.data) - 1
                )
            else:
                # Assume 1 sample per second, thus our duration because 'number of samples'
                _axis_gain = 1.0
        else:
            _axis_gain = in_axis.gain

        maxlen = int(self.duration / _axis_gain)
        self._data_buffer = HybridBuffer(
            get_namespace(first_msg.data),
            maxlen,
            other_shape=first_msg.data.shape[1:],
            dtype=first_msg.data.dtype,
            **self.buffer_kwargs,
        )

        if hasattr(in_axis, "data"):
            self._axis_buffer = HybridBuffer(
                get_namespace(in_axis.data),
                maxlen,
                other_shape=(),
                dtype=in_axis.data.dtype,
                **self.buffer_kwargs,
            )

    def peek(self, n_samples: int | None = None) -> AxisArray | None:
        """Retrieves the oldest unread data as a new AxisArray without advancing the read head."""
        arr, _ = self.peek_padded(n_samples)
        return arr

    def peek_padded(
        self, n_samples: int | None = None, padding: int = 0
    ) -> tuple[AxisArray | None, int]:
        if self._data_buffer is None:
            return None, 0

        data_array, num_padded = self._data_buffer.peek_padded(n_samples, padding)

        if data_array is None or data_array.shape[0] == 0:
            return None, 0

        if hasattr(self._template_msg.axes[self._axis], "data"):
            out_axis_data, _ = self._axis_buffer.peek_padded(n_samples, padding)
            out_axis = replace(self._template_msg.axes[self._axis], data=out_axis_data)
        else:
            gain = self._template_msg.axes[self._axis].gain
            # The offset of the last unread sample
            last_unread_offset = self._axis_offset - (self.n_unread - 1) * gain
            # The offset of the first sample in the returned array (including padding)
            first_sample_offset = last_unread_offset - num_padded * gain
            out_axis = replace(
                self._template_msg.axes[self._axis], offset=first_sample_offset
            )

        return (
            replace(
                self._template_msg,
                data=data_array,
                axes={**self._template_msg.axes, self._axis: out_axis},
            ),
            num_padded,
        )

    def skip(self, n_samples: int) -> int:
        """Advances the read head by n_samples, discarding them."""
        if self._data_buffer is None:
            return 0

        skipped_data_count = self._data_buffer.skip(n_samples)

        if hasattr(self._template_msg.axes[self._axis], "data"):
            self._axis_buffer.skip(skipped_data_count)

        return skipped_data_count

    def get_data(self, n_samples: int | None = None) -> AxisArray | None:
        """Retrieves the oldest unread data as a new AxisArray and advances the read head."""
        retrieved_axis_array, _ = self.get_data_padded(n_samples, padding=0)
        return retrieved_axis_array

    def get_data_padded(
        self, n_samples: int | None = None, padding: int = 0
    ) -> tuple[AxisArray | None, int]:
        retrieved_axis_array, num_padded = self.peek_padded(n_samples, padding)

        if retrieved_axis_array is None or retrieved_axis_array.shape[0] == 0:
            return None, 0

        self.skip(retrieved_axis_array.shape[0] - num_padded)

        return retrieved_axis_array, num_padded

    def prune(self, n_samples: int) -> int:
        """Discards all but the last n_samples from the buffer."""
        if self._data_buffer is None:
            return 0

        n_to_discard = self.n_unread - n_samples
        if n_to_discard <= 0:
            return 0

        return self.skip(n_to_discard)

    def searchsorted(
        self, values: typing.Union[float, Array], side: str = "left"
    ) -> typing.Union[int, Array]:
        """
        Find the indices into which the given values would be inserted
        into the target axis data to maintain order.
        """
        if self._data_buffer is None:
            raise RuntimeError("Buffer not initialized. Cannot search.")

        targ_axis = self._template_msg.axes[self._axis]

        if hasattr(targ_axis, "data"):
            buffered_times = self._axis_buffer.peek(self.n_unread)
            return self._data_buffer.xp.searchsorted(buffered_times, values, side=side)

        else:
            if self.n_unread == 0:
                return (
                    0
                    if isinstance(values, float)
                    else self._data_buffer.xp.zeros_like(values, dtype=int)
                )

            # Correctly calculate the timestamp of the first sample in the buffer
            first_sample_offset = (
                self._axis_offset - (self.n_unread - 1) * targ_axis.gain
            )

            # Calculate indices
            indices = (values - first_sample_offset) / targ_axis.gain
            if side == "right":
                return self._data_buffer.xp.ceil(indices).astype(int)
            else:
                return self._data_buffer.xp.floor(indices).astype(int)
