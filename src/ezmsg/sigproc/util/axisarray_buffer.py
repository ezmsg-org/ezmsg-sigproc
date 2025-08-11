import typing

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from array_api_compat import get_namespace

from .buffer import HybridBuffer


Array = typing.TypeVar("Array")


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

    def __init__(self, duration: float, axis: str = "time", **kwargs):
        self.duration = duration
        self._axis = axis
        self.buffer_kwargs = kwargs
        self._data_buffer = None
        self._axis_offset = 0.0
        self._axis_buffer = None
        self._template_msg = None

    @property
    def n_unread(self) -> int:
        """The total number of unread samples currently available in the buffer."""
        if self._data_buffer is None:
            return 0
        return self._data_buffer.n_unread

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
        if self._data_buffer is None:
            return None

        total_unread = self.n_unread
        data_array = self._data_buffer.peek(n_samples)

        if data_array is None or data_array.shape[0] == 0:
            return None

        num_peeked = data_array.shape[0]

        if hasattr(self._template_msg.axes[self._axis], "data"):
            out_axis_data = self._axis_buffer.peek(num_peeked)
            out_axis = replace(self._template_msg.axes[self._axis], data=out_axis_data)
        else:
            gain = self._template_msg.axes[self._axis].gain
            offset = self._axis_offset - (total_unread - 1) * gain
            out_axis = replace(self._template_msg.axes[self._axis], offset=offset)

        return replace(
            self._template_msg,
            data=data_array,
            axes={**self._template_msg.axes, self._axis: out_axis},
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
        retrieved_axis_array = self.peek(n_samples)

        if retrieved_axis_array is None or retrieved_axis_array.shape[0] == 0:
            return None

        self.skip(retrieved_axis_array.shape[0])

        return retrieved_axis_array
