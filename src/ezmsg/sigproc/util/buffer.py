import collections
import typing

Array = typing.TypeVar("Array")
ArrayNamespace = typing.Any
DType = typing.Any
UpdateStrategy = typing.Literal["immediate", "threshold", "on_demand"]


class HybridBuffer:
    """A stateful, FIFO buffer that combines a deque for fast appends with a
    contiguous circular buffer for efficient, advancing reads.

    This buffer is designed to be agnostic to the array library used (e.g., NumPy,
    CuPy, PyTorch) via the Python Array API standard.

    Args:
        array_namespace: The array library (e.g., numpy, cupy) that conforms to the Array API.
        capacity: The current maximum number of samples to store in the circular buffer.
        other_shape: A tuple defining the shape of the non-sample dimensions.
        dtype: The data type of the samples, belonging to the provided array_namespace.
        update_strategy: The strategy for synchronizing the deque to the circular buffer.
        threshold: The number of samples to accumulate in the deque before syncing.
    """

    def __init__(
        self,
        array_namespace: ArrayNamespace,
        capacity: int,
        other_shape: tuple[int, ...],
        dtype: DType,
        update_strategy: UpdateStrategy = "on_demand",
        threshold: int = 0,
    ):
        self.xp = array_namespace
        self._deque = collections.deque()
        self._buffer = self.xp.empty((capacity, *other_shape), dtype=dtype)
        self._capacity = capacity
        self._other_shape = other_shape
        self._update_strategy = update_strategy
        self._threshold = threshold

        self._head = 0  # Write pointer
        self._tail = 0  # Read pointer
        self._buff_unread = 0
        self._deque_len = 0

    @property
    def n_unread(self) -> int:
        """The total number of unread samples available (in buffer and deque)."""
        return self._buff_unread + self._deque_len

    def add_message(self, message: Array):
        """Appends a new message (an array of samples) to the internal deque."""
        if self._other_shape == (1,) and message.ndim == 1:
            message = message[:, self.xp.newaxis]

        if message.shape[1:] != self._other_shape:
            raise ValueError(
                f"Message shape {message.shape[1:]} does not match buffer's other_shape {self._other_shape}"
            )

        self._deque.append(message)
        self._deque_len += message.shape[0]

        if self._update_strategy == "immediate":
            self._sync()
        elif (
            self._update_strategy == "threshold"
            and self._threshold > 0
            and self.n_unread >= self._threshold
        ):
            self._sync()

    def get_data(self, n_samples: int | None = None) -> Array:
        """
        Retrieves the oldest unread samples from the buffer and advances the read head.

        Args:
            n_samples: The number of samples to retrieve. If None, returns all
                unread samples.

        Returns:
            An array containing the requested samples. This may be a view or a copy.
        """
        data = self.peek(n_samples)
        self.skip(data.shape[0])
        return data

    def get_data_padded(
        self, n_samples: int | None = None, padding: int = 0
    ) -> tuple[Array, int]:
        """
        Retrieves the oldest unread samples from the buffer with padding and advances the read head.

        Args:
            n_samples: The number of samples to retrieve. If None, returns all
                unread samples.
            padding: The number of samples to include before the oldest unread sample.

        Returns:
            A tuple containing:
                - An array containing the requested samples. This may be a view or a copy.
                - The number of padded samples included.
        """
        data, num_padded = self.peek_padded(n_samples, padding=padding)
        self.skip(data.shape[0] - num_padded)
        return data, num_padded

    def peek(self, n_samples: int | None = None) -> Array:
        """
        Retrieves the oldest unread samples from the buffer without
        advancing the read head.

        Args:
            n_samples: The number of samples to retrieve. If None, returns all
                unread samples.

        Returns:
            An array containing the requested samples. This may be a view or a copy.
        """
        data, _ = self.peek_padded(n_samples, padding=0)
        return data

    def peek_padded(
        self, n_samples: int | None = None, padding: int = 0
    ) -> tuple[Array, int]:
        """
        Retrieves the oldest unread samples from the buffer with padding without
        advancing the read head.

        Args:
            n_samples: The number of samples to retrieve. If None, returns all
                unread samples.
            padding: The number of samples to include before the oldest unread sample.

        Returns:
            A tuple containing:
                - An array containing the requested samples. This may be a view or a copy.
                - The number of padded samples included.
        """
        self._sync_if_needed()

        if n_samples is None:
            n_samples = self._buff_unread
        elif n_samples > self._buff_unread:
            raise ValueError(
                f"Requested to peek {n_samples} samples, but only {self._buff_unread} are available."
            )

        n_to_peek = min(n_samples, self._buff_unread)

        if self._head >= self._tail:
            n_hist = self._tail
        else:
            n_hist = self._tail - self._head

        num_padded = min(padding, n_hist)

        if n_to_peek == 0 and num_padded == 0:
            return self.xp.empty((0, *self._other_shape), dtype=self._buffer.dtype), 0

        start_idx = (self._tail - num_padded + self._capacity) % self._capacity
        n_to_peek_padded = n_to_peek + num_padded

        if start_idx + n_to_peek_padded > self._capacity:
            part1_len = self._capacity - start_idx
            part2_len = n_to_peek_padded - part1_len
            data = self.xp.empty(
                (n_to_peek_padded, *self._other_shape), dtype=self._buffer.dtype
            )
            data[:part1_len] = self._buffer[start_idx:]
            data[part1_len:] = self._buffer[:part2_len]
        else:
            data = self._buffer[start_idx : start_idx + n_to_peek_padded]

        return data, num_padded

    def skip(self, n_samples: int) -> int:
        """
        Advances the read head by n_samples, discarding them.

        Args:
            n_samples: The number of samples to skip.

        Returns:
            The number of samples actually skipped.
        """
        self._sync_if_needed()

        n_to_skip = min(n_samples, self._buff_unread)

        if n_to_skip == 0:
            return 0

        self._tail = (self._tail + n_to_skip) % self._capacity
        self._buff_unread -= n_to_skip

        return n_to_skip

    def _sync_if_needed(self):
        if self._update_strategy == "on_demand" and self._deque:
            self._sync()

    def _sync(self):
        """Transfers all data from the deque to the circular buffer."""
        if not self._deque:
            return

        all_new_data = self.xp.concatenate(list(self._deque), axis=0)
        self._deque.clear()
        self._deque_len = 0

        n_new = all_new_data.shape[0]

        # If new data is larger than buffer, just keep the latest
        if n_new >= self._capacity:
            self._buffer[:] = all_new_data[-self._capacity :, ...]
            self._head = 0
            self._tail = 0
            self._buff_unread = self._capacity
            return

        n_free = self._capacity - self._buff_unread

        # Determine how many samples we will overwrite then skip those.
        n_overwrite = n_new - n_free
        if n_overwrite > 0:
            self.skip(n_overwrite)

        # Copy data to buffer
        space_til_end = self._capacity - self._head
        if n_new > space_til_end:
            # Two-part copy (wraps around)
            part1_len = space_til_end
            part2_len = n_new - part1_len
            self._buffer[self._head :] = all_new_data[:part1_len]
            self._buffer[:part2_len] = all_new_data[part1_len:]
        else:
            # Single-part copy
            self._buffer[self._head : self._head + n_new] = all_new_data

        self._head = (self._head + n_new) % self._capacity
        self._buff_unread += n_new
