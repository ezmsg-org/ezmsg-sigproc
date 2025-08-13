import collections
import typing

Array = typing.TypeVar("Array")
ArrayNamespace = typing.Any
DType = typing.Any
UpdateStrategy = typing.Literal["immediate", "threshold", "on_demand"]
# OverflowStrategy = typing.Literal["skip", "overwrite", "raise", "grow"]


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
        update_strategy: The strategy for synchronizing the deque to the circular buffer (flushing).
        threshold: The number of samples to accumulate in the deque before flushing.
          Ignored if update_strategy is "immediate" or "on_demand".
    """

    def __init__(
        self,
        array_namespace: ArrayNamespace,
        capacity: int,
        other_shape: tuple[int, ...],
        dtype: DType,
        update_strategy: UpdateStrategy = "on_demand",
        threshold: int = 0,
        # overflow_strategy: OverflowStrategy = "overwrite",
    ):
        self.xp = array_namespace
        self._capacity = capacity
        self._deque = collections.deque()
        self._other_shape = other_shape
        self._update_strategy = update_strategy
        self._threshold = threshold

        self._buffer = self.xp.empty((capacity, *other_shape), dtype=dtype)
        self._head = 0  # Write pointer
        self._tail = 0  # Read pointer
        self._buff_unread = 0  # Number of unread samples in the circular buffer
        self._buff_read = 0  # Tracks samples read and still in buffer
        self._deque_len = 0  # Number of unread samples in the deque

    @property
    def capacity(self) -> int:
        """The maximum number of samples that can be stored in the buffer."""
        return self._capacity

    def available(self) -> int:
        """The total number of unread samples available (in buffer and deque)."""
        return self._buff_unread + self._deque_len

    def is_empty(self) -> bool:
        """Returns True if there are no unread samples in the buffer or deque."""
        return self.available() == 0

    def is_full(self) -> bool:
        """Returns True if the buffer is full and cannot _flush_ more samples without overwriting."""
        return self._buff_unread == self._capacity

    def tell(self) -> int:
        """Returns the number of samples that have been read and are still in the buffer."""
        return self._buff_read

    def write(self, block: Array):
        """Appends a new block (an array of samples) to the internal deque."""
        if self._other_shape == (1,) and block.ndim == 1:
            block = block[:, self.xp.newaxis]

        if block.shape[1:] != self._other_shape:
            raise ValueError(
                f"Block shape {block.shape[1:]} does not match buffer's other_shape {self._other_shape}"
            )

        self._deque.append(block)
        self._deque_len += block.shape[0]

        if self._update_strategy == "immediate" or (
            self._update_strategy == "threshold"
            and (0 < self._threshold <= self._deque_len)
        ):
            self.flush()

    def read(
        self,
        n_samples: int | None = None,
    ) -> tuple[Array, int]:
        """
        Retrieves the oldest unread samples from the buffer with padding and advances the read head.

        Args:
            n_samples: The number of samples to retrieve. If None, returns all
                unread samples.

        Returns:
            An array containing the requested samples. This may be a view or a copy.
        """
        data = self.peek(n_samples)
        self.seek(data.shape[0])
        return data

    def peek(self, n_samples: int | None = None) -> Array:
        """
        Retrieves the oldest unread samples from the buffer with padding without
        advancing the read head.

        Args:
            n_samples: The number of samples to retrieve. If None, returns all
                unread samples.

        Returns:
            An array containing the requested samples. This may be a view or a copy.
        """
        if n_samples is None:
            n_samples = self.available()
        elif n_samples > self.available():
            raise ValueError(
                f"Requested to peek {n_samples} samples, but only {self.available()} are available."
            )
        n_samples = min(n_samples, self.available())

        # TODO: If flush would overflow and strategy != "grow" then we should read the number of overflow
        #  samples and prepend to the output.

        self._flush_if_needed()

        if n_samples == 0:
            return self.xp.empty((0, *self._other_shape), dtype=self._buffer.dtype)

        if self._tail + n_samples > self._capacity:
            # discontiguous read (wraps around)
            part1_len = self._capacity - self._tail
            part2_len = n_samples - part1_len
            data = self.xp.empty(
                (n_samples, *self._other_shape), dtype=self._buffer.dtype
            )
            data[:part1_len] = self._buffer[self._tail :]
            data[part1_len:] = self._buffer[:part2_len]
        else:
            data = self._buffer[self._tail : self._tail + n_samples]

        return data

    def seek(self, n_samples: int) -> int:
        """
        Advances the read head by n_samples.

        Args:
            n_samples: The number of samples to seek.
            Will seek forward if positive or backward if negative.

        Returns:
            The number of samples actually skipped.
        """
        self._flush_if_needed()

        n_to_seek = max(min(n_samples, self._buff_unread), -self._buff_read)

        if n_to_seek == 0:
            return 0

        self._tail = (self._tail + n_to_seek) % self._capacity
        self._buff_unread -= n_to_seek
        self._buff_read += n_to_seek

        return n_to_seek

    def _flush_if_needed(self):
        if self._update_strategy == "on_demand" and self._deque:
            self.flush()

    def flush(self):
        """Transfers all data from the deque to the circular buffer."""
        if not self._deque:
            return

        all_new_data = self.xp.concatenate(list(self._deque), axis=0)
        self._deque.clear()
        self._deque_len = 0

        n_new = all_new_data.shape[0]

        # If new data is larger than buffer, just keep the latest
        #  new data that fits in the buffer.
        #  TODO: Handle growing the buffer if self._overflow_strategy == "grow".
        if n_new >= self._capacity:
            self._buffer[:] = all_new_data[-self._capacity :, ...]
            self._head = 0
            self._tail = 0
            self._buff_unread = self._capacity
            self._buff_read = 0
            return

        n_free = self._capacity - self._buff_unread

        # Determine how many samples would overflow
        n_overflow = max(n_new - n_free, 0)
        if n_overflow > 0:  # TODO: and self._overflow_strategy == "overwrite":
            # If we have overflow, we need to seek forward to make room.
            self.seek(n_overflow)
            self._buff_read = 0
        # TODO: Handle other overflow strategies like "skip", "raise", or "grow".

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
        if (self._buff_read > self._tail) or (self._tail > self._head):
            # We have wrapped around the buffer; our count of read samples
            #  is simply the buffer capacity minus the count of unread samples.
            self._buff_read = self._capacity - self._buff_unread
