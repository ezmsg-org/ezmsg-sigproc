import collections
import math
import typing
import warnings

Array = typing.TypeVar("Array")
ArrayNamespace = typing.Any
DType = typing.Any
UpdateStrategy = typing.Literal["immediate", "threshold", "on_demand"]
OverflowStrategy = typing.Literal["grow", "raise", "drop", "warn-overwrite"]


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
        overflow_strategy: The strategy for handling overflow when the buffer is full.
            Options are "grow", "raise", "drop", or "warn-overwrite". If "grow" (default), the buffer will
            increase its capacity to accommodate new samples up to max_size. If "raise", an error will be
            raised when the buffer is full. If "drop", the overflowing samples will be ignored.
            If "warn-overwrite", a warning will be logged then the overflowing samples will
            overwrite previously-unread samples.
        max_size: The maximum size of the buffer in bytes.
            If the buffer exceeds this size, it will raise an error.
        warn_once: If True, will only warn once on overflow when using "warn-overwrite" strategy.
    """

    def __init__(
        self,
        array_namespace: ArrayNamespace,
        capacity: int,
        other_shape: tuple[int, ...],
        dtype: DType,
        update_strategy: UpdateStrategy = "on_demand",
        threshold: int = 0,
        overflow_strategy: OverflowStrategy = "grow",
        max_size: int = 1024**3,  # 1 GB default max size
        warn_once: bool = True,
    ):
        self.xp = array_namespace
        self._capacity = capacity
        self._deque = collections.deque()
        self._update_strategy = update_strategy
        self._threshold = threshold
        self._overflow_strategy = overflow_strategy
        self._max_size = max_size
        self._warn_once = warn_once

        self._buffer = self.xp.empty((capacity, *other_shape), dtype=dtype)
        self._head = 0  # Write pointer
        self._tail = 0  # Read pointer
        self._buff_unread = 0  # Number of unread samples in the circular buffer
        self._buff_read = 0  # Tracks samples read and still in buffer
        self._deque_len = 0  # Number of unread samples in the deque
        self._last_overflow = (
            0  # Tracks the last overflow count, overwritten or skipped
        )
        self._warned = False  # Tracks if we've warned already (for warn_once)

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
        other_shape = self._buffer.shape[1:]
        if other_shape == (1,) and block.ndim == 1:
            block = block[:, self.xp.newaxis]

        if block.shape[1:] != other_shape:
            raise ValueError(
                f"Block shape {block.shape[1:]} does not match buffer's other_shape {other_shape}"
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

        self._flush_if_needed()

        if n_samples == 0:
            return self._buffer[:0]

        if self._tail + n_samples > self._capacity:
            # discontiguous read (wraps around)
            part1_len = self._capacity - self._tail
            part2_len = n_samples - part1_len
            data = self.xp.empty(
                (n_samples, *self._buffer.shape[1:]), dtype=self._buffer.dtype
            )
            data[:part1_len] = self._buffer[self._tail :]
            data[part1_len:] = self._buffer[:part2_len]
        else:
            data = self._buffer[self._tail : self._tail + n_samples]

        return data

    def peek_at(self, idx: int) -> Array:
        """
        Retrieves a specific sample from the buffer without advancing the read head.
        Note: This method can read into the deque if the requested sample is beyond
         the unread samples in the buffer, but it will not cause a flush.

        Args:
            idx: The index of the sample to retrieve, relative to the read head.

        Returns:
            An array containing the requested sample. This may be a view or a copy.
        """
        if idx < 0 or idx >= self.available():
            raise IndexError(f"Index {idx} out of bounds for unread samples.")

        if idx < self._buff_unread:
            # The requested sample is within the unread samples in the buffer.
            idx = (self._tail + idx) % self._capacity
            return self._buffer[idx : idx + 1]
        # The requested sample is in the deque.
        idx -= self._buff_unread
        deq_splits = self.xp.cumsum([0] + [_.shape[0] for _ in self._deque], dtype=int)
        arr_idx = self.xp.searchsorted(deq_splits, idx, side="right") - 1
        idx -= deq_splits[arr_idx]
        return self._deque[arr_idx][idx : idx + 1]

    def peek_last(self) -> Array:
        """
        Retrieves the last sample in the buffer without advancing the read head.
        """
        if self._deque:
            return self._deque[-1][-1:]
        elif self._buff_unread > 0:
            idx = (self._head - 1 + self._capacity) % self._capacity
            return self._buffer[idx : idx + 1]
        else:
            raise IndexError("Cannot peek last from an empty buffer.")

    def seek(self, n_samples: int) -> int:
        """
        Advances the read head by n_samples.

        Args:
            n_samples: The number of samples to seek.
            Will seek forward if positive or backward if negative.

        Returns:
            The number of samples actually skipped.
        """
        if n_samples > self._buff_unread:
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
        """
        Transfers all data from the deque to the circular buffer.
        Note: This may overwrite data depending on the overflow strategy,
            which will invalidate previous state variables.
            TODO: if 'warn-overwrite', also keep a member variable
             to track how many samples were overwritten.
        """
        if not self._deque:
            return

        all_new_data = self.xp.concatenate(list(self._deque), axis=0)
        self._deque.clear()
        self._deque_len = 0

        n_new = all_new_data.shape[0]
        n_free = self._capacity - self._buff_unread
        n_overflow = max(0, n_new - n_free)

        # If new data is larger than buffer and overflow strategy is "warn-overwrite",
        #  then we can take a shortcut and replace the entire buffer.
        if n_new >= self._capacity and self._overflow_strategy == "warn-overwrite":
            if n_overflow > 0 and (not self._warn_once or not self._warned):
                self._warned = True
                warnings.warn(
                    f"Buffer overflow: {n_new} samples received, but only {self._capacity - self._buff_unread} available. "
                    f"Overwriting {n_overflow} previous samples.",
                    RuntimeWarning,
                )
            self._buffer[:] = all_new_data[-self._capacity :, ...]
            self._head = 0
            self._tail = 0
            self._buff_unread = self._capacity
            self._buff_read = 0
            self._last_overflow = n_overflow
            return

        if n_overflow > 0:
            if self._overflow_strategy == "raise":
                raise OverflowError(
                    f"Buffer overflow: {n_new} samples received, but only {n_free} available."
                )
            elif self._overflow_strategy == "warn-overwrite":
                if not self._warn_once or not self._warned:
                    self._warned = True
                    warnings.warn(
                        f"Buffer overflow: {n_new} samples received, but only {n_free} available. "
                        f"Overwriting {n_overflow} previous samples.",
                        RuntimeWarning,
                    )
                # Move the tail forward to make room for the new data.
                self.seek(n_overflow)
                self._buff_read = 0
                self._last_overflow = n_overflow
            elif self._overflow_strategy == "drop":
                # Drop the overflow samples
                all_new_data = all_new_data[:n_free, ...]
                n_new = all_new_data.shape[0]
                self._last_overflow = n_overflow
                if n_new == 0:
                    return
            elif self._overflow_strategy == "grow":
                self._grow_buffer(self._capacity + n_new)
                self._last_overflow = 0

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

    def _grow_buffer(self, min_capacity: int):
        """
        Grows the buffer to at least min_capacity.
        This is a helper method for the overflow strategy "grow".
        """
        if self._capacity >= min_capacity:
            return

        other_shape = self._buffer.shape[1:]
        max_capacity = self._max_size / (
            self._buffer.dtype.itemsize * math.prod(other_shape)
        )
        if min_capacity > max_capacity:
            raise OverflowError(
                f"Cannot grow buffer to {min_capacity} samples, "
                f"maximum capacity is {max_capacity} samples ({self._max_size} bytes)."
            )

        new_capacity = min(max_capacity, max(self._capacity * 2, min_capacity))
        new_buffer = self.xp.empty(
            (new_capacity, *other_shape), dtype=self._buffer.dtype
        )

        # Copy existing data to new buffer
        total_samples = self._buff_read + self._buff_unread
        if total_samples > 0:
            start_idx = (self._tail - self._buff_read) % self._capacity
            stop_idx = (self._tail + self._buff_unread) % self._capacity
            if stop_idx > start_idx:
                # Data is contiguous
                new_buffer[:total_samples] = self._buffer[start_idx:stop_idx]
            else:
                # Data wraps around. We write it in 2 parts.
                part1_len = self._capacity - start_idx
                part2_len = stop_idx
                new_buffer[:part1_len] = self._buffer[start_idx:]
                new_buffer[part1_len : part1_len + part2_len] = self._buffer[:stop_idx]
            # self._buff_read stays the same
            self._tail = self._buff_read
            # self._buff_unread stays the same
            self._head = self._tail + self._buff_unread
        else:
            self._tail = 0
            self._head = 0

        self._buffer = new_buffer
        self._capacity = new_capacity
