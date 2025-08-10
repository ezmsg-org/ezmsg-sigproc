import collections
import numpy as np
import typing

UpdateStrategy = typing.Literal["immediate", "threshold", "on_demand"]


class HybridBuffer:
    """A buffer that combines a deque for fast, non-blocking appends with a
    contiguous NumPy circular buffer for fast, zero-copy reads.

    This buffer is designed for scenarios where data is written in small, frequent
    chunks (e.g., from a real-time stream) and read in larger, less frequent blocks.

    The buffer stores samples along the first dimension, and other dimensions
    (e.g., channels, spatial dimensions) are defined by `other_shape`.

    Args:
        maxlen: The maximum number of samples to store in the circular buffer.
        other_shape: A tuple defining the shape of the non-sample dimensions.
        dtype: The numpy data type of the samples.
        update_strategy: The strategy for synchronizing the deque to the circular buffer.
            - "immediate": Sync on every write. Most overhead, but buffer is always up-to-date.
            - "threshold": Sync when the deque size exceeds a certain number of samples.
            - "on_demand": Sync only when data is requested. Best for write-heavy workloads.
        threshold: The number of samples to accumulate in the deque before syncing when
            using the "threshold" strategy.

    """

    def __init__(
        self,
        maxlen: int,
        other_shape: typing.Tuple[int, ...],
        dtype: np.dtype,
        update_strategy: UpdateStrategy = "on_demand",
        threshold: int = 0,
    ):
        self._deque = collections.deque()
        self._buffer = np.empty((maxlen, *other_shape), dtype=dtype)
        self._maxlen = maxlen
        self._other_shape = other_shape
        self._update_strategy = update_strategy
        self._threshold = threshold

        self._head = 0  # Index of the next write position
        self._n_samples = 0  # Number of valid samples in the circular buffer
        self._deque_len = 0  # Total number of samples across all messages in the deque

    @property
    def n_samples(self) -> int:
        """The total number of samples currently available in the buffer (circular + deque)."""
        self._sync_if_needed()
        return self._n_samples

    def add_message(self, message: np.ndarray):
        """Appends a new message (a numpy array of samples) to the internal deque."""
        if self._other_shape == (1,) and message.ndim == 1:
            message = message[:, np.newaxis]

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
            and self._deque_len >= self._threshold
        ):
            self._sync()

    def get_data(self, n_samples: int | None = None) -> np.ndarray:
        """
        Retrieves the most recent `n_samples` from the buffer.

        This may trigger a synchronization of the deque and the circular buffer
        if the update strategy is "on_demand".

        Args:
            n_samples: The number of recent samples to retrieve. If None, returns all
                available samples in the buffer.

        Returns:
            A numpy array containing the requested samples. This may be a view
            or a copy, depending on whether the data is contiguous in memory.
        """
        self._sync_if_needed()

        if n_samples is None:
            n_samples = self._n_samples

        if n_samples > self._n_samples:
            raise ValueError(
                f"Requested {n_samples} samples, but only {self._n_samples} are available."
            )

        if n_samples == 0:
            return np.empty((0, *self._other_shape), dtype=self._buffer.dtype)

        start_idx = (self._head - n_samples) % self._maxlen
        end_idx = self._head

        if start_idx < end_idx:
            # Contiguous block, return a view
            return self._buffer[start_idx:end_idx]
        else:
            # Wraps around, requires a copy
            data = np.empty((n_samples, *self._other_shape), dtype=self._buffer.dtype)
            part1_len = self._maxlen - start_idx
            data[:part1_len] = self._buffer[start_idx:]
            data[part1_len:] = self._buffer[:end_idx]
            return data

    def _sync_if_needed(self):
        if self._update_strategy == "on_demand" and self._deque:
            self._sync()

    def _sync(self):
        """Transfers all data from the deque to the circular numpy buffer."""
        if not self._deque:
            return

        # Concatenate all messages in the deque into a single array
        all_new_data = np.concatenate(list(self._deque), axis=0)
        self._deque.clear()
        self._deque_len = 0

        n_new = all_new_data.shape[0]

        if n_new >= self._maxlen:
            # If new data is larger than buffer, just take the latest part
            self._buffer[:] = all_new_data[-self._maxlen :, ...]
            self._head = 0
            self._n_samples = self._maxlen
            return

        end_idx = self._head

        # Two-part copy if it wraps around the end of the buffer
        space_til_end = self._maxlen - end_idx
        if n_new > space_til_end:
            # Wraps around
            part1_len = space_til_end
            part2_len = n_new - part1_len
            self._buffer[end_idx:] = all_new_data[:part1_len]
            self._buffer[:part2_len] = all_new_data[part1_len:]
            self._head = part2_len
        else:
            # Fits contiguously
            self._buffer[end_idx : end_idx + n_new] = all_new_data
            self._head = (end_idx + n_new) % self._maxlen

        self._n_samples = min(self._n_samples + n_new, self._maxlen)
