import collections
import typing

UpdateStrategy = typing.Literal["immediate", "threshold", "on_demand"]
Array = typing.TypeVar("Array")
ArrayNamespace = typing.Any
DType = typing.Any


class HybridBuffer:
    """A buffer that combines a deque for fast, non-blocking appends with a
    contiguous circular buffer for fast, zero-copy reads.

    This buffer is designed to be agnostic to the array library used (e.g., NumPy,
    CuPy, PyTorch) via the Python Array API standard.

    The buffer stores samples along the first dimension, and other dimensions
    are defined by `other_shape`.

    Args:
        array_namespace: The array library (e.g., numpy, cupy) that conforms to the Array API.
        maxlen: The maximum number of samples to store in the circular buffer.
        other_shape: A tuple defining the shape of the non-sample dimensions.
        dtype: The data type of the samples, belonging to the provided array_namespace.
        update_strategy: The strategy for synchronizing the deque to the circular buffer.
        threshold: The number of samples to accumulate in the deque before syncing.

    """

    def __init__(
        self,
        array_namespace: ArrayNamespace,
        maxlen: int,
        other_shape: tuple[int, ...],
        dtype: DType,
        update_strategy: UpdateStrategy = "on_demand",
        threshold: int = 0,
    ):
        self.xp = array_namespace
        self._deque = collections.deque()
        self._buffer = self.xp.empty((maxlen, *other_shape), dtype=dtype)
        self._maxlen = maxlen
        self._other_shape = other_shape
        self._update_strategy = update_strategy
        self._threshold = threshold

        self._head = 0
        self._n_samples = 0
        self._deque_len = 0

    @property
    def n_samples(self) -> int:
        """The total number of samples currently available in the buffer."""
        self._sync_if_needed()
        return self._n_samples

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
            and self._deque_len >= self._threshold
        ):
            self._sync()

    def get_data(self, n_samples: int | None = None) -> Array:
        """
        Retrieves the most recent `n_samples` from the buffer.

        Args:
            n_samples: The number of recent samples to retrieve. If None, returns all
                available samples in the buffer.

        Returns:
            An array containing the requested samples. This may be a view or a copy.
        """
        self._sync_if_needed()

        if n_samples is None:
            n_samples = self._n_samples

        if n_samples > self._n_samples:
            raise ValueError(
                f"Requested {n_samples} samples, but only {self._n_samples} are available."
            )

        if n_samples == 0:
            return self.xp.empty((0, *self._other_shape), dtype=self._buffer.dtype)

        start_idx = (self._head - n_samples) % self._maxlen
        end_idx = self._head

        if start_idx < end_idx:
            return self._buffer[start_idx:end_idx]
        else:
            data = self.xp.empty(
                (n_samples, *self._other_shape), dtype=self._buffer.dtype
            )
            part1_len = self._maxlen - start_idx
            data[:part1_len] = self._buffer[start_idx:]
            data[part1_len:] = self._buffer[:end_idx]
            return data

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

        if n_new >= self._maxlen:
            self._buffer[:] = all_new_data[-self._maxlen :, ...]
            self._head = 0
            self._n_samples = self._maxlen
            return

        end_idx = self._head

        space_til_end = self._maxlen - end_idx
        if n_new > space_til_end:
            part1_len = space_til_end
            part2_len = n_new - part1_len
            self._buffer[end_idx:] = all_new_data[:part1_len]
            self._buffer[:part2_len] = all_new_data[part1_len:]
            self._head = part2_len
        else:
            self._buffer[end_idx : end_idx + n_new] = all_new_data
            self._head = (end_idx + n_new) % self._maxlen

        self._n_samples = min(self._n_samples + n_new, self._maxlen)
