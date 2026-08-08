"""A stateful, FIFO buffer that combines a deque for fast appends with a
contiguous circular buffer for efficient, advancing reads.
"""

import collections
import math
import typing
import warnings

from .array import xp_empty, xp_itemsize

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
        other_shape: A tuple defining the shape of the non-sample dimensions, in order and
            excluding the sample dimension itself.
        dtype: The data type of the samples, belonging to the provided array_namespace.
        sample_axis: The position of the sample (e.g. time) dimension within the stored
            arrays. The circular buffer is allocated as
            ``(*other_shape[:sample_axis], capacity, *other_shape[sample_axis:])``, so
            ``sample_axis=-1`` keeps samples contiguous for time-last data and avoids a
            transpose on every write. Negative values are accepted and normalized.
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
        sample_axis: int = 0,
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

        self._other_shape = tuple(other_shape)
        # Normalize once; every downstream index is built from the non-negative form.
        self._sample_axis = sample_axis % (len(self._other_shape) + 1)
        # Prebuilt full-slice padding so an index for the sample axis is a tuple
        # concatenation rather than a per-call list build.
        self._lead: tuple[slice, ...] = (slice(None),) * self._sample_axis
        self._trail: tuple[slice, ...] = (slice(None),) * (len(self._other_shape) - self._sample_axis)

        self._buffer = xp_empty(self.xp, self._shape_with(capacity), dtype=dtype)
        self._head = 0  # Write pointer
        self._tail = 0  # Read pointer
        self._buff_unread = 0  # Number of unread samples in the circular buffer
        self._buff_read = 0  # Tracks samples read and still in buffer
        self._deque_len = 0  # Number of unread samples in the deque
        self._last_overflow = 0  # Tracks the last overflow count, overwritten or skipped
        self._warned = False  # Tracks if we've warned already (for warn_once)

    # -- Sample-axis helpers -------------------------------------------------
    #
    # The sample dimension is not necessarily dim 0, so every length query and
    # every slice of a stored array routes through these. Keeping them tiny and
    # in one place is what makes the rest of the class layout-agnostic.

    @property
    def sample_axis(self) -> int:
        """Position of the sample (e.g. time) dimension within stored arrays."""
        return self._sample_axis

    def _n(self, block: Array) -> int:
        """Number of samples in ``block`` along the sample axis."""
        return block.shape[self._sample_axis]

    def _idx(self, sl) -> tuple:
        """Index tuple selecting ``sl`` along the sample axis, all else full.

        Usable for both reads and in-place writes, which is why this returns an
        index tuple rather than delegating to ``slice_along_axis``.
        """
        return self._lead + (sl,) + self._trail

    def _shape_with(self, n_samples: int) -> tuple[int, ...]:
        """Full array shape holding ``n_samples`` samples in this layout."""
        return (*self._other_shape[: self._sample_axis], n_samples, *self._other_shape[self._sample_axis :])

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
        other_shape = self._other_shape
        if other_shape == (1,) and block.ndim == 1:
            # Convenience: accept a bare 1-D block for a single-channel buffer.
            # The new axis goes wherever the sample axis is not.
            block = block[:, self.xp.newaxis] if self._sample_axis == 0 else block[self.xp.newaxis, :]

        block_other = block.shape[: self._sample_axis] + block.shape[self._sample_axis + 1 :]
        if block_other != other_shape:
            raise ValueError(f"Block shape {block_other} does not match buffer's other_shape {other_shape}")

        # Most overflow strategies are handled during flush, but there are a couple
        # scenarios that can be evaluated on write to give immediate feedback.
        new_len = self._deque_len + self._n(block)
        if new_len > self._capacity and self._overflow_strategy == "raise":
            raise OverflowError(
                f"Buffer overflow: {new_len} samples awaiting in deque exceeds buffer capacity {self._capacity}."
            )
        bytes_per_sample = xp_itemsize(block.dtype) * math.prod(other_shape)
        if new_len * bytes_per_sample > self._max_size:
            raise OverflowError(
                f"deque contents would exceed max_size ({self._max_size}) on subsequent flush."
                "Are you reading samples frequently enough?"
            )

        self._deque.append(block)
        self._deque_len += self._n(block)

        if self._update_strategy == "immediate" or (
            self._update_strategy == "threshold" and (0 < self._threshold <= self._deque_len)
        ):
            self.flush()

    def _estimate_overflow(self, n_samples: int) -> int:
        """
        Estimates the number of samples that would overflow we requested n_samples
        from the buffer.
        """
        if n_samples > self.available():
            raise ValueError(f"Requested {n_samples} samples, but only {self.available()} are available.")
        n_overflow = 0
        if self._deque and (n_samples > self._buff_unread):
            # We would cause a flush, but would that cause an overflow?
            n_free = self._capacity - self._buff_unread
            n_overflow = max(0, self._deque_len - n_free)
        return n_overflow

    def read(
        self,
        n_samples: int | None = None,
    ) -> Array:
        """
        Retrieves the oldest unread samples from the buffer with padding
        and advances the read head.

        Args:
            n_samples: The number of samples to retrieve. If None, returns all
                unread samples.

        Returns:
            An array containing the requested samples. This may be a view or a copy.
            Note: The result may have more samples than the buffer.capacity as it
            may include samples from the deque in the output.
        """
        n_samples = n_samples if n_samples is not None else self.available()
        data = None
        offset = 0
        n_overflow = self._estimate_overflow(n_samples)
        if n_overflow > 0:
            first_read = self._buff_unread
            if (n_overflow - first_read) < self.capacity or (self._overflow_strategy == "drop"):
                # We can prevent the overflow (or at least *some* if using "drop"
                # strategy) by reading the samples in the buffer first to make room.
                data = xp_empty(self.xp, self._shape_with(n_samples), dtype=self._buffer.dtype)
                self.peek(first_read, out=data[self._idx(slice(None, first_read))])
                offset += first_read
                self.seek(first_read)
                n_samples -= first_read
        if data is None:
            data = self.peek(n_samples)
            self.seek(self._n(data))
        else:
            d2 = self.peek(n_samples, out=data[self._idx(slice(offset, None))])
            self.seek(self._n(d2))

        return data

    def peek(self, n_samples: int | None = None, out: Array | None = None) -> Array:
        """
        Retrieves the oldest unread samples from the buffer with padding without
        advancing the read head.

        Args:
            n_samples: The number of samples to retrieve. If None, returns all
                unread samples.
            out: Optionally, a destination array to store the samples.
                If provided, must have shape ``(n_samples, *other_shape)`` where
                other_shape matches the shape of the samples in the buffer.
                If ``out`` is provided then the data will always be copied into it,
                even if they are contiguous in the buffer.

        Returns:
            An array containing the requested samples. This may be a view or a copy.
            Note: The result may have more samples than the buffer.capacity as it
            may include samples from the deque in the output.
        """
        if n_samples is None:
            n_samples = self.available()
        elif n_samples > self.available():
            raise ValueError(f"Requested to peek {n_samples} samples, but only {self.available()} are available.")
        if out is not None and self._n(out) < n_samples:
            raise ValueError(f"Output array shape {out.shape} is smaller than requested {n_samples} samples.")

        if n_samples == 0:
            return self._buffer[self._idx(slice(0, 0))]

        self._flush_if_needed(n_samples=n_samples)

        if self._tail + n_samples > self._capacity:
            # discontiguous read (wraps around)
            part1_len = self._capacity - self._tail
            part2_len = n_samples - part1_len
            out = out if out is not None else xp_empty(self.xp, self._shape_with(n_samples), dtype=self._buffer.dtype)
            out[self._idx(slice(None, part1_len))] = self._buffer[self._idx(slice(self._tail, None))]
            out[self._idx(slice(part1_len, None))] = self._buffer[self._idx(slice(None, part2_len))]
        else:
            if out is not None:
                out[self._idx(slice(None))] = self._buffer[self._idx(slice(self._tail, self._tail + n_samples))]
            else:
                # No output array provided, just return a view
                out = self._buffer[self._idx(slice(self._tail, self._tail + n_samples))]

        return out

    def peek_at(self, idx: int, allow_flush: bool = False) -> Array:
        """
        Retrieves a specific sample from the buffer without advancing the read head.

        Args:
            idx: The index of the sample to retrieve, relative to the read head.
            allow_flush: If True, allows flushing the deque to the buffer if the
                requested sample is not in the buffer. If False and the sample is
                in the deque, the sample will be retrieved from the deque (slow!).

        Returns:
            An array containing the requested sample. This may be a view or a copy.
        """
        if idx < 0 or idx >= self.available():
            raise IndexError(f"Index {idx} out of bounds for unread samples.")

        if not allow_flush and idx >= self._buff_unread:
            # The requested sample is in the deque.
            idx -= self._buff_unread
            deq_splits = self.xp.cumsum([0] + [self._n(_) for _ in self._deque], dtype=int)
            arr_idx = self.xp.searchsorted(deq_splits, idx, side="right") - 1
            idx -= deq_splits[arr_idx]
            return self._deque[arr_idx][self._idx(slice(idx, idx + 1))]

        self._flush_if_needed(n_samples=idx + 1)

        # The requested sample is within the unread samples in the buffer.
        idx = (self._tail + idx) % self._capacity
        return self._buffer[self._idx(slice(idx, idx + 1))]

    def peek_last(self) -> Array:
        """
        Retrieves the last sample in the buffer without advancing the read head.
        """
        if self._deque:
            return self._deque[-1][self._idx(slice(-1, None))]
        elif self._buff_unread > 0:
            idx = (self._head - 1 + self._capacity) % self._capacity
            return self._buffer[self._idx(slice(idx, idx + 1))]
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
        self._flush_if_needed(n_samples=n_samples)

        n_to_seek = max(min(n_samples, self._buff_unread), -self._buff_read)

        if n_to_seek == 0:
            return 0

        self._tail = (self._tail + n_to_seek) % self._capacity
        self._buff_unread -= n_to_seek
        self._buff_read += n_to_seek

        return n_to_seek

    def _flush_if_needed(self, n_samples: int | None = None):
        if (
            self._update_strategy == "on_demand"
            and self._deque
            and (n_samples is None or n_samples > self._buff_unread)
        ):
            self.flush()

    def flush(self):
        """
        Transfers all data from the deque to the circular buffer.

        Note: This may overwrite data depending on the overflow strategy,
        which will invalidate previous state variables.
        """
        if not self._deque:
            return

        n_new = self._deque_len
        n_free = self._capacity - self._buff_unread
        n_overflow = max(0, n_new - n_free)

        # If new data is larger than buffer and overflow strategy is "warn-overwrite",
        #  then we can take a shortcut and replace the entire buffer.
        if n_new >= self._capacity and self._overflow_strategy == "warn-overwrite":
            if n_overflow > 0 and (not self._warn_once or not self._warned):
                self._warned = True
                warnings.warn(
                    f"Buffer overflow: {n_new} samples received, "
                    f"but only {self._capacity - self._buff_unread} available. "
                    f"Overwriting {n_overflow} previous samples.",
                    RuntimeWarning,
                )

            # We need to grab the last `self._capacity` samples from the deque
            samples_to_copy = self._capacity
            copied_samples = 0
            for block in reversed(self._deque):
                if copied_samples >= samples_to_copy:
                    break
                n_block = self._n(block)
                n_to_copy = min(n_block, samples_to_copy - copied_samples)
                start_idx = n_block - n_to_copy
                dst = self._idx(slice(samples_to_copy - copied_samples - n_to_copy, samples_to_copy - copied_samples))
                self._buffer[dst] = block[self._idx(slice(start_idx, None))]
                copied_samples += n_to_copy

            self._head = 0
            self._tail = 0
            self._buff_unread = self._capacity
            self._buff_read = 0
            self._last_overflow = n_overflow

        else:
            if n_overflow > 0:
                if self._overflow_strategy == "raise":
                    raise OverflowError(f"Buffer overflow: {n_new} samples received, but only {n_free} available.")
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
                    # Adjust the read pointer to account for the overflow. Should always be 0.
                    self._buff_read = max(0, self._buff_read - n_overflow)
                    self._last_overflow = n_overflow
                elif self._overflow_strategy == "drop":
                    # Drop the overflow samples from the deque
                    samples_to_drop = n_overflow
                    while samples_to_drop > 0 and self._deque:
                        block = self._deque[-1]
                        n_block = self._n(block)
                        if samples_to_drop >= n_block:
                            samples_to_drop -= n_block
                            self._deque.pop()
                        else:
                            block = self._deque.pop()
                            self._deque.append(block[self._idx(slice(None, -samples_to_drop))])
                            samples_to_drop = 0
                    n_new -= n_overflow
                    self._last_overflow = n_overflow

                elif self._overflow_strategy == "grow":
                    self._grow_buffer(self._capacity + n_new)
                    self._last_overflow = 0

            # Copy data to buffer by iterating over the deque
            for block in self._deque:
                n_block = self._n(block)
                space_til_end = self._capacity - self._head
                if n_block > space_til_end:
                    # Two-part copy (wraps around)
                    part1_len = space_til_end
                    part2_len = n_block - part1_len
                    self._buffer[self._idx(slice(self._head, None))] = block[self._idx(slice(None, part1_len))]
                    self._buffer[self._idx(slice(None, part2_len))] = block[self._idx(slice(part1_len, None))]
                else:
                    # Single-part copy
                    self._buffer[self._idx(slice(self._head, self._head + n_block))] = block
                self._head = (self._head + n_block) % self._capacity

            self._buff_unread += n_new
            if (self._buff_read > self._tail) or (self._tail > self._head):
                # We have wrapped around the buffer; our count of read samples
                #  is simply the buffer capacity minus the count of unread samples.
                self._buff_read = self._capacity - self._buff_unread
            if self._buff_read + self._buff_unread > self._capacity:
                self._buff_read = self._capacity - self._buff_unread

        self._deque.clear()
        self._deque_len = 0

    def _grow_buffer(self, min_capacity: int):
        """
        Grows the buffer to at least min_capacity.
        This is a helper method for the overflow strategy "grow".
        """
        if self._capacity >= min_capacity:
            return

        other_shape = self._other_shape
        bytes_per_sample = xp_itemsize(self._buffer.dtype) * math.prod(other_shape)
        if bytes_per_sample == 0:
            # A 0-size non-sample dim (e.g. a 0-channel stream from a source
            # sliced to no channels) makes each sample 0 bytes, so the max_size
            # byte budget never bounds capacity. Grow by the normal doubling
            # policy with no byte cap -- avoids dividing by 0 bytes/sample.
            new_capacity = max(self._capacity * 2, min_capacity)
        else:
            # Floor-divide: the byte budget bounds capacity to a whole number of
            # samples. True division would yield a float, which then flows into
            # both the xp_empty shape below and self._capacity -- array dims must
            # be ints, and a non-int capacity corrupts the ring-buffer arithmetic.
            max_capacity = self._max_size // bytes_per_sample
            if min_capacity > max_capacity:
                raise OverflowError(
                    f"Cannot grow buffer to {min_capacity} samples, "
                    f"maximum capacity is {max_capacity} samples ({self._max_size} bytes)."
                )
            new_capacity = min(max_capacity, max(self._capacity * 2, min_capacity))
        new_buffer = xp_empty(self.xp, self._shape_with(new_capacity), dtype=self._buffer.dtype)

        # Copy existing data to new buffer
        total_samples = self._buff_read + self._buff_unread
        if total_samples > 0:
            start_idx = (self._tail - self._buff_read) % self._capacity
            stop_idx = (self._tail + self._buff_unread) % self._capacity
            if stop_idx > start_idx:
                # Data is contiguous
                new_buffer[self._idx(slice(None, total_samples))] = self._buffer[self._idx(slice(start_idx, stop_idx))]
            else:
                # Data wraps around. We write it in 2 parts.
                part1_len = self._capacity - start_idx
                part2_len = stop_idx
                new_buffer[self._idx(slice(None, part1_len))] = self._buffer[self._idx(slice(start_idx, None))]
                new_buffer[self._idx(slice(part1_len, part1_len + part2_len))] = self._buffer[
                    self._idx(slice(None, stop_idx))
                ]
            # self._buff_read stays the same
            self._tail = self._buff_read
            # self._buff_unread stays the same
            self._head = self._tail + self._buff_unread
        else:
            self._tail = 0
            self._head = 0

        self._buffer = new_buffer
        self._capacity = new_capacity
