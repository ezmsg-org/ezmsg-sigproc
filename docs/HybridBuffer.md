## HybridBuffer

The HybridBuffer is a stateful, FIFO buffer that combines a deque for fast appends with a contiguous circular buffer for efficient, advancing reads. The synchronization between the deque and the circular buffer can be immediate, upon threshold reaching, or on demand, allowing for flexible data management strategies.

This buffer is designed to be agnostic to the array library used (e.g., NumPy, CuPy, PyTorch) via the Python Array API standard.

### Basic Reading and Writing Behaviour

The following diagram illustrates the states of the HybridBuffer across data writes and reads when `update_strategy="on_demand"`:

![HybridBuffer Basic States](img/HybridBufferBasic.svg)

**Figure 1**

A. In the initial state, the buffer is empty, with no data in either the deque or the circular buffer.
   * deq_len=0; available=0, tell=0

B. After we `write()` 4 samples, the deque contains the new data, but the circular buffer is still empty.
   * deq_len=4; available=4, tell=0

C. After we `write()` 4 more samples, the deque now has 2 messages, each with 4 samples, and the circular buffer remains untouched.
   * deq_len=8; available=8, tell=0

D. Panels D-F depict a single call to `read(4)` which is implemented as calls to other methods. If we don't have 4 unread samples in the circular buffer, but we do have >= 4 samples 'available' (i.e., including the deque), then a `flush()` is performed: the entirety of the data in the deque are copied to the circular buffer and the deque is cleared.
    * deq_len=0; available=8, tell=0
    * TODO: Currently `flush()` copies the data twice, once from the deque to a contiguous array, and then from that contiguous array to the circular buffer. This should be optimized to copy directly from the deque to the circular buffer.

E. Next we `peek(4)` which returns the first 4 samples from the circular buffer; the return value may be a view on the data if the data are contiguous in the circular buffer, or a copy if the data are not contiguous. Note that the tail (read pointer) does not advance with `peek()`.
    * deq_len=0; available=8, tell=0

F. Finally, we `seek(4)` to advance the tail.
   * deq_len=0; available=4, tell=4

G. We `write()` 4 more samples, which are appended to the deque, leaving the circular buffer unchanged from the previous step.
   * deq_len=4; available=8, tell=4

H. We then `read(4)` again. This time, a `flush()` is not triggered because we have enough unread samples in the circular buffer, but `peek(4)` and `seek(4)` are still called. The read pointer advances by 4, leaving 0 unread samples in the circular buffer and 4 in the deque.
   * deq_len=4; available=0, tell=8

Note: `peek(n)` and `seek(n)`, where `n` > `n_available` will raise an error. However, `peek(None)` will return all available samples without error, and `seek(None)` will advance the tail to the end of the available data.

### Overflow Behaviour

The criteria to trigger an overflow are as follows:
* the deque has more data than there is space in the circular buffer, where space is the combination of previously read samples and unwritten samples in the circular buffer.
* the caller triggers a flush either manually (`flush()`) or by requesting (via `read`, `peek`, or `seek`) more samples than are available in the circular buffer but not more than the total size of the available samples in the buffer + available samples in the deque.

![HybridBuffer Overflow Behaviour](img/HybridBufferOverflow.svg)

**Figure 2**

A. We start with a circular buffer that has been running for a while (it has wrapped around several times). At this particular moment, we have more data in the deque (12) than we have room in the buffer (8). The remaining figures describe what happens when `flush()` is called with different overflow strategies. The samples are labeled to make it easier to follow the flow of data.
   * deq_len=12; available=20, tell=1

B. "warn-overwrite": If the overflow_strategy is set to 'warn-overwrite', the HybridBuffer will log a warning and overwrite the oldest data in the circular buffer with the new data from the deque. Here, samples 'a-d' are lost.
   * deq_len=0; available=16, tell=0

C. "drop": As much as possible of the data from the deque are copied into the circular buffer, but remaining data are dropped. In this case, samples 'q-t' are lost.
   * deq_len=0; available=16, tell=0

D. "grow": The HybridBuffer will attempt to grow the circular buffer to the lesser of double its current size or the size required to accommodate all read + unread + deque data. If the buffer cannot grow (e.g., due to memory constraints; default max_size is 1GB), it will raise an error.
   * deq_len=0; available=20, tell=8

Additionally, one can configure the HybridBuffer overflow_strategy to 'raise', which will raise an error if there is insufficient space (empty or read samples) in the buffer to perform the flush.

There are a few mitigations to defer flushing to help prevent overflows:

* If the requested number of samples to read, peek, or seek is less than the number of unread samples in the circular buffer, then no flush is performed.
* Helper methods `peek_at(k, allow_flush=False)` (False is default), and `peek_last()` will retrieve the target sample from the buffer-OR-deque without flushing.
  * Be cautious relying on repeated calls to `peek_at(k, allow_flush=False)` as it scans over the items in the deque which can be slow.
* When calling `read(n)`, if a flush is necessary, and it will cause an overflow, and the overflow could be prevented with a pre-emptive read up to `n`, then it will do the read in 2 parts. First it will call `peek(n_unread_in_buffer)` and `seek(n_unread_in_buffer)` to read the unread samples in the circular buffer. Second, it will call `peek(n_remaining)` and `seek(n_remaining)` to trigger a flush -- which should no longer cause an overflow -- then read the remaining requested samples and stitch them together.

### Advanced Pointer Manipulation

The previous section describes how `read`, `peek`, `seek`, and `peek_at` function in normal use cases. It is also possible to call `seek` with a negative value, which will attempt to move the tail pointer backwards over previously-read (or previously sought-over) data by that many samples. `seek` returns the number of samples that were actually moved, which may be less than the requested value if there was insufficient room. Negative seeks can only rewind into previously read data, and positive seeks can only advance into unread data, possibly including data that gets flushed from the deque.

## HybridAxisBuffer

The `HybridAxisBuffer` carries the semantics of the `HybridBuffer` but it is designed to handle either a `LinearAxis` or a `CoordinateAxis`. Its `write` method expects an axis object and its `peek` and `read` methods return an axis, not just the data.

For a `LinearAxis`, the `HybridAxisBuffer` simply maintains the `gain`, the `offset`, and the 'number of samples available'. Since this does not store actual data, it has no capacity. If this object is intended to be synchronized with another `HybridBuffer`-using object that does have a capacity, then the other object should be manipulated first and then the number of samples actually moved should be used to call the `HybridAxisBuffer`'s methods, otherwise these objects will be out of sync.

For a `CoordinateAxis`, the `HybridAxisBuffer` maintains the `data` in a `HybridBuffer` and thus behaves like a `HybridBuffer` with respect to the capacity. The returned `CoordinateAxis` object might have its `.data` field as a view on the data in the buffer, so it should not be modified in place.

## HybridAxisArrayBuffer

This is a convenience class that combines the `HybridAxisBuffer` and `HybridBuffer` into a single object that can be used to manage both axis and data in a single object. This class is particularly useful when you need to manage both the axis information and the data samples together, as is the case for an `AxisArray` object. Its `write` method expects an `AxisArary` object and its `peek` and `read` methods return an `AxisArray` object. Note that the return object's `.data` field might be a view on the data in the buffer so it should not be modified in place. Similarly so for the `CoordinateAxis` data.
