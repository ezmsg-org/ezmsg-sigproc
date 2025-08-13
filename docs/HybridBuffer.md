## HybridBuffer

The HybridBuffer is a stateful, FIFO buffer that combines a deque for fast appends with a contiguous circular buffer for efficient, advancing reads. The synchronization between the deque and the circular buffer can be immediate, upon threshold reaching, or on demand, allowing for flexible data management strategies.

This buffer is designed to be agnostic to the array library used (e.g., NumPy, CuPy, PyTorch) via the Python Array API standard.

### Basic Reading and Writing Behaviour

The following diagram illustrates the states of the HybridBuffer across data additions and reads when the sync mode is set to `"on_demand"`:

![HybridBuffer Basic States](img/HybridBufferBasic.svg)

A. In the initial state, the buffer is empty, with no data in either the deque or the circular buffer.
   * deq_len=0; available=0, tell=0

B. After adding 4 samples, the deque contains the new data, but the circular buffer is still empty.
   * deq_len=4; available=4, tell=0

C. After adding 4 more samples, the deque now has 2 messages, each with 4 samples, and the circular buffer remains untouched.
   * deq_len=8; available=8, tell=0
   * Note: The deque is not synchronized with the circular buffer yet, so the circular buffer still has no data.

D. Upon the request to read 4 samples, the buffer synchronizes, copying ALL data from the deque to the circular buffer and the deque is cleared. Then our read index is advanced by 4 samples, leaving the circular buffer with 4 unread samples.
   * deq_len=0; available=4, tell=4

Not shown. After adding 10 samples, the deque contains the new data, and the circular buffer is unchanged from C.
   * deq_len=10; available=14, tell=4

E. After reading 1 more sample, the buffer synchronizes again, copying the data from the deque to the circular buffer. This time, the new data wraps around the circular buffer and overwrites 2 samples that were previously read, which decreases our tell by 2 to 2. Then we read a sample which increases tell to 3.
   * deq_len=0; available=13, tell=3

### Overflow Behaviour

TODO: Diagrams to describe "overwrite" behaviour.
TODO: Describe other behaviours 


### Advanced Pointer Manipulation

TODO: read vs peek vs seek
TODO: seek(-tell())
TODO: Upon read, if flush would cause overflow, then read first then flush then read again.
