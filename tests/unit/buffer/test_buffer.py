import numpy as np
import pytest

from ezmsg.sigproc.util.buffer import HybridBuffer


@pytest.fixture
def buffer_params():
    return {
        "array_namespace": np,
        "capacity": 100,
        "other_shape": (2,),
        "dtype": np.float32,
        "update_strategy": "immediate",
        "threshold": 0,
        "overflow_strategy": "warn-overwrite",
        "max_size": 1024**3,  # 1 GB
    }


def test_initialization(buffer_params):
    buf = HybridBuffer(**buffer_params)
    assert buf.available() == 0
    assert not buf.is_full()
    assert buf.is_empty()
    assert buf.capacity == buffer_params["capacity"]
    assert buf._update_strategy == buffer_params["update_strategy"]
    assert buf._threshold == buffer_params["threshold"]
    assert buf._overflow_strategy == buffer_params["overflow_strategy"]
    assert buf._max_size == buffer_params["max_size"]
    assert buf._buffer.shape[1:] == buffer_params["other_shape"]
    assert buf._buffer.dtype == buffer_params["dtype"]


def test_add_and_get_simple(buffer_params):
    buf = HybridBuffer(**buffer_params)
    shape = (10, *buffer_params["other_shape"])
    data = np.arange(np.prod(shape), dtype=buffer_params["dtype"]).reshape(shape)
    buf.write(data)
    assert buf.available() == 10
    retrieved_data = buf.read(10)
    np.testing.assert_array_equal(data, retrieved_data)


def test_add_1d_message():
    buf = HybridBuffer(
        array_namespace=np,
        capacity=10,
        other_shape=(1,),
        dtype=np.float32,
        update_strategy="immediate",
    )
    data = np.arange(5, dtype=np.float32)
    buf.write(data)
    assert buf.available() == 5
    retrieved = buf.read(5)
    assert retrieved.shape == (5, 1)
    np.testing.assert_array_equal(data, retrieved.squeeze())


def test_get_data_raises_error(buffer_params):
    buf = HybridBuffer(**buffer_params)
    data = np.zeros((10, *buffer_params["other_shape"]))
    buf.write(data)
    with pytest.raises(ValueError):
        buf.read(11)


def test_add_raises_error_on_shape(buffer_params):
    buf = HybridBuffer(**buffer_params)
    wrong_shape = (10, *[d + 1 for d in buffer_params["other_shape"]])
    data = np.zeros(wrong_shape)
    with pytest.raises(ValueError):
        buf.write(data)


def test_strategy_on_demand(buffer_params):
    buf = HybridBuffer(**{**buffer_params, "update_strategy": "on_demand"})

    n_write_1 = 10
    shape = (n_write_1, *buffer_params["other_shape"])
    data1 = np.ones(shape)
    buf.write(data1)
    assert len(buf._deque) == 1
    assert buf._buff_unread == 0  # Not synced yet
    assert buf.available() == n_write_1

    n_write_2 = 5
    shape2 = (n_write_2, *buffer_params["other_shape"])
    data2 = np.ones(shape2) * 2
    buf.write(data2)
    assert len(buf._deque) == 2
    assert buf._buff_unread == 0
    assert buf.available() == n_write_1 + n_write_2

    n_read_1 = 7
    n_read_2 = (n_write_1 + n_write_2) - n_read_1
    retrieved = buf.read(n_read_1)
    assert len(buf._deque) == 0  # Synced now
    assert buf.available() == n_read_2
    assert retrieved.shape == (n_read_1, *buffer_params["other_shape"])
    np.testing.assert_array_equal(retrieved, data1[:n_read_1])

    retrieved = buf.read()  # Get all remaining
    assert buf.available() == 0
    assert retrieved.shape == (n_read_2, *buffer_params["other_shape"])
    np.testing.assert_array_equal(retrieved[: (n_write_1 - n_read_1)], data1[n_read_1:])
    np.testing.assert_array_equal(retrieved[(n_write_1 - n_read_1) :], data2)


def test_strategy_immediate(buffer_params):
    buf = HybridBuffer(**buffer_params)

    n_write_1 = 10
    shape1 = (n_write_1, *buffer_params["other_shape"])
    data1 = np.ones(shape1)
    buf.write(data1)
    assert len(buf._deque) == 0
    assert buf._buff_unread == n_write_1
    assert buf.available() == n_write_1

    n_write_2 = 5
    shape2 = (n_write_2, *buffer_params["other_shape"])
    data2 = np.ones(shape2) * 2
    buf.write(data2)
    assert len(buf._deque) == 0
    assert buf._buff_unread == (n_write_1 + n_write_2)
    assert buf.available() == (n_write_1 + n_write_2)

    retrieved = buf.read()
    np.testing.assert_array_equal(retrieved[:n_write_1], data1)
    np.testing.assert_array_equal(retrieved[n_write_1:], data2)


def test_strategy_threshold(buffer_params):
    new_params = {**buffer_params, "update_strategy": "threshold", "threshold": 15}
    buf = HybridBuffer(**new_params)

    shape1 = (10, *buffer_params["other_shape"])
    data1 = np.ones(shape1)
    buf.write(data1)
    assert len(buf._deque) == 1
    assert buf.available() == 10
    assert buf._buff_unread == 0

    shape2 = (4, *buffer_params["other_shape"])  # Total = 14, under threshold
    data2 = np.ones(shape2)
    buf.write(data2)
    assert len(buf._deque) == 2
    assert buf.available() == 14
    assert buf._buff_unread == 0

    shape3 = (1, *buffer_params["other_shape"])  # Total = 15, meets threshold
    data3 = np.ones(shape3)
    buf.write(data3)
    assert len(buf._deque) == 0
    assert buf.available() == 15
    assert buf._buff_unread == 15


def test_buffer_overflow_warn_overwrite(buffer_params):
    buf = HybridBuffer(**buffer_params)
    cap = buffer_params["capacity"]
    # Fill the buffer completely
    buf.write(np.zeros((cap, *buffer_params["other_shape"])))
    assert buf._head == 0
    assert buf._tail == 0
    assert buf.available() == cap

    # Add more data to cause a wrap + overflow
    shape = (10, *buffer_params["other_shape"])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with pytest.warns(RuntimeWarning):
        buf.write(data)
    assert buf._head == 10
    assert buf._tail == 10  # Tail moves forward with head during overflow
    assert buf.available() == cap

    retrieved = buf.read(10)
    assert np.all(retrieved == 0)

    # Check that the oldest data was overwritten
    reamining_buffer_data = buf.read()
    assert reamining_buffer_data.shape == (cap - 10, *buffer_params["other_shape"])
    # np.testing.assert_array_equal(reamining_buffer_data[-10:], data)
    assert np.all(reamining_buffer_data[: cap - 20] == 0)


def test_read_wrap_around(buffer_params):
    buf = HybridBuffer(**buffer_params)

    shape1 = (80, *buffer_params["other_shape"])
    first_data = np.arange(np.prod(shape1), dtype=np.float32).reshape(shape1)
    buf.write(first_data)
    assert buf._head == 80
    assert buf._tail == 0

    shape2 = (40, *buffer_params["other_shape"])
    latest_data = np.arange(np.prod(shape2), dtype=np.float32).reshape(shape2) + 1000
    with pytest.warns(RuntimeWarning):
        buf.write(latest_data)
    assert buf._head == 20
    assert buf._tail == 20  # Tail moves forward with head during overflow

    retrieved = buf.read()
    assert buf.available() == 0
    assert retrieved.shape == (100, *buffer_params["other_shape"])
    np.testing.assert_array_equal(retrieved[:60], first_data[20:])
    np.testing.assert_array_equal(retrieved[60:], latest_data)


def test_overflow_single_message(buffer_params):
    buf = HybridBuffer(**buffer_params)
    shape = (200, *buffer_params["other_shape"])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with pytest.warns(RuntimeWarning):
        buf.write(data)
    assert buf.available() == 100
    retrieved = buf.read()
    np.testing.assert_array_equal(data[-100:], retrieved)


def test_get_zero_samples(buffer_params):
    buf = HybridBuffer(**buffer_params)
    data = buf.read(0)
    assert data.shape == (0, *buffer_params["other_shape"])

    buf.write(np.ones((10, *buffer_params["other_shape"])))
    data = buf.read(0)
    assert data.shape == (0, *buffer_params["other_shape"])


def test_nd_tensor():
    params = {
        "array_namespace": np,
        "capacity": 50,
        "other_shape": (3, 4),
        "dtype": np.int16,
    }
    buf = HybridBuffer(**params)
    shape = (10, *params["other_shape"])
    data = np.arange(np.prod(shape), dtype=params["dtype"]).reshape(shape)
    buf.write(data)
    assert buf.available() == 10
    retrieved = buf.read(10)
    assert retrieved.shape == shape
    np.testing.assert_array_equal(retrieved, data)


def test_get_data_default_all(buffer_params):
    buf = HybridBuffer(**{**buffer_params, "update_strategy": "on_demand"})
    shape1 = (10, *buffer_params["other_shape"])
    data1 = np.ones(shape1)
    buf.write(data1)

    shape2 = (15, *buffer_params["other_shape"])
    data2 = np.ones(shape2) * 2
    buf.write(data2)

    # Should trigger sync and get all 25 samples
    retrieved = buf.read()
    assert retrieved.shape[0] == 25

    expected = np.concatenate((data1, data2), axis=0)
    np.testing.assert_array_equal(retrieved, expected)


def test_interleaved_read_write(buffer_params):
    buf = HybridBuffer(**buffer_params)
    # Add 50
    data1 = np.arange(50 * 2).reshape(50, 2)
    buf.write(data1)
    assert buf.available() == 50

    # Get 20
    read1 = buf.read(20)
    np.testing.assert_array_equal(read1, data1[:20])
    assert buf.available() == 30
    assert buf._tail == 20

    # Add 30
    data2 = np.arange(30 * 2).reshape(30, 2) + 1000
    buf.write(data2)
    assert buf.available() == 60  # 30 remaining + 30 new
    assert buf._head == 80  # 50 + 30

    # Get 60 (all remaining)
    read2 = buf.read(60)
    assert buf.available() == 0
    expected_data = np.concatenate([data1[20:], data2])
    np.testing.assert_array_equal(read2, expected_data)


def test_read_to_empty(buffer_params):
    buf = HybridBuffer(**buffer_params)
    data = np.arange(30 * 2).reshape(30, 2)
    buf.write(data)
    assert buf.available() == 30

    _ = buf.read(30)
    assert buf.available() == 0
    assert buf._tail == 30

    # Reading again should return empty array
    empty_read = buf.read()
    assert empty_read.shape[0] == 0


def test_read_operation_wraps(buffer_params):
    buf = HybridBuffer(**buffer_params)
    # Add 80 samples, tail is at 0, head is at 80
    data1 = np.arange(80 * 2).reshape(80, 2)
    buf.write(data1)

    # Read 60 samples, tail is at 60, head is at 80
    buf.read(60)
    assert buf.available() == 20
    assert buf._tail == 60

    # Add 40 samples. This will wrap the head around to 20.
    data2 = np.arange(40 * 2).reshape(40, 2) + 1000
    buf.write(data2)
    assert buf.available() == 60  # 20 remaining + 40 new
    assert buf._head == 20

    # Read 30 samples. This will force the read to wrap.
    # It will read 20 from data1 (60->80) and 10 from data2 (80->90)
    read_data = buf.read(30)
    assert read_data.shape[0] == 30
    assert buf.available() == 30
    assert buf._tail == 90  # 60 + 30

    expected = np.concatenate([data1[60:], data2[:10]])
    np.testing.assert_array_equal(read_data, expected)


def test_peek_simple(buffer_params):
    buf = HybridBuffer(**buffer_params)
    data = np.arange(20 * 2).reshape(20, 2)
    buf.write(data)

    peeked_data = buf.peek(10)
    np.testing.assert_array_equal(peeked_data, data[:10])

    # Assert that state has not changed
    assert buf.available() == 20
    assert buf._tail == 0

    # Get the data to prove it was still there
    retrieved_data = buf.read(10)
    np.testing.assert_array_equal(retrieved_data, data[:10])
    assert buf.available() == 10


def test_seek_simple(buffer_params):
    buf = HybridBuffer(**buffer_params)
    data = np.arange(20 * 2).reshape(20, 2)
    buf.write(data)

    skipped = buf.seek(10)
    assert skipped == 10
    assert buf.available() == 10
    assert buf._tail == 10

    retrieved_data = buf.read()
    np.testing.assert_array_equal(retrieved_data, data[10:])


def test_peek_and_skip(buffer_params):
    buf = HybridBuffer(**buffer_params)
    data = np.arange(20 * 2).reshape(20, 2)
    buf.write(data)

    peeked = buf.peek(5)
    np.testing.assert_array_equal(peeked, data[:5])

    peeked_again = buf.peek(5)
    np.testing.assert_array_equal(peeked_again, data[:5])

    skipped = buf.seek(5)
    assert skipped == 5

    retrieved = buf.read(5)
    np.testing.assert_array_equal(retrieved, data[5:10])


def test_tell(buffer_params):
    buf = HybridBuffer(**buffer_params)

    # 1. Initially empty. tell() should return 0.
    assert buf.tell() == 0

    # 2. Add 50 samples. tell() should return 0.
    buf.write(np.zeros((50, 2)))
    assert buf.tell() == 0

    # 3. Read 20 samples. tell() should return 20.
    buf.read(20)
    assert buf.tell() == 20

    # Read another 10 samples. tell() should return 30.
    buf.read(10)
    assert buf.tell() == 30

    # Read remaining 20 samples. tell() should return 50.
    buf.read(20)
    assert buf.tell() == 50

    # Try to read more than available. Should still return 50.
    with pytest.raises(ValueError):
        buf.read(1)
    assert buf.tell() == 50

    # 4. Add 80 samples -> overwrite the first 30.
    # tell() should return 20: 50 - 30
    final_msg = np.zeros((80, 2))
    buf.write(final_msg)
    assert buf.tell() == 20


def test_peek_at(buffer_params):
    buf = HybridBuffer(**{**buffer_params, "update_strategy": "on_demand"})
    # Add 50 samples in 5 blocks
    for i in range(5):
        buf.write(np.ones((10, 2)) * i)

    # Peek at a value in the buffer before flushing
    with pytest.raises(IndexError):
        buf.peek_at(50)

    # Read some data to cause a flush
    _ = buf.read(1)

    # Test peeking at various locations
    np.testing.assert_array_equal(buf.peek_at(0), np.ones((1, 2)) * 0)
    np.testing.assert_array_equal(buf.peek_at(10), np.ones((1, 2)) * 1)
    np.testing.assert_array_equal(buf.peek_at(20), np.ones((1, 2)) * 2)
    np.testing.assert_array_equal(buf.peek_at(30), np.ones((1, 2)) * 3)
    np.testing.assert_array_equal(buf.peek_at(48), np.ones((1, 2)) * 4)

    # Test peeking out of bounds
    with pytest.raises(IndexError):
        buf.peek_at(49)
