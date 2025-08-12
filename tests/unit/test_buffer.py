import pytest
import numpy as np
from ezmsg.sigproc.util.buffer import HybridBuffer


@pytest.fixture
def buffer_params():
    return {
        "array_namespace": np,
        "capacity": 100,
        "other_shape": (2,),
        "dtype": np.float32,
    }


def test_initialization(buffer_params):
    buf = HybridBuffer(**buffer_params)
    assert buf.n_unread == 0
    assert buf._buffer.shape == (
        buffer_params["capacity"],
        *buffer_params["other_shape"],
    )
    assert buf._buffer.dtype == buffer_params["dtype"]


def test_add_and_get_simple(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    shape = (10, *buffer_params["other_shape"])
    data = np.arange(np.prod(shape), dtype=buffer_params["dtype"]).reshape(shape)
    buf.add_message(data)
    assert buf.n_unread == 10
    retrieved_data = buf.get_data(10)
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
    buf.add_message(data)
    assert buf.n_unread == 5
    retrieved = buf.get_data(5)
    assert retrieved.shape == (5, 1)
    np.testing.assert_array_equal(data, retrieved.squeeze())


def test_get_data_raises_error(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    data = np.zeros((10, *buffer_params["other_shape"]))
    buf.add_message(data)
    with pytest.raises(ValueError):
        buf.get_data(11)


def test_add_raises_error_on_shape(buffer_params):
    buf = HybridBuffer(**buffer_params)
    wrong_shape = (10, *[d + 1 for d in buffer_params["other_shape"]])
    data = np.zeros(wrong_shape)
    with pytest.raises(ValueError):
        buf.add_message(data)


def test_strategy_on_demand(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="on_demand")

    n_write_1 = 10
    shape = (n_write_1, *buffer_params["other_shape"])
    data1 = np.ones(shape)
    buf.add_message(data1)
    assert len(buf._deque) == 1
    assert buf._buff_unread == 0  # Not synced yet
    assert buf.n_unread == n_write_1

    n_write_2 = 5
    shape2 = (n_write_2, *buffer_params["other_shape"])
    data2 = np.ones(shape2) * 2
    buf.add_message(data2)
    assert len(buf._deque) == 2
    assert buf._buff_unread == 0
    assert buf.n_unread == n_write_1 + n_write_2

    n_read_1 = 7
    n_read_2 = (n_write_1 + n_write_2) - n_read_1
    retrieved = buf.get_data(n_read_1)
    assert len(buf._deque) == 0  # Synced now
    assert buf.n_unread == n_read_2
    assert retrieved.shape == (n_read_1, *buffer_params["other_shape"])
    np.testing.assert_array_equal(retrieved, data1[:n_read_1])

    retrieved = buf.get_data()  # Get all remaining
    assert buf.n_unread == 0
    assert retrieved.shape == (n_read_2, *buffer_params["other_shape"])
    np.testing.assert_array_equal(retrieved[: (n_write_1 - n_read_1)], data1[n_read_1:])
    np.testing.assert_array_equal(retrieved[(n_write_1 - n_read_1) :], data2)


def test_strategy_immediate(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")

    n_write_1 = 10
    shape1 = (n_write_1, *buffer_params["other_shape"])
    data1 = np.ones(shape1)
    buf.add_message(data1)
    assert len(buf._deque) == 0
    assert buf.n_unread == n_write_1

    n_write_2 = 5
    shape2 = (n_write_2, *buffer_params["other_shape"])
    data2 = np.ones(shape2) * 2
    buf.add_message(data2)
    assert len(buf._deque) == 0
    assert buf.n_unread == (n_write_1 + n_write_2)

    retrieved = buf.get_data()
    np.testing.assert_array_equal(retrieved[:n_write_1], data1)
    np.testing.assert_array_equal(retrieved[n_write_1:], data2)


def test_strategy_threshold(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="threshold", threshold=15)

    shape1 = (10, *buffer_params["other_shape"])
    data1 = np.ones(shape1)
    buf.add_message(data1)
    assert len(buf._deque) == 1
    assert buf.n_unread == 10
    assert buf._buff_unread == 0

    shape2 = (4, *buffer_params["other_shape"])  # Total = 14, under threshold
    data2 = np.ones(shape2)
    buf.add_message(data2)
    assert len(buf._deque) == 2
    assert buf.n_unread == 14
    assert buf._buff_unread == 0

    shape3 = (1, *buffer_params["other_shape"])  # Total = 15, meets threshold
    data3 = np.ones(shape3)
    buf.add_message(data3)
    assert len(buf._deque) == 0
    assert buf.n_unread == 15
    assert buf._buff_unread == 15


def test_buffer_wrap_around(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    # Fill the buffer completely
    buf.add_message(
        np.zeros((buffer_params["capacity"], *buffer_params["other_shape"]))
    )
    assert buf._head == 0
    assert buf._tail == 0
    assert buf.n_unread == 100

    # Add more data to cause a wrap
    shape = (10, *buffer_params["other_shape"])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    buf.add_message(data)
    assert buf._head == 10
    assert buf._tail == 10
    assert buf.n_unread == 100

    retrieved = buf.get_data(10)
    assert np.all(retrieved == 0)

    # Check that the oldest data was overwritten
    reamining_buffer_data = buf.get_data()
    assert reamining_buffer_data.shape == (90, *buffer_params["other_shape"])
    np.testing.assert_array_equal(reamining_buffer_data[-10:], data)
    assert np.all(reamining_buffer_data[:80] == 0)


def test_get_data_wrap_around(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")

    shape1 = (80, *buffer_params["other_shape"])
    first_data = np.arange(np.prod(shape1), dtype=np.float32).reshape(shape1)
    buf.add_message(first_data)
    assert buf._head == 80
    assert buf._tail == 0

    shape2 = (40, *buffer_params["other_shape"])
    latest_data = np.arange(np.prod(shape2), dtype=np.float32).reshape(shape2) + 1000
    buf.add_message(latest_data)
    assert buf._head == 20
    assert buf._tail == 20

    retrieved = buf.get_data()
    assert buf.n_unread == 0
    assert retrieved.shape == (100, *buffer_params["other_shape"])
    np.testing.assert_array_equal(retrieved[:60], first_data[20:])
    np.testing.assert_array_equal(retrieved[60:], latest_data)


def test_overflow_single_message(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    shape = (200, *buffer_params["other_shape"])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    buf.add_message(data)
    assert buf.n_unread == 100
    retrieved = buf.get_data(100)
    np.testing.assert_array_equal(data[-100:], retrieved)


def test_get_zero_samples(buffer_params):
    buf = HybridBuffer(**buffer_params)
    data = buf.get_data(0)
    assert data.shape == (0, *buffer_params["other_shape"])

    buf.add_message(np.ones((10, *buffer_params["other_shape"])))
    data = buf.get_data(0)
    assert data.shape == (0, *buffer_params["other_shape"])


def test_nd_tensor():
    params = {
        "array_namespace": np,
        "capacity": 50,
        "other_shape": (3, 4),
        "dtype": np.int16,
    }
    buf = HybridBuffer(**params, update_strategy="immediate")
    shape = (10, *params["other_shape"])
    data = np.arange(np.prod(shape), dtype=params["dtype"]).reshape(shape)
    buf.add_message(data)
    assert buf.n_unread == 10
    retrieved = buf.get_data(10)
    assert retrieved.shape == shape
    np.testing.assert_array_equal(retrieved, data)


def test_get_data_default_all(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="on_demand")
    shape1 = (10, *buffer_params["other_shape"])
    data1 = np.ones(shape1)
    buf.add_message(data1)

    shape2 = (15, *buffer_params["other_shape"])
    data2 = np.ones(shape2) * 2
    buf.add_message(data2)

    # Should trigger sync and get all 25 samples
    retrieved = buf.get_data()
    assert retrieved.shape[0] == 25

    expected = np.concatenate((data1, data2), axis=0)
    np.testing.assert_array_equal(retrieved, expected)


def test_interleaved_read_write(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    # Add 50
    data1 = np.arange(50 * 2).reshape(50, 2)
    buf.add_message(data1)
    assert buf.n_unread == 50

    # Get 20
    read1 = buf.get_data(20)
    np.testing.assert_array_equal(read1, data1[:20])
    assert buf.n_unread == 30
    assert buf._tail == 20

    # Add 30
    data2 = np.arange(30 * 2).reshape(30, 2) + 1000
    buf.add_message(data2)
    assert buf.n_unread == 60  # 30 remaining + 30 new
    assert buf._head == 80  # 50 + 30

    # Get 60 (all remaining)
    read2 = buf.get_data(60)
    assert buf.n_unread == 0
    expected_data = np.concatenate([data1[20:], data2])
    np.testing.assert_array_equal(read2, expected_data)


def test_read_to_empty(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    data = np.arange(30 * 2).reshape(30, 2)
    buf.add_message(data)
    assert buf.n_unread == 30

    _ = buf.get_data(30)
    assert buf.n_unread == 0
    assert buf._tail == 30

    # Reading again should return empty array
    empty_read = buf.get_data()
    assert empty_read.shape[0] == 0


def test_read_operation_wraps(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    # Add 80 samples, tail is at 0, head is at 80
    data1 = np.arange(80 * 2).reshape(80, 2)
    buf.add_message(data1)

    # Read 60 samples, tail is at 60, head is at 80
    buf.get_data(60)
    assert buf.n_unread == 20
    assert buf._tail == 60

    # Add 40 samples. This will wrap the head around to 20.
    data2 = np.arange(40 * 2).reshape(40, 2) + 1000
    buf.add_message(data2)
    assert buf.n_unread == 60  # 20 remaining + 40 new
    assert buf._head == 20

    # Read 30 samples. This will force the read to wrap.
    # It will read 20 from data1 (60->80) and 10 from data2 (80->90)
    read_data = buf.get_data(30)
    assert read_data.shape[0] == 30
    assert buf.n_unread == 30
    assert buf._tail == 90  # 60 + 30

    expected = np.concatenate([data1[60:], data2[:10]])
    np.testing.assert_array_equal(read_data, expected)


def test_peek_simple(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    data = np.arange(20 * 2).reshape(20, 2)
    buf.add_message(data)

    peeked_data = buf.peek(10)
    np.testing.assert_array_equal(peeked_data, data[:10])

    # Assert that state has not changed
    assert buf.n_unread == 20
    assert buf._tail == 0

    # Get the data to prove it was still there
    retrieved_data = buf.get_data(10)
    np.testing.assert_array_equal(retrieved_data, data[:10])
    assert buf.n_unread == 10


def test_skip_simple(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    data = np.arange(20 * 2).reshape(20, 2)
    buf.add_message(data)

    skipped = buf.skip(10)
    assert skipped == 10
    assert buf.n_unread == 10
    assert buf._tail == 10

    retrieved_data = buf.get_data()
    np.testing.assert_array_equal(retrieved_data, data[10:])


def test_peek_and_skip(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    data = np.arange(20 * 2).reshape(20, 2)
    buf.add_message(data)

    peeked = buf.peek(5)
    skipped = buf.skip(5)
    assert skipped == 5
    np.testing.assert_array_equal(peeked, data[:5])

    retrieved = buf.get_data(5)
    np.testing.assert_array_equal(retrieved, data[5:10])


def test_peek_padded_simple(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    data1 = np.arange(20 * 2).reshape(20, 2)
    buf.add_message(data1)
    buf.skip(10)

    # 10 samples of history, 10 samples unread
    data, num_padded = buf.peek_padded(n_samples=5, padding=5)
    assert num_padded == 5
    assert data.shape[0] == 10
    expected = data1[5:15]
    np.testing.assert_array_equal(data, expected)


def test_peek_padded_no_history(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    data1 = np.arange(20 * 2).reshape(20, 2)
    buf.add_message(data1)

    # 0 samples of history, 20 samples unread
    data, num_padded = buf.peek_padded(n_samples=5, padding=5)
    assert num_padded == 0
    assert data.shape[0] == 5
    np.testing.assert_array_equal(data, data1[:5])


def test_peek_padded_more_padding_than_history(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    data1 = np.arange(20 * 2).reshape(20, 2)
    buf.add_message(data1)
    buf.skip(10)

    # 10 samples of history, 10 samples unread
    data, num_padded = buf.peek_padded(n_samples=5, padding=15)
    assert num_padded == 10
    assert data.shape[0] == 15
    expected = data1[:15]
    np.testing.assert_array_equal(data, expected)


def test_get_data_padded(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    data1 = np.arange(20 * 2).reshape(20, 2)
    buf.add_message(data1)
    buf.skip(10)

    # 10 samples of history, 10 samples unread
    data, num_padded = buf.get_data_padded(n_samples=5, padding=5)
    assert num_padded == 5
    assert data.shape[0] == 10
    expected = data1[5:15]
    np.testing.assert_array_equal(data, expected)

    # The 5 unread samples should be gone
    assert buf.n_unread == 5
    remaining_data = buf.get_data()
    np.testing.assert_array_equal(remaining_data, data1[15:])
