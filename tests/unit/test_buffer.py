import pytest
import numpy as np
from ezmsg.sigproc.util.buffer import HybridBuffer


@pytest.fixture
def buffer_params():
    return {
        "array_namespace": np,
        "maxlen": 100,
        "other_shape": (2,),
        "dtype": np.float32,
    }


def test_initialization(buffer_params):
    buf = HybridBuffer(**buffer_params)
    assert buf.n_samples == 0
    assert buf._buffer.shape == (buffer_params["maxlen"], *buffer_params["other_shape"])
    assert buf._buffer.dtype == buffer_params["dtype"]


def test_add_and_get_simple(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    shape = (10, *buffer_params["other_shape"])
    data = np.arange(np.prod(shape), dtype=buffer_params["dtype"]).reshape(shape)
    buf.add_message(data)
    assert buf.n_samples == 10
    retrieved_data = buf.get_data(10)
    np.testing.assert_array_equal(data, retrieved_data)


def test_add_1d_message():
    buf = HybridBuffer(
        array_namespace=np,
        maxlen=10,
        other_shape=(1,),
        dtype=np.float32,
        update_strategy="immediate",
    )
    data = np.arange(5, dtype=np.float32)
    buf.add_message(data)
    assert buf.n_samples == 5
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
    shape = (10, *buffer_params["other_shape"])
    data1 = np.ones(shape)
    buf.add_message(data1)
    assert len(buf._deque) == 1
    assert buf._n_samples == 0  # Not synced yet

    shape2 = (5, *buffer_params["other_shape"])
    data2 = np.ones(shape2) * 2
    buf.add_message(data2)
    assert len(buf._deque) == 2
    assert buf._n_samples == 0

    retrieved = buf.get_data(15)
    assert len(buf._deque) == 0  # Synced now
    assert buf.n_samples == 15
    assert retrieved.shape == (15, *buffer_params["other_shape"])
    np.testing.assert_array_equal(retrieved[:10], data1)
    np.testing.assert_array_equal(retrieved[10:], data2)


def test_strategy_immediate(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    shape1 = (10, *buffer_params["other_shape"])
    data1 = np.ones(shape1)
    buf.add_message(data1)
    assert len(buf._deque) == 0
    assert buf.n_samples == 10

    shape2 = (5, *buffer_params["other_shape"])
    data2 = np.ones(shape2) * 2
    buf.add_message(data2)
    assert len(buf._deque) == 0
    assert buf.n_samples == 15

    retrieved = buf.get_data(15)
    np.testing.assert_array_equal(retrieved[:10], data1)
    np.testing.assert_array_equal(retrieved[10:], data2)


def test_strategy_threshold(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="threshold", threshold=15)
    shape1 = (10, *buffer_params["other_shape"])
    data1 = np.ones(shape1)
    buf.add_message(data1)
    assert len(buf._deque) == 1
    assert buf._n_samples == 0

    shape2 = (4, *buffer_params["other_shape"])  # Total = 14, under threshold
    data2 = np.ones(shape2)
    buf.add_message(data2)
    assert len(buf._deque) == 2
    assert buf._n_samples == 0

    shape3 = (1, *buffer_params["other_shape"])  # Total = 15, meets threshold
    data3 = np.ones(shape3)
    buf.add_message(data3)
    assert len(buf._deque) == 0
    assert buf.n_samples == 15


def test_buffer_wrap_around(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    # Fill the buffer completely
    buf.add_message(np.zeros((buffer_params["maxlen"], *buffer_params["other_shape"])))
    assert buf._head == 0
    assert buf.n_samples == 100

    # Add more data to cause a wrap
    shape = (10, *buffer_params["other_shape"])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    buf.add_message(data)
    assert buf._head == 10
    assert buf.n_samples == 100

    retrieved = buf.get_data(10)
    np.testing.assert_array_equal(data, retrieved)

    # Check that the oldest data was overwritten
    full_buffer_data = buf.get_data(100)
    np.testing.assert_array_equal(full_buffer_data[-10:], data)
    assert np.all(full_buffer_data[:90] == 0)


def test_get_data_wrap_around(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    shape1 = (80, *buffer_params["other_shape"])
    buf.add_message(np.arange(np.prod(shape1), dtype=np.float32).reshape(shape1))

    shape2 = (40, *buffer_params["other_shape"])
    latest_data = np.arange(np.prod(shape2), dtype=np.float32).reshape(shape2) + 1000
    buf.add_message(latest_data)
    assert buf._head == 20

    retrieved = buf.get_data(30)
    expected = latest_data[10:]
    np.testing.assert_array_equal(retrieved, expected)


def test_overflow_single_message(buffer_params):
    buf = HybridBuffer(**buffer_params, update_strategy="immediate")
    shape = (200, *buffer_params["other_shape"])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    buf.add_message(data)
    assert buf.n_samples == 100
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
        "maxlen": 50,
        "other_shape": (3, 4),
        "dtype": np.int16,
    }
    buf = HybridBuffer(**params, update_strategy="immediate")
    shape = (10, *params["other_shape"])
    data = np.arange(np.prod(shape), dtype=params["dtype"]).reshape(shape)
    buf.add_message(data)
    assert buf.n_samples == 10
    assert buf.get_data(10).shape == shape
    np.testing.assert_array_equal(buf.get_data(10), data)


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
