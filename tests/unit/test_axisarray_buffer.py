import pytest
import numpy as np

from ezmsg.util.messages.axisarray import AxisArray, LinearAxis, CoordinateAxis
from ezmsg.sigproc.util.axisarray_buffer import HybridAxisArrayBuffer


@pytest.fixture
def linear_axis_message():
    def _create(samples=10, channels=2, fs=100.0, offset=0.0):
        shape = (samples, channels)
        dims = ["time", "ch"]
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        gain = 1.0 / fs if fs > 0 else 0
        axes = {
            "time": LinearAxis(gain=gain, offset=offset),
            "ch": CoordinateAxis(data=np.arange(channels).astype(str), dims=["ch"]),
        }
        return AxisArray(data, dims, axes=axes)

    return _create


@pytest.fixture
def coordinate_axis_message():
    def _create(samples=10, channels=2, start_time=0.0, interval=0.01):
        shape = (samples, channels)
        dims = ["time", "ch"]
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        timestamps = np.arange(samples) * interval + start_time
        axes = {
            "time": CoordinateAxis(data=timestamps, dims=["time"]),
            "ch": CoordinateAxis(data=np.arange(channels).astype(str), dims=["ch"]),
        }
        return AxisArray(data, dims, axes=axes)

    return _create


def test_deferred_initialization_linear(linear_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0)  # 1 second buffer
    assert buf.n_unread == 0
    assert buf._data_buffer is None
    assert buf._axis_offset == 0.0
    assert buf._axis_buffer is None
    assert buf._template_msg is None

    msg = linear_axis_message(fs=100.0)
    buf.add_message(msg)

    assert buf.n_unread == 10
    assert buf._data_buffer is not None
    assert buf._data_buffer._maxlen == 100  # 1.0s * 100Hz
    assert buf._axis_offset == 0.09  # First timestamp is 0.0, last is 0.09
    assert buf._axis_buffer is None  # Still not initialized for LinearAxis
    assert buf._template_msg is not None and buf._template_msg.dims == ["time", "ch"]


def test_deferred_initialization_coordinate(coordinate_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0)
    msg = coordinate_axis_message(samples=10, interval=0.01)  # Effective fs = 100Hz
    buf.add_message(msg)

    assert buf.n_unread == 10
    assert buf._data_buffer is not None
    assert buf._data_buffer._maxlen == 100
    assert buf._axis_buffer is not None
    assert buf._axis_buffer._maxlen == 100


def test_add_and_get_linear(linear_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    msg1 = linear_axis_message(samples=10, fs=100.0, offset=0.0)
    buf.add_message(msg1)

    msg2 = linear_axis_message(samples=10, fs=100.0, offset=0.1)
    buf.add_message(msg2)

    assert buf.n_unread == 20
    retrieved_msg = buf.get_data(15)
    assert retrieved_msg.shape == (15, 2)
    assert retrieved_msg.dims == msg1.dims
    # Last sample of msg2 is at 0.1 + 9*0.01 = 0.19. Total unread was 20.
    # Offset of oldest sample = 0.19 - (20-1)*0.01 = 0.0
    assert retrieved_msg.axes["time"].offset == pytest.approx(0.0)
    expected_data = np.concatenate([msg1.data, msg2.data[:5]])
    np.testing.assert_array_equal(retrieved_msg.data, expected_data)

    # Check that the buffer now has 5 samples left
    assert buf.n_unread == 5
    remaining_msg = buf.get_data()
    np.testing.assert_array_equal(remaining_msg.data, msg2.data[5:])


def test_get_all_data_default(linear_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0)
    msg1 = linear_axis_message(samples=10)
    msg2 = linear_axis_message(samples=15)
    buf.add_message(msg1)
    buf.add_message(msg2)

    retrieved = buf.get_data()
    assert retrieved.shape[0] == 25
    assert buf.n_unread == 0


def test_add_and_get_coordinate(coordinate_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    msg1 = coordinate_axis_message(samples=10, start_time=0.0)
    buf.add_message(msg1)

    msg2 = coordinate_axis_message(samples=10, start_time=0.1)
    buf.add_message(msg2)

    assert buf.n_unread == 20
    retrieved_msg = buf.get_data(15)
    assert retrieved_msg.shape == (15, 2)
    assert retrieved_msg.dims == msg1.dims

    expected_data = np.concatenate([msg1.data, msg2.data[:5]])
    np.testing.assert_array_equal(retrieved_msg.data, expected_data)

    expected_times = np.concatenate(
        [msg1.axes["time"].data, msg2.axes["time"].data[:5]]
    )
    np.testing.assert_allclose(retrieved_msg.axes["time"].data, expected_times)

    assert buf.n_unread == 5


def test_type_mismatch_error(linear_axis_message, coordinate_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0)
    buf.add_message(linear_axis_message())
    with pytest.raises(TypeError):
        buf.add_message(coordinate_axis_message())
