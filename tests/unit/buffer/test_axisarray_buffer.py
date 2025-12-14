import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, LinearAxis

from ezmsg.sigproc.util.axisarray_buffer import HybridAxisArrayBuffer, HybridAxisBuffer


class TestHybridAxisBuffer:
    """Test suite for HybridAxisBuffer"""

    def test_uninitialized_state(self):
        """Test buffer state before initialization"""
        buf = HybridAxisBuffer(duration=1.0)

        assert buf.capacity == 0
        assert buf.available() == 0
        assert buf.is_empty() is True
        assert buf.is_full() is False
        assert buf.gain is None
        assert buf.final_value is None

    def test_linear_axis_initialization(self):
        """Test initialization with LinearAxis"""
        buf = HybridAxisBuffer(duration=1.0)

        # Create a LinearAxis with 1kHz sampling (gain=0.001)
        axis = LinearAxis(gain=0.001, offset=0.0)
        buf._initialize(axis)

        assert buf._linear_axis is not None
        assert buf._coords_buffer is None
        assert buf.capacity == 1000  # 1.0 sec / 0.001 gain
        assert buf.gain == 0.001
        assert buf.available() == 0

    def test_coordinate_axis_initialization(self):
        """Test initialization with CoordinateAxis"""
        buf = HybridAxisBuffer(duration=1.0)

        # Create CoordinateAxis with timestamps - note it needs dims parameter
        timestamps = np.linspace(0, 0.1, 101)  # 100 intervals, 0.001 gain
        axis = CoordinateAxis(data=timestamps, dims=["time"])
        buf._initialize(axis)

        assert buf._coords_buffer is not None
        assert buf._linear_axis is None
        assert buf.capacity == 1000  # 1.0 sec / 0.001 gain
        assert buf.gain == pytest.approx(0.001)
        assert buf.available() == 0

    def test_coordinate_axis_single_sample(self):
        """Test CoordinateAxis initialization with single sample"""
        buf = HybridAxisBuffer(duration=1.0)

        # Single timestamp should default to gain of 1.0
        axis = CoordinateAxis(data=np.array([0.0]), dims=["time"])
        buf._initialize(axis)

        assert buf.capacity == 1  # 1.0 sec / 1.0 gain
        assert buf.gain == 1.0

    def test_linear_axis_write_and_read(self):
        """Test writing and reading with LinearAxis"""
        buf = HybridAxisBuffer(duration=1.0)

        # Initialize with LinearAxis
        axis1 = LinearAxis(gain=0.001, offset=0.0)
        buf.write(axis1, n_samples=100)

        assert buf.available() == 100
        assert buf._linear_n_available == 100
        assert buf._linear_axis.offset == 0.0

        # Write more samples with different offset
        #  The expected offset for the next write was 0.1,
        #  but we provide 0.15, which causes the original
        #  samples write operation to be adjusted to 0.05
        axis2 = LinearAxis(gain=0.001, offset=0.15)
        buf.write(axis2, n_samples=50)

        assert buf.available() == 150
        # Offset should be adjusted to oldest sample
        assert buf._linear_axis.offset == pytest.approx(0.05)  # 0.15 - 100*0.001

        # Peek at the axis
        peeked_axis = buf.peek(50)
        assert isinstance(peeked_axis, LinearAxis)
        assert peeked_axis.offset == pytest.approx(0.05)
        assert peeked_axis.gain == 0.001

        # Seek forward
        sought = buf.seek(50)
        assert sought == 50
        assert buf.available() == 100
        assert buf._linear_axis.offset == pytest.approx(0.1)  # 0.05 + 50*0.001

    def test_coordinate_axis_write_and_read(self):
        """Test writing and reading with CoordinateAxis"""
        buf = HybridAxisBuffer(duration=1.0, update_strategy="immediate")

        # Initialize with CoordinateAxis
        timestamps1 = np.linspace(0, 0.099, 100)
        axis1 = CoordinateAxis(data=timestamps1, dims=["time"])
        buf.write(axis1, n_samples=100)

        assert buf.available() == 100

        # Write more samples
        timestamps2 = np.linspace(0.1, 0.149, 50)
        axis2 = CoordinateAxis(data=timestamps2, dims=["time"])
        buf.write(axis2, n_samples=50)

        assert buf.available() == 150

        # Peek at the axis
        peeked_axis = buf.peek(75)
        assert isinstance(peeked_axis, CoordinateAxis)
        assert len(peeked_axis.data) == 75
        np.testing.assert_array_almost_equal(peeked_axis.data, timestamps1[:75])

        # Seek forward
        sought = buf.seek(50)
        assert sought == 50
        assert buf.available() == 100

        # Peek again - should start from sample 50
        peeked_axis = buf.peek(50)
        np.testing.assert_array_almost_equal(peeked_axis.data, timestamps1[50:])

    def test_prune(self):
        """Test pruning samples from buffer"""
        buf = HybridAxisBuffer(duration=1.0, update_strategy="immediate")

        # Write 200 samples with LinearAxis
        axis = LinearAxis(gain=0.001, offset=0.0)
        buf.write(axis, n_samples=200)

        assert buf.available() == 200

        # Prune to keep only last 50 samples
        pruned = buf.prune(50)
        assert pruned == 150
        assert buf.available() == 50
        assert buf._linear_axis.offset == pytest.approx(0.15)  # 0.0 + 150*0.001

        # Try pruning more than available (should do nothing)
        pruned = buf.prune(100)
        assert pruned == 0
        assert buf.available() == 50

    def test_final_value_linear(self):
        """Test getting final value with LinearAxis"""
        buf = HybridAxisBuffer(duration=1.0)

        axis = LinearAxis(gain=0.01, offset=1.0)
        buf.write(axis, n_samples=50)

        # Final value should be offset + (n_samples-1) * gain
        expected = 1.0 + 49 * 0.01
        assert buf.final_value == pytest.approx(expected)

    def test_final_value_coordinate(self):
        """Test getting final value with CoordinateAxis"""
        buf = HybridAxisBuffer(duration=1.0, update_strategy="immediate")

        timestamps = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        axis = CoordinateAxis(data=timestamps, dims=["time"])
        buf.write(axis, n_samples=5)

        # Final value should be the last timestamp
        assert buf.final_value == pytest.approx(1.4)

        # Note: final_value accesses the last element directly from peek()
        # which returns the actual data value

    def test_searchsorted_linear(self):
        """Test searchsorted with LinearAxis"""
        buf = HybridAxisBuffer(duration=1.0)

        # Create axis with samples at 0.0, 0.01, 0.02, ..., 0.49
        axis = LinearAxis(gain=0.01, offset=0.0)
        buf.write(axis, n_samples=50)

        sim_values = axis.value(np.arange(50))

        # Test single value in between
        test_val = 0.025
        expected = np.searchsorted(sim_values, test_val)
        idx = buf.searchsorted(test_val)
        assert idx == expected

        # Test array of values, at least one of which should be equal to an axis value
        values = np.array([0.015, 0.035, 0.055, 0.1])
        for side in ["left", "right"]:
            expected_indices = np.searchsorted(sim_values, values, side=side)
            indices = buf.searchsorted(values, side=side)
            np.testing.assert_array_equal(indices, expected_indices)

        # Test with empty buffer
        buf.seek(50)  # Clear buffer
        sim_values = sim_values[50:]
        assert buf.searchsorted(test_val) == np.searchsorted(sim_values, test_val)
        np.testing.assert_array_equal(buf.searchsorted(values), np.searchsorted(sim_values, values))

    def test_searchsorted_coordinate(self):
        """Test searchsorted with CoordinateAxis"""
        buf = HybridAxisBuffer(duration=1.0, update_strategy="immediate")

        timestamps = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        axis = CoordinateAxis(data=timestamps, dims=["time"])
        buf.write(axis, n_samples=5)

        # Test single value
        idx = buf.searchsorted(0.015)
        assert idx == 1 or idx == 2  # Depends on searchsorted implementation

        # Test array of values
        values = np.array([0.005, 0.025, 0.045])
        indices = buf.searchsorted(values)
        assert all(0 <= idx <= 5 for idx in indices)

    def test_overflow_behavior(self):
        """Test buffer overflow with LinearAxis"""
        buf = HybridAxisBuffer(duration=0.1)  # Small buffer

        # Initialize with high-frequency axis (10kHz)
        axis = LinearAxis(gain=0.0001, offset=0.0)
        buf.write(axis, n_samples=500)  # 0.05 seconds of data

        assert buf.available() == 500
        assert buf.capacity == 1000  # 0.1 sec / 0.0001

        # Write more data to cause overflow
        axis2 = LinearAxis(gain=0.0001, offset=0.1)
        buf.write(axis2, n_samples=700)  # Total would be 1200

        # Even though LinearAxis doesn't have a true capacity limit,
        #  we simulate one anyway to stay in sync with sister buffers
        #  (e.g., in HybridAxisArrayBuffer)
        assert buf.available() == 1000

        # But capacity remains the same
        assert buf.capacity == 1000

    def test_mixed_axis_types_error(self):
        """Test that mixing axis types raises an error"""
        buf = HybridAxisBuffer(duration=1.0, update_strategy="immediate")

        # Initialize with LinearAxis
        linear_axis = LinearAxis(gain=0.001, offset=0.0)
        buf.write(linear_axis, n_samples=10)

        # Try to write CoordinateAxis - should fail
        coord_axis = CoordinateAxis(data=np.linspace(0, 0.01, 11), dims=["time"])
        with pytest.raises(TypeError):
            buf.write(coord_axis)

    def test_buffer_kwargs_passthrough(self):
        """Test that kwargs are passed through to underlying buffer"""
        buf = HybridAxisBuffer(duration=1.0, update_strategy="threshold", threshold=50)

        # Initialize with CoordinateAxis to create internal buffer
        timestamps = np.linspace(0, 0.1, 101)
        axis = CoordinateAxis(data=timestamps, dims=["time"])
        buf.write(axis, n_samples=101)

        # Check that kwargs were passed through
        assert buf._coords_buffer._update_strategy == "threshold"
        assert buf._coords_buffer._threshold == 50

    def test_linear_axis_value_and_index(self):
        """Test LinearAxis value() and index() methods"""
        buf = HybridAxisBuffer(duration=1.0)

        # Create axis with specific gain and offset
        axis = LinearAxis(gain=0.01, offset=5.0, unit="ms")
        buf._initialize(axis)

        # Test that the axis methods work correctly
        assert axis.value(0) == 5.0
        assert axis.value(10) == 5.1
        assert axis.value(np.array([0, 10, 20])) == pytest.approx([5.0, 5.1, 5.2])

        # Test index calculation (inverse of value)
        assert axis.index(5.0) == 0
        assert axis.index(5.1) == 10
        assert axis.index(5.05) == 5  # Should round by default

        # Test with numpy array
        values = np.array([5.0, 5.1, 5.2])
        indices = axis.index(values)
        np.testing.assert_array_equal(indices, [0, 10, 20])

    def test_edge_cases(self):
        """Test various edge cases"""
        # Test with very small gain (large capacity)
        buf = HybridAxisBuffer(duration=1.0)
        axis_small_gain = LinearAxis(gain=0.00001, offset=0.0)
        buf._initialize(axis_small_gain)
        assert buf.capacity == 100000  # 1.0 / 0.00001

        # Test with zero duration (would cause division by zero)
        buf2 = HybridAxisBuffer(duration=0.0)
        axis = LinearAxis(gain=0.001, offset=0.0)
        buf2._initialize(axis)
        assert buf2.capacity == 0

        # Test peek with None (should return all available)
        buf3 = HybridAxisBuffer(duration=1.0)
        axis3 = LinearAxis(gain=0.01, offset=0.0)
        buf3.write(axis3, n_samples=50)
        peeked = buf3.peek(None)
        assert peeked.offset == 0.0
        assert peeked.gain == 0.01


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
    assert buf.available() == 0
    assert buf._data_buffer is None
    assert buf._axis_buffer is not None
    assert buf._axis_buffer._linear_axis is None
    assert buf._axis_buffer._coords_buffer is None
    assert buf._template_msg is None

    msg = linear_axis_message(fs=100.0)
    buf.write(msg)

    assert buf.available() == 10
    assert buf._data_buffer is not None
    assert buf._data_buffer.capacity == 100  # 1.0s * 100Hz
    assert buf._axis_buffer._linear_axis.offset == 0.00
    assert buf._template_msg is not None and buf._template_msg.dims == ["time", "ch"]


def test_deferred_initialization_coordinate(coordinate_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0)
    msg = coordinate_axis_message(samples=10, interval=0.01)  # Effective fs = 100Hz
    buf.write(msg)

    assert buf.available() == 10
    assert buf._data_buffer is not None
    assert buf._data_buffer.capacity == 100
    assert buf._axis_buffer is not None
    assert buf._axis_buffer.capacity == 100


def test_add_and_get_linear(linear_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    msg1 = linear_axis_message(samples=10, fs=100.0, offset=0.0)
    buf.write(msg1)

    msg2 = linear_axis_message(samples=10, fs=100.0, offset=0.1)
    buf.write(msg2)

    assert buf.available() == 20
    retrieved_msg = buf.read(15)
    assert retrieved_msg.shape == (15, 2)
    assert retrieved_msg.dims == msg1.dims
    # Last sample of msg2 is at 0.1 + 9*0.01 = 0.19. Total unread was 20.
    # Offset of oldest sample = 0.19 - (20-1)*0.01 = 0.0
    assert retrieved_msg.axes["time"].offset == pytest.approx(0.0)
    expected_data = np.concatenate([msg1.data, msg2.data[:5]])
    np.testing.assert_array_equal(retrieved_msg.data, expected_data)

    # Check that the buffer now has 5 samples left
    assert buf.available() == 5
    remaining_msg = buf.read()
    np.testing.assert_array_equal(remaining_msg.data, msg2.data[5:])


def test_get_all_data_default(linear_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0)
    msg1 = linear_axis_message(samples=10)
    msg2 = linear_axis_message(samples=15)
    buf.write(msg1)
    buf.write(msg2)

    retrieved = buf.read()
    assert retrieved.shape[0] == 25
    assert buf.available() == 0


def test_add_and_get_coordinate(coordinate_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    msg1 = coordinate_axis_message(samples=10, start_time=0.0)
    buf.write(msg1)

    msg2 = coordinate_axis_message(samples=10, start_time=0.1)
    buf.write(msg2)

    assert buf.available() == 20
    retrieved_msg = buf.read(15)
    assert retrieved_msg.shape == (15, 2)
    assert retrieved_msg.dims == msg1.dims

    expected_data = np.concatenate([msg1.data, msg2.data[:5]])
    np.testing.assert_array_equal(retrieved_msg.data, expected_data)

    expected_times = np.concatenate([msg1.axes["time"].data, msg2.axes["time"].data[:5]])
    np.testing.assert_allclose(retrieved_msg.axes["time"].data, expected_times)

    assert buf.available() == 5


def test_type_mismatch_error(linear_axis_message, coordinate_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0)
    buf.write(linear_axis_message())
    with pytest.raises(TypeError):
        buf.write(coordinate_axis_message())


def test_peek_linear(linear_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    msg1 = linear_axis_message(samples=10, fs=100.0, offset=0.0)
    buf.write(msg1)
    msg2 = linear_axis_message(samples=10, fs=100.0, offset=0.1)
    buf.write(msg2)

    assert buf.available() == 20
    peeked_msg = buf.peek(15)
    assert peeked_msg.shape == (15, 2)
    assert peeked_msg.dims == msg1.dims
    assert peeked_msg.axes["time"].offset == pytest.approx(0.0)
    expected_data = np.concatenate([msg1.data, msg2.data[:5]])
    np.testing.assert_array_equal(peeked_msg.data, expected_data)

    # Assert that state has not changed
    assert buf.available() == 20
    # The underlying _data_buffer._tail should still be 0
    assert buf._data_buffer._tail == 0

    # Get the data to prove it was still there
    retrieved_msg = buf.read(15)
    np.testing.assert_array_equal(retrieved_msg.data, expected_data)
    assert buf.available() == 5


def test_peek_coordinate(coordinate_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    msg1 = coordinate_axis_message(samples=10, start_time=0.0)
    buf.write(msg1)
    msg2 = coordinate_axis_message(samples=10, start_time=0.1)
    buf.write(msg2)

    assert buf.available() == 20
    peeked_msg = buf.peek(15)
    assert peeked_msg.shape == (15, 2)
    assert peeked_msg.dims == msg1.dims
    expected_data = np.concatenate([msg1.data, msg2.data[:5]])
    np.testing.assert_array_equal(peeked_msg.data, expected_data)
    expected_times = np.concatenate([msg1.axes["time"].data, msg2.axes["time"].data[:5]])
    np.testing.assert_allclose(peeked_msg.axes["time"].data, expected_times)

    # Assert that state has not changed
    assert buf.available() == 20
    assert buf._data_buffer.tell() == 0
    assert buf._axis_buffer._coords_buffer.tell() == 0

    # Get the data to prove it was still there
    retrieved_msg = buf.read(15)
    np.testing.assert_array_equal(retrieved_msg.data, expected_data)
    assert buf.available() == 5


def test_seek_linear(linear_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    msg1 = linear_axis_message(samples=10, fs=100.0, offset=0.0)
    buf.write(msg1)
    msg2 = linear_axis_message(samples=10, fs=100.0, offset=0.1)
    buf.write(msg2)

    assert buf.available() == 20
    skipped_count = buf.seek(10)
    assert skipped_count == 10
    assert buf.available() == 10
    assert buf._data_buffer._tail == 10
    assert buf._axis_buffer._linear_axis.offset == pytest.approx(0.1)

    # Get the remaining data
    retrieved_msg = buf.read()
    assert retrieved_msg.shape == (10, 2)
    np.testing.assert_array_equal(retrieved_msg.data, msg2.data)
    # Offset should be 0.1 (start of msg2)
    assert retrieved_msg.axes["time"].offset == pytest.approx(0.1)


def test_seek_coordinate(coordinate_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    msg1 = coordinate_axis_message(samples=10, start_time=0.0)
    buf.write(msg1)
    msg2 = coordinate_axis_message(samples=10, start_time=0.1)
    buf.write(msg2)

    assert buf.available() == 20
    skipped_count = buf.seek(10)
    assert skipped_count == 10
    assert buf.available() == 10
    assert buf._data_buffer._tail == 10
    assert buf._axis_buffer.available() == 10

    # Get the remaining data
    retrieved_msg = buf.read()
    assert retrieved_msg.shape == (10, 2)
    np.testing.assert_array_equal(retrieved_msg.data, msg2.data)
    np.testing.assert_allclose(retrieved_msg.axes["time"].data, msg2.axes["time"].data)


def test_prune(linear_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    buf.write(linear_axis_message(samples=20))
    assert buf.available() == 20
    pruned_count = buf.prune(5)
    assert pruned_count == 15
    assert buf.available() == 5
    retrieved = buf.read()
    assert retrieved.shape[0] == 5


def test_searchsorted_linear(linear_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    buf.write(linear_axis_message(samples=20, fs=100.0, offset=0.1))
    # Buffer now has timestamps from 0.1 to 0.29
    indices = buf.axis_searchsorted(np.array([0.1, 0.15, 0.29]))
    np.testing.assert_array_equal(indices, np.array([0, 5, 19]))


def test_searchsorted_coordinate(coordinate_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, update_strategy="immediate")
    buf.write(coordinate_axis_message(samples=20, start_time=0.1, interval=0.01))
    indices = buf.axis_searchsorted(np.array([0.1, 0.15, 0.29]))
    np.testing.assert_array_equal(indices, np.array([0, 5, 19]))


def test_permute_dims(linear_axis_message):
    buf = HybridAxisArrayBuffer(duration=1.0, axis="time", update_strategy="immediate")
    msg = linear_axis_message(samples=10, fs=100.0, offset=0.0)
    # Swap the axes
    msg.dims = ["ch", "time"]
    msg.data = np.ascontiguousarray(msg.data.T)
    # Write the message; it should automatically permute the dimensions back to ["time", "ch"]
    buf.write(msg)
    assert buf.available() == 10
    assert buf._data_buffer is not None
    assert buf._data_buffer.capacity == 100  # 1.0s * 100Hz
    assert buf._axis_buffer._linear_axis.offset == 0.00
    assert buf._template_msg is not None and buf._template_msg.dims == ["time", "ch"]
    assert msg.dims == ["ch", "time"]  # Unchanged
    retrieved = buf.read()
    assert retrieved.dims == ["time", "ch"]
    assert retrieved.shape == (10, 2)
