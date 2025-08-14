import pytest
import numpy as np
from ezmsg.sigproc.util.buffer import HybridBuffer
from ezmsg.sigproc.util.axisarray_buffer import HybridAxisBuffer, HybridAxisArrayBuffer
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis, CoordinateAxis


@pytest.fixture
def buffer_params():
    return {
        "array_namespace": np,
        "capacity": 100,
        "other_shape": (2,),
        "dtype": np.float32,
        "update_strategy": "immediate",
        "threshold": 0,
        "max_size": 1024**3,  # 1 GB
    }


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


class TestHybridBufferOverflow:
    def test_overflow_strategy_raise(self, buffer_params):
        buf = HybridBuffer(**{**buffer_params, "overflow_strategy": "raise"})
        buf.write(np.zeros((100, 2)))
        with pytest.raises(OverflowError):
            buf.write(np.zeros((1, 2)))

    def test_overflow_strategy_drop(self, buffer_params):
        buf = HybridBuffer(**{**buffer_params, "overflow_strategy": "drop"})
        buf.write(np.ones((80, 2)))
        buf.write(np.ones((30, 2)) * 2)  # 10 samples should be dropped
        assert buf.available() == 100
        data = buf.read()
        assert data.shape[0] == 100
        np.testing.assert_array_equal(data[:80], np.ones((80, 2)))
        np.testing.assert_array_equal(data[80:], np.ones((20, 2)) * 2)

    def test_overflow_strategy_grow(self, buffer_params):
        buf = HybridBuffer(**{**buffer_params, "overflow_strategy": "grow"})
        assert buf.capacity == 100
        buf.write(np.zeros((80, 2)))
        assert buf.capacity == 100
        buf.write(np.zeros((30, 2)))
        assert buf.capacity > 100
        assert buf.available() == 110

        # Test that it fails when max_size is reached
        buf = HybridBuffer(
            **{
                **buffer_params,
                "overflow_strategy": "grow",
                "capacity": 10,
                "max_size": 20 * 2 * 4,  # 20 samples * 2 channels * 4 bytes/float32
            }
        )
        buf.write(np.zeros((10, 2)))
        with pytest.raises(OverflowError):
            buf.write(np.zeros((11, 2)))


class TestHybridAxisBufferOverflow:
    def test_hybrid_axis_buffer_overflow_raise(self):
        buf = HybridAxisBuffer(
            duration=0.1, overflow_strategy="raise", update_strategy="immediate"
        )
        axis = CoordinateAxis(data=np.linspace(0, 0.099, 100), dims=["time"])
        buf.write(axis, n_samples=100)
        with pytest.raises(OverflowError):
            buf.write(axis, n_samples=1)

    def test_hybrid_axis_buffer_overflow_drop(self):
        buf = HybridAxisBuffer(
            duration=0.1, overflow_strategy="drop", update_strategy="immediate"
        )
        axis = CoordinateAxis(data=np.linspace(0, 0.099, 100), dims=["time"])
        buf.write(axis, n_samples=100)
        axis2 = CoordinateAxis(data=np.linspace(0.1, 0.109, 10), dims=["time"])
        buf.write(axis2, n_samples=10)
        assert buf.available() == 100

    def test_hybrid_axis_buffer_overflow_grow(self):
        buf = HybridAxisBuffer(
            duration=0.1, overflow_strategy="grow", update_strategy="immediate"
        )
        axis = CoordinateAxis(data=np.linspace(0, 0.099, 100), dims=["time"])
        buf.write(axis, n_samples=100)
        axis2 = CoordinateAxis(data=np.linspace(0.1, 0.109, 10), dims=["time"])
        buf.write(axis2, n_samples=10)
        assert buf.available() == 110


class TestHybridAxisArrayBufferOverflow:
    def test_hybrid_axis_array_buffer_overflow_raise(self, linear_axis_message):
        buf = HybridAxisArrayBuffer(
            duration=0.1, overflow_strategy="raise", update_strategy="immediate"
        )
        buf.write(linear_axis_message(samples=10, fs=100.0))
        with pytest.raises(OverflowError):
            buf.write(linear_axis_message(samples=1, fs=100.0))

    def test_hybrid_axis_array_buffer_overflow_drop(self, linear_axis_message):
        buf = HybridAxisArrayBuffer(
            duration=0.1, overflow_strategy="drop", update_strategy="immediate"
        )
        buf.write(linear_axis_message(samples=8, fs=100.0))
        buf.write(linear_axis_message(samples=4, fs=100.0))
        assert buf.available() == 10

    def test_hybrid_axis_array_buffer_overflow_grow(self, linear_axis_message):
        buf = HybridAxisArrayBuffer(
            duration=0.1, overflow_strategy="grow", update_strategy="immediate"
        )
        buf.write(linear_axis_message(samples=8, fs=100.0))
        buf.write(linear_axis_message(samples=4, fs=100.0))
        assert buf.available() == 12
