"""Unit tests for ezmsg.sigproc.math.difference module."""

import asyncio
import copy

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.math.difference import (
    ConstDifferenceSettings,
    ConstDifferenceTransformer,
    DifferenceProcessor,
    DifferenceState,
    const_difference,
)
from tests.helpers.util import assert_messages_equal


class TestConstDifferenceTransformer:
    """Tests for ConstDifferenceTransformer."""

    def test_subtract_positive(self):
        """Test subtracting a positive constant from input."""
        transformer = ConstDifferenceTransformer(ConstDifferenceSettings(value=5.0, subtrahend=True))

        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        msg_in = AxisArray(
            data,
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )
        backup = copy.deepcopy(msg_in)

        msg_out = transformer(msg_in)

        expected = np.array([[5.0, 15.0], [25.0, 35.0]])
        assert np.allclose(msg_out.data, expected)
        assert_messages_equal([msg_in], [backup])

    def test_subtract_from_value(self):
        """Test subtracting input from a constant value."""
        transformer = ConstDifferenceTransformer(ConstDifferenceSettings(value=100.0, subtrahend=False))

        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        msg_in = AxisArray(
            data,
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        msg_out = transformer(msg_in)

        # value - data = 100 - data
        expected = np.array([[90.0, 80.0], [70.0, 60.0]])
        assert np.allclose(msg_out.data, expected)

    def test_subtract_zero(self):
        """Test subtracting zero (identity operation)."""
        transformer = ConstDifferenceTransformer(ConstDifferenceSettings(value=0.0, subtrahend=True))

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        msg_in = AxisArray(
            data,
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        msg_out = transformer(msg_in)

        assert np.allclose(msg_out.data, data)

    def test_preserves_axes(self):
        """Test that axes are preserved in output."""
        transformer = ConstDifferenceTransformer(ConstDifferenceSettings(value=1.0))

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        ch_axis = AxisArray.CoordinateAxis(data=np.array(["A", "B"]), dims=["ch"])
        msg_in = AxisArray(
            data,
            dims=["time", "ch"],
            axes=frozendict(
                {
                    "time": AxisArray.TimeAxis(fs=100.0, offset=1.5),
                    "ch": ch_axis,
                }
            ),
        )

        msg_out = transformer(msg_in)

        assert msg_out.dims == msg_in.dims
        assert msg_out.axes["time"].gain == msg_in.axes["time"].gain
        assert msg_out.axes["time"].offset == msg_in.axes["time"].offset


class TestConstDifferenceFactory:
    """Tests for const_difference factory function."""

    def test_factory_creates_transformer(self):
        """Test that factory creates a properly configured transformer."""
        transformer = const_difference(value=7.5, subtrahend=True)

        assert isinstance(transformer, ConstDifferenceTransformer)
        assert transformer.settings.value == 7.5
        assert transformer.settings.subtrahend is True

    def test_factory_subtrahend_false(self):
        """Test factory with subtrahend=False."""
        transformer = const_difference(value=50.0, subtrahend=False)

        assert transformer.settings.value == 50.0
        assert transformer.settings.subtrahend is False

    def test_factory_default_values(self):
        """Test factory with default values."""
        transformer = const_difference()

        assert transformer.settings.value == 0.0
        assert transformer.settings.subtrahend is True


class TestDifferenceProcessor:
    """Tests for DifferenceProcessor."""

    def test_basic_difference(self):
        """Test basic subtraction of two messages (A - B)."""
        processor = DifferenceProcessor()

        data_a = np.array([[10.0, 20.0], [30.0, 40.0]])
        data_b = np.array([[1.0, 2.0], [3.0, 4.0]])

        msg_a = AxisArray(
            data_a,
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )
        msg_b = AxisArray(
            data_b,
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        processor.push_a(msg_a)
        processor.push_b(msg_b)

        # Use sync call
        result = processor()

        expected = np.array([[9.0, 18.0], [27.0, 36.0]])
        assert np.allclose(result.data, expected)

    def test_queue_ordering(self):
        """Test that messages are paired in order."""
        processor = DifferenceProcessor()

        # Push multiple messages to each queue
        for i in range(3):
            msg_a = AxisArray(
                np.array([[float(i * 10)]]),
                dims=["time", "ch"],
                axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
            )
            msg_b = AxisArray(
                np.array([[float(i)]]),
                dims=["time", "ch"],
                axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
            )
            processor.push_a(msg_a)
            processor.push_b(msg_b)

        # Results should be paired in order: (0 - 0), (10 - 1), (20 - 2)
        for i in range(3):
            result = processor()  # Use sync call
            expected = float(i * 10) - float(i)
            assert np.allclose(result.data, np.array([[expected]]))

    def test_state_property(self):
        """Test state getter and setter."""
        processor = DifferenceProcessor()

        assert isinstance(processor.state, DifferenceState)

        new_state = DifferenceState()
        processor.state = new_state
        assert processor.state is new_state

        # Setting None should not change state
        old_state = processor.state
        processor.state = None
        assert processor.state is old_state

    def test_sync_call(self):
        """Test synchronous __call__ method."""
        processor = DifferenceProcessor()

        msg_a = AxisArray(
            np.array([[10.0]]),
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )
        msg_b = AxisArray(
            np.array([[3.0]]),
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        processor.push_a(msg_a)
        processor.push_b(msg_b)

        result = processor()
        assert np.allclose(result.data, np.array([[7.0]]))

    def test_legacy_interface(self):
        """Test legacy __next__ and __anext__ interfaces."""
        processor = DifferenceProcessor()

        msg_a = AxisArray(
            np.array([[20.0]]),
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )
        msg_b = AxisArray(
            np.array([[8.0]]),
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        processor.push_a(msg_a)
        processor.push_b(msg_b)

        # Test __next__
        result = next(processor)
        assert np.allclose(result.data, np.array([[12.0]]))

    def test_negative_result(self):
        """Test that negative results are handled correctly (B > A)."""
        processor = DifferenceProcessor()

        msg_a = AxisArray(
            np.array([[5.0]]),
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )
        msg_b = AxisArray(
            np.array([[10.0]]),
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        processor.push_a(msg_a)
        processor.push_b(msg_b)

        result = processor()
        assert np.allclose(result.data, np.array([[-5.0]]))


class TestDifferenceState:
    """Tests for DifferenceState dataclass."""

    def test_default_queues(self):
        """Test that default queues are created."""
        state = DifferenceState()

        assert isinstance(state.queue_a, asyncio.Queue)
        assert isinstance(state.queue_b, asyncio.Queue)
        assert state.queue_a.empty()
        assert state.queue_b.empty()

    def test_independent_queues(self):
        """Test that queues are independent between instances."""
        state1 = DifferenceState()
        state2 = DifferenceState()

        state1.queue_a.put_nowait("test")

        assert not state1.queue_a.empty()
        assert state2.queue_a.empty()
