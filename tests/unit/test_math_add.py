"""Unit tests for ezmsg.sigproc.math.add module."""

import asyncio
import copy

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.math.add import (
    AddProcessor,
    AddState,
    ConstAddSettings,
    ConstAddTransformer,
)
from tests.helpers.util import assert_messages_equal


class TestConstAddTransformer:
    """Tests for ConstAddTransformer."""

    def test_basic_add_positive(self):
        """Test adding a positive constant."""
        transformer = ConstAddTransformer(ConstAddSettings(value=5.0))

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        msg_in = AxisArray(
            data,
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )
        backup = copy.deepcopy(msg_in)

        msg_out = transformer(msg_in)

        expected = np.array([[6.0, 7.0], [8.0, 9.0]])
        assert np.allclose(msg_out.data, expected)
        assert_messages_equal([msg_in], [backup])

    def test_basic_add_negative(self):
        """Test adding a negative constant (effectively subtraction)."""
        transformer = ConstAddTransformer(ConstAddSettings(value=-3.0))

        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        msg_in = AxisArray(
            data,
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        msg_out = transformer(msg_in)

        expected = np.array([[7.0, 17.0], [27.0, 37.0]])
        assert np.allclose(msg_out.data, expected)

    def test_add_zero(self):
        """Test adding zero (identity operation)."""
        transformer = ConstAddTransformer(ConstAddSettings(value=0.0))

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
        transformer = ConstAddTransformer(ConstAddSettings(value=1.0))

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

    def test_stateless_across_chunks(self):
        """Test that transformer is stateless across multiple chunks."""
        transformer = ConstAddTransformer(ConstAddSettings(value=10.0))

        chunks = [
            AxisArray(
                np.array([[i * 1.0]]),
                dims=["time", "ch"],
                axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0, offset=i * 0.01)}),
            )
            for i in range(5)
        ]

        outputs = [transformer(chunk) for chunk in chunks]

        for i, out in enumerate(outputs):
            assert np.allclose(out.data, np.array([[i * 1.0 + 10.0]]))


class TestAddProcessor:
    """Tests for AddProcessor."""

    def test_basic_add(self):
        """Test basic addition of two messages."""
        processor = AddProcessor()

        data_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_b = np.array([[10.0, 20.0], [30.0, 40.0]])

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

        expected = np.array([[11.0, 22.0], [33.0, 44.0]])
        assert np.allclose(result.data, expected)

    def test_queue_ordering(self):
        """Test that messages are paired in order."""
        processor = AddProcessor()

        # Push multiple messages to each queue
        for i in range(3):
            msg_a = AxisArray(
                np.array([[float(i)]]),
                dims=["time", "ch"],
                axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
            )
            msg_b = AxisArray(
                np.array([[float(i * 10)]]),
                dims=["time", "ch"],
                axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
            )
            processor.push_a(msg_a)
            processor.push_b(msg_b)

        # Results should be paired in order
        for i in range(3):
            result = processor()  # Use sync call
            expected = float(i) + float(i * 10)
            assert np.allclose(result.data, np.array([[expected]]))

    def test_state_property(self):
        """Test state getter and setter."""
        processor = AddProcessor()

        assert isinstance(processor.state, AddState)

        new_state = AddState()
        processor.state = new_state
        assert processor.state is new_state

        # Setting None should not change state
        old_state = processor.state
        processor.state = None
        assert processor.state is old_state

    def test_sync_call(self):
        """Test synchronous __call__ method."""
        processor = AddProcessor()

        msg_a = AxisArray(
            np.array([[1.0]]),
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )
        msg_b = AxisArray(
            np.array([[2.0]]),
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        processor.push_a(msg_a)
        processor.push_b(msg_b)

        result = processor()
        assert np.allclose(result.data, np.array([[3.0]]))

    def test_legacy_interface(self):
        """Test legacy __next__ and __anext__ interfaces."""
        processor = AddProcessor()

        msg_a = AxisArray(
            np.array([[5.0]]),
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )
        msg_b = AxisArray(
            np.array([[7.0]]),
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        processor.push_a(msg_a)
        processor.push_b(msg_b)

        # Test __next__
        result = next(processor)
        assert np.allclose(result.data, np.array([[12.0]]))


class TestAddState:
    """Tests for AddState dataclass."""

    def test_default_queues(self):
        """Test that default queues are created."""
        state = AddState()

        assert isinstance(state.queue_a, asyncio.Queue)
        assert isinstance(state.queue_b, asyncio.Queue)
        assert state.queue_a.empty()
        assert state.queue_b.empty()

    def test_independent_queues(self):
        """Test that queues are independent between instances."""
        state1 = AddState()
        state2 = AddState()

        state1.queue_a.put_nowait("test")

        assert not state1.queue_a.empty()
        assert state2.queue_a.empty()
