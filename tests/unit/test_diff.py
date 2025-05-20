import pytest
import numpy as np

from ezmsg.sigproc.diff import DiffTransformer, DiffSettings
from ezmsg.util.messages.axisarray import AxisArray


@pytest.fixture
def time_axis():
    return AxisArray.TimeAxis(
        fs=10.0,
        offset=0.0,
    )


@pytest.fixture
def channel_axis():
    return AxisArray.CoordinateAxis(data=np.array(["ch1"]), dims=["ch"])


@pytest.fixture
def message(time_axis, channel_axis):
    data = np.array([[0.0], [1.0], [2.0], [3.0]])
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": time_axis, "ch": channel_axis},
        key="test_diff",
    )


@pytest.fixture
def multich_message(time_axis):
    channel_axis = AxisArray.CoordinateAxis(data=np.array(["ch1", "ch2"]), dims=["ch"])
    data = np.array([[0.0, 2.0], [1.0, 4.0], [2.0, 6.0], [3.0, 8.0]])
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": time_axis, "ch": channel_axis},
        key="test_diff",
    )


def test_basic_diff(message):
    """Test basic differentiation along the default axis"""
    # Initialize transformer
    transformer = DiffTransformer(DiffSettings(axis="time", scale_by_fs=False))

    result = transformer(message)

    # First diff is always 0.0
    expected = np.array([[0.0], [1.0], [1.0], [1.0]])
    assert np.array_equal(result.data, expected)


def test_multi_channel_diff(multich_message):
    """Test differentiation with multiple channels"""
    # Initialize transformer
    transformer = DiffTransformer(DiffSettings(axis="time", scale_by_fs=False))

    result = transformer(multich_message)

    expected = np.array([[0.0, 0.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    assert np.array_equal(result.data, expected)


def test_diff_along_channel_axis(multich_message):
    """Test differentiation along channel axis instead of time"""

    # Initialize transformer
    transformer = DiffTransformer(DiffSettings(axis="ch", scale_by_fs=False))

    result = transformer(multich_message)

    expected = np.array([[0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0]])
    assert np.array_equal(result.data, expected)


def test_scale_by_fs(message):
    """Test scaling by sampling frequency"""
    # Create test data

    # Initialize transformer with scale_by_fs=True
    transformer = DiffTransformer(DiffSettings(axis="time", scale_by_fs=True))

    result = transformer(message)

    # Expected diffs: [0.0, 1.0, 1.0, 1.0]
    # dt = 0.1 for each step scales 10x
    expected = np.array([[0.0], [10.0], [10.0], [10.0]])
    assert np.array_equal(result.data, expected)


def test_continuous_processing(channel_axis):
    """Test that state is properly maintained between messages"""
    # Create first message
    message1 = AxisArray(
        data=np.array([[0.0], [0.1], [0.2]]),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(offset=0, fs=10.0), "ch": channel_axis},
        key="test_diff",
    )

    # Create second message (continuing from first)
    message2 = AxisArray(
        data=np.array([[3.0], [4.0], [5.0]]),  # Notice 10x mag jump
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(offset=0.3, fs=10.0), "ch": channel_axis},
        key="test_diff",
    )

    # Initialize transformer
    transformer = DiffTransformer(DiffSettings(axis="time", scale_by_fs=False))

    # Process first message
    result1 = transformer(message1)

    # Evaluate its output
    expected1 = np.array([[0.0], [0.1], [0.1]])
    assert np.array_equal(result1.data, expected1)

    # Process second message (should include diff between messages)
    result2 = transformer(message2)

    # Evaluate its output
    expected2 = np.array([[2.8], [1.0], [1.0]])
    assert np.array_equal(result2.data, expected2)
