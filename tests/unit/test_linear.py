import copy

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.linear import (
    LinearTransformSettings,
    LinearTransformTransformer,
)
from tests.helpers.util import assert_messages_equal


class TestLinearTransformScalar:
    """Tests for scalar scale and offset."""

    def test_scale_only(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])
        backup = [copy.deepcopy(msg_in)]

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=2.0))
        msg_out = xformer(msg_in)

        assert_messages_equal([msg_in], backup)
        assert np.allclose(msg_out.data, data * 2.0)

    def test_offset_only(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])
        backup = [copy.deepcopy(msg_in)]

        xformer = LinearTransformTransformer(LinearTransformSettings(offset=10.0))
        msg_out = xformer(msg_in)

        assert_messages_equal([msg_in], backup)
        assert np.allclose(msg_out.data, data + 10.0)

    def test_scale_and_offset(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])
        backup = [copy.deepcopy(msg_in)]

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=2.0, offset=1.0))
        msg_out = xformer(msg_in)

        assert_messages_equal([msg_in], backup)
        assert np.allclose(msg_out.data, data * 2.0 + 1.0)

    def test_identity(self):
        """Default settings should be identity transform."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])

        xformer = LinearTransformTransformer(LinearTransformSettings())
        msg_out = xformer(msg_in)

        assert np.allclose(msg_out.data, data)

    def test_negative_scale(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=-1.0, offset=0.0))
        msg_out = xformer(msg_in)

        assert np.allclose(msg_out.data, -data)


class TestLinearTransformPerChannel:
    """Tests for per-channel scale and offset."""

    def test_per_channel_scale(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])
        backup = [copy.deepcopy(msg_in)]

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=[1.0, 2.0, 3.0], axis="ch"))
        msg_out = xformer(msg_in)

        assert_messages_equal([msg_in], backup)
        expected = data * np.array([1.0, 2.0, 3.0])
        assert np.allclose(msg_out.data, expected)

    def test_per_channel_offset(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])
        backup = [copy.deepcopy(msg_in)]

        xformer = LinearTransformTransformer(LinearTransformSettings(offset=[10.0, 20.0, 30.0], axis="ch"))
        msg_out = xformer(msg_in)

        assert_messages_equal([msg_in], backup)
        expected = data + np.array([10.0, 20.0, 30.0])
        assert np.allclose(msg_out.data, expected)

    def test_per_channel_scale_and_offset(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])
        backup = [copy.deepcopy(msg_in)]

        xformer = LinearTransformTransformer(
            LinearTransformSettings(
                scale=[2.0, 0.5],
                offset=[1.0, -1.0],
                axis="ch",
            )
        )
        msg_out = xformer(msg_in)

        assert_messages_equal([msg_in], backup)
        expected = np.array(
            [
                [1.0 * 2.0 + 1.0, 2.0 * 0.5 - 1.0],
                [3.0 * 2.0 + 1.0, 4.0 * 0.5 - 1.0],
            ]
        )
        assert np.allclose(msg_out.data, expected)

    def test_velocity_to_beta_use_case(self):
        """Test the velocity magnitude/angle to beta scaling use case."""
        # Simulate CART2POL output: magnitude ~314, angle 0-2π
        data = np.array(
            [
                [100.0, 0.0],
                [200.0, np.pi],
                [314.0, 2 * np.pi],
            ]
        )
        msg_in = AxisArray(data, dims=["time", "ch"])

        # Scale to beta range 0.5-2.0
        xformer = LinearTransformTransformer(
            LinearTransformSettings(
                scale=[1.5 / 314, 1.5 / (2 * np.pi)],
                offset=[0.5, 0.5],
                axis="ch",
            )
        )
        msg_out = xformer(msg_in)

        # Check output is in expected beta range
        assert msg_out.data.min() >= 0.5 - 1e-10
        assert msg_out.data.max() <= 2.0 + 1e-10

        # Check specific values
        expected = np.array(
            [
                [100.0 * 1.5 / 314 + 0.5, 0.0 * 1.5 / (2 * np.pi) + 0.5],
                [200.0 * 1.5 / 314 + 0.5, np.pi * 1.5 / (2 * np.pi) + 0.5],
                [314.0 * 1.5 / 314 + 0.5, 2 * np.pi * 1.5 / (2 * np.pi) + 0.5],
            ]
        )
        assert np.allclose(msg_out.data, expected)


class TestLinearTransformDifferentAxes:
    """Tests for operating on different axes."""

    def test_time_axis(self):
        """Test per-sample scaling along time axis."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=[1.0, 2.0, 3.0], axis="time"))
        msg_out = xformer(msg_in)

        expected = np.array(
            [
                [1.0 * 1.0, 2.0 * 1.0],
                [3.0 * 2.0, 4.0 * 2.0],
                [5.0 * 3.0, 6.0 * 3.0],
            ]
        )
        assert np.allclose(msg_out.data, expected)

    def test_3d_data_middle_axis(self):
        """Test with 3D data operating on middle axis."""
        data = np.ones((2, 3, 4))  # time, ch, freq
        msg_in = AxisArray(data, dims=["time", "ch", "freq"])

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=[1.0, 2.0, 3.0], axis="ch"))
        msg_out = xformer(msg_in)

        assert msg_out.data.shape == (2, 3, 4)
        assert np.allclose(msg_out.data[:, 0, :], 1.0)
        assert np.allclose(msg_out.data[:, 1, :], 2.0)
        assert np.allclose(msg_out.data[:, 2, :], 3.0)


class TestLinearTransformState:
    """Tests for state persistence and hash-based reset."""

    def test_state_persistence(self):
        """State should persist across multiple calls with same shape."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=[2.0, 0.5], offset=[1.0, -1.0], axis="ch"))

        # First call
        msg_out1 = xformer(msg_in)

        # Second call - should use cached state
        msg_out2 = xformer(msg_in)

        assert np.allclose(msg_out1.data, msg_out2.data)

    def test_state_reset_on_shape_change(self):
        """State should reset when input shape changes."""
        xformer = LinearTransformTransformer(LinearTransformSettings(scale=[1.0, 2.0], axis="ch"))

        # First call with 2 channels
        data1 = np.array([[1.0, 2.0]])
        msg1 = AxisArray(data1, dims=["time", "ch"])
        out1 = xformer(msg1)
        assert np.allclose(out1.data, np.array([[1.0, 4.0]]))

        # Second call with 3 channels - should reset state
        # Note: This will fail because scale array doesn't match new shape
        # The transformer should handle this gracefully or raise an error
        data2 = np.array([[1.0, 2.0, 3.0]])
        msg2 = AxisArray(data2, dims=["time", "ch"])
        with pytest.raises((ValueError, IndexError)):
            xformer(msg2)

    def test_state_reset_on_ndim_change(self):
        """State should reset when input ndim changes."""
        xformer = LinearTransformTransformer(LinearTransformSettings(scale=2.0, offset=1.0))

        # First call with 2D data
        data1 = np.array([[1.0, 2.0]])
        msg1 = AxisArray(data1, dims=["time", "ch"])
        out1 = xformer(msg1)
        assert np.allclose(out1.data, data1 * 2.0 + 1.0)

        # Second call with 3D data
        data2 = np.ones((2, 3, 4))
        msg2 = AxisArray(data2, dims=["time", "ch", "freq"])
        out2 = xformer(msg2)
        assert np.allclose(out2.data, data2 * 2.0 + 1.0)


class TestLinearTransformEdgeCases:
    """Tests for edge cases."""

    def test_single_element(self):
        data = np.array([[1.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=2.0, offset=1.0))
        msg_out = xformer(msg_in)

        assert np.allclose(msg_out.data, np.array([[3.0]]))

    def test_empty_data(self):
        data = np.array([]).reshape(0, 2)
        msg_in = AxisArray(data, dims=["time", "ch"])

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=2.0, offset=1.0))
        msg_out = xformer(msg_in)

        assert msg_out.data.shape == (0, 2)

    def test_large_values(self):
        data = np.array([[1e10, 1e-10]])
        msg_in = AxisArray(data, dims=["time", "ch"])

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=2.0, offset=1.0))
        msg_out = xformer(msg_in)

        expected = data * 2.0 + 1.0
        assert np.allclose(msg_out.data, expected)

    def test_preserves_dtype(self):
        """Output dtype should match computation dtype (float64)."""
        data = np.array([[1, 2], [3, 4]], dtype=np.int32)
        msg_in = AxisArray(data, dims=["time", "ch"])

        xformer = LinearTransformTransformer(LinearTransformSettings(scale=2.0, offset=0.5))
        msg_out = xformer(msg_in)

        # Result should be float due to float scale/offset
        assert msg_out.data.dtype in [np.float64, np.float32]
        assert np.allclose(msg_out.data, np.array([[2.5, 4.5], [6.5, 8.5]]))

    def test_numpy_array_settings(self):
        """Test with numpy arrays in settings instead of lists."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])

        xformer = LinearTransformTransformer(
            LinearTransformSettings(
                scale=np.array([2.0, 0.5]),
                offset=np.array([1.0, -1.0]),
                axis="ch",
            )
        )
        msg_out = xformer(msg_in)

        expected = np.array(
            [
                [1.0 * 2.0 + 1.0, 2.0 * 0.5 - 1.0],
                [3.0 * 2.0 + 1.0, 4.0 * 0.5 - 1.0],
            ]
        )
        assert np.allclose(msg_out.data, expected)
