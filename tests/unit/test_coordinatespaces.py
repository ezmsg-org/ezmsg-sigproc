import copy

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.coordinatespaces import (
    CoordinateMode,
    CoordinateSpacesSettings,
    CoordinateSpacesTransformer,
    cart2pol,
    cart2z,
    pol2cart,
    polar2z,
    z2cart,
    z2polar,
)
from tests.helpers.util import assert_messages_equal


class TestUtilityFunctions:
    """Test the standalone coordinate transformation utility functions."""

    def test_polar2z_scalar(self):
        """Test polar to complex conversion with scalars."""
        # r=1, theta=0 -> z=1+0j
        z = polar2z(1.0, 0.0)
        assert np.isclose(z, 1.0 + 0j)

        # r=1, theta=pi/2 -> z=0+1j
        z = polar2z(1.0, np.pi / 2)
        assert np.isclose(z, 1j, atol=1e-10)

    def test_polar2z_array(self):
        """Test polar to complex conversion with arrays."""
        r = np.array([1.0, 2.0, 3.0])
        theta = np.array([0.0, np.pi / 2, np.pi])
        z = polar2z(r, theta)
        expected = np.array([1.0 + 0j, 2j, -3.0 + 0j])
        assert np.allclose(z, expected, atol=1e-10)

    def test_z2polar_scalar(self):
        """Test complex to polar conversion with scalars."""
        r, theta = z2polar(1.0 + 1j)
        assert np.isclose(r, np.sqrt(2))
        assert np.isclose(theta, np.pi / 4)

    def test_z2polar_array(self):
        """Test complex to polar conversion with arrays."""
        z = np.array([1.0 + 0j, 0 + 1j, -1.0 + 0j])
        r, theta = z2polar(z)
        assert np.allclose(r, [1.0, 1.0, 1.0])
        assert np.allclose(theta, [0.0, np.pi / 2, np.pi])

    def test_cart2z_scalar(self):
        """Test Cartesian to complex conversion."""
        z = cart2z(3.0, 4.0)
        assert z == 3.0 + 4j

    def test_z2cart_scalar(self):
        """Test complex to Cartesian conversion."""
        x, y = z2cart(3.0 + 4j)
        assert x == 3.0
        assert y == 4.0

    def test_cart2pol_known_values(self):
        """Test Cartesian to polar with known values."""
        # (1, 0) -> r=1, theta=0
        r, theta = cart2pol(1.0, 0.0)
        assert np.isclose(r, 1.0)
        assert np.isclose(theta, 0.0)

        # (0, 1) -> r=1, theta=pi/2
        r, theta = cart2pol(0.0, 1.0)
        assert np.isclose(r, 1.0)
        assert np.isclose(theta, np.pi / 2)

        # (1, 1) -> r=sqrt(2), theta=pi/4
        r, theta = cart2pol(1.0, 1.0)
        assert np.isclose(r, np.sqrt(2))
        assert np.isclose(theta, np.pi / 4)

    def test_pol2cart_known_values(self):
        """Test polar to Cartesian with known values."""
        # r=1, theta=0 -> (1, 0)
        x, y = pol2cart(1.0, 0.0)
        assert np.isclose(x, 1.0)
        assert np.isclose(y, 0.0, atol=1e-10)

        # r=1, theta=pi/2 -> (0, 1)
        x, y = pol2cart(1.0, np.pi / 2)
        assert np.isclose(x, 0.0, atol=1e-10)
        assert np.isclose(y, 1.0)

    def test_roundtrip_cart_pol_cart(self):
        """Test round-trip conversion: cart -> pol -> cart."""
        x_orig = np.array([1.0, 2.0, -1.0, 0.0])
        y_orig = np.array([0.0, 3.0, 1.0, -2.0])

        r, theta = cart2pol(x_orig, y_orig)
        x_back, y_back = pol2cart(r, theta)

        assert np.allclose(x_back, x_orig)
        assert np.allclose(y_back, y_orig)

    def test_roundtrip_pol_cart_pol(self):
        """Test round-trip conversion: pol -> cart -> pol."""
        r_orig = np.array([1.0, 2.0, 3.0])
        theta_orig = np.array([0.0, np.pi / 4, np.pi / 2])

        x, y = pol2cart(r_orig, theta_orig)
        r_back, theta_back = cart2pol(x, y)

        assert np.allclose(r_back, r_orig)
        assert np.allclose(theta_back, theta_orig)


class TestCoordinateSpacesTransformer:
    """Test the CoordinateSpacesTransformer class."""

    def test_cart2pol_basic(self):
        """Test basic Cartesian to polar conversion."""
        # Data: 3 time points, 2 coordinates (x, y)
        data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        msg_in = AxisArray(data, dims=["time", "ch"])

        backup = [copy.deepcopy(msg_in)]

        transformer = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch"))
        msg_out = transformer(msg_in)

        # Check shape preserved
        assert msg_out.data.shape == data.shape

        # Check values: (1,0) -> (1, 0), (0,1) -> (1, pi/2), (1,1) -> (sqrt(2), pi/4)
        expected_r = np.array([1.0, 1.0, np.sqrt(2)])
        expected_theta = np.array([0.0, np.pi / 2, np.pi / 4])
        assert np.allclose(msg_out.data[:, 0], expected_r)
        assert np.allclose(msg_out.data[:, 1], expected_theta)

        # Verify input not mutated
        assert_messages_equal([msg_in], backup)

    def test_pol2cart_basic(self):
        """Test basic polar to Cartesian conversion."""
        # Data: (r, theta) pairs
        data = np.array([[1.0, 0.0], [1.0, np.pi / 2], [np.sqrt(2), np.pi / 4]])
        msg_in = AxisArray(data, dims=["time", "ch"])

        transformer = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.POL2CART, axis="ch"))
        msg_out = transformer(msg_in)

        # Check values
        expected_x = np.array([1.0, 0.0, 1.0])
        expected_y = np.array([0.0, 1.0, 1.0])
        assert np.allclose(msg_out.data[:, 0], expected_x, atol=1e-10)
        assert np.allclose(msg_out.data[:, 1], expected_y, atol=1e-10)

    def test_roundtrip_transformer(self):
        """Test round-trip transformation preserves data."""
        data = np.random.randn(100, 2)  # Random (x, y) pairs
        msg_in = AxisArray(data, dims=["time", "ch"])

        # cart -> pol
        c2p = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch"))
        msg_polar = c2p(msg_in)

        # pol -> cart
        p2c = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.POL2CART, axis="ch"))
        msg_back = p2c(msg_polar)

        assert np.allclose(msg_back.data, data)

    def test_default_axis(self):
        """Test that default axis is the last dimension."""
        data = np.array([[1.0, 0.0], [0.0, 1.0]])
        msg_in = AxisArray(data, dims=["time", "xy"])

        # No axis specified - should use last dim ("xy")
        transformer = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL))
        msg_out = transformer(msg_in)

        assert msg_out.data.shape == data.shape
        assert np.isclose(msg_out.data[0, 0], 1.0)  # r=1 for (1,0)
        assert np.isclose(msg_out.data[0, 1], 0.0)  # theta=0 for (1,0)

    def test_axis_not_last(self):
        """Test transformation when coordinate axis is not the last dimension."""
        # Shape: (2 coords, 5 time points)
        data = np.array([[1.0, 0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 1.0, 0.0, 2.0]])
        msg_in = AxisArray(data, dims=["ch", "time"])

        transformer = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch"))
        msg_out = transformer(msg_in)

        assert msg_out.data.shape == data.shape
        # First column: (1, 0) -> r=1
        assert np.isclose(msg_out.data[0, 0], 1.0)
        # Second column: (0, 1) -> r=1, theta=pi/2
        assert np.isclose(msg_out.data[0, 1], 1.0)
        assert np.isclose(msg_out.data[1, 1], np.pi / 2)

    def test_3d_array(self):
        """Test with 3D array (batch, time, coord)."""
        data = np.random.randn(4, 10, 2)  # 4 batches, 10 time points, 2 coords
        msg_in = AxisArray(data, dims=["batch", "time", "ch"])

        c2p = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch"))
        msg_polar = c2p(msg_in)

        p2c = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.POL2CART, axis="ch"))
        msg_back = p2c(msg_polar)

        assert msg_polar.data.shape == data.shape
        assert np.allclose(msg_back.data, data)

    def test_wrong_axis_size_raises(self):
        """Test that wrong axis size raises ValueError."""
        data = np.random.randn(10, 3)  # 3 coordinates instead of 2
        msg_in = AxisArray(data, dims=["time", "ch"])

        transformer = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch"))

        with pytest.raises(ValueError, match="exactly 2 elements"):
            transformer(msg_in)

    def test_axis_labels_updated_cart2pol(self):
        """Test that axis labels are updated for cart2pol."""
        data = np.array([[1.0, 0.0]])
        coord_axis = AxisArray.CoordinateAxis(data=np.array(["x", "y"]), dims=["ch"])
        msg_in = AxisArray(
            data,
            dims=["time", "ch"],
            axes={"ch": coord_axis},
        )

        transformer = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch"))
        msg_out = transformer(msg_in)

        assert "ch" in msg_out.axes
        assert list(msg_out.axes["ch"].data) == ["r", "theta"]

    def test_axis_labels_updated_pol2cart(self):
        """Test that axis labels are updated for pol2cart."""
        data = np.array([[1.0, 0.0]])
        coord_axis = AxisArray.CoordinateAxis(data=np.array(["r", "theta"]), dims=["ch"])
        msg_in = AxisArray(
            data,
            dims=["time", "ch"],
            axes={"ch": coord_axis},
        )

        transformer = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.POL2CART, axis="ch"))
        msg_out = transformer(msg_in)

        assert "ch" in msg_out.axes
        assert list(msg_out.axes["ch"].data) == ["x", "y"]

    def test_mode_string_enum(self):
        """Test that CoordinateMode can be constructed from string."""
        assert CoordinateMode("cart2pol") == CoordinateMode.CART2POL
        assert CoordinateMode("pol2cart") == CoordinateMode.POL2CART

    def test_multiple_sends(self):
        """Test that transformer can process multiple messages."""
        transformer = CoordinateSpacesTransformer(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch"))

        for _ in range(5):
            data = np.random.randn(10, 2)
            msg_in = AxisArray(data, dims=["time", "ch"])
            msg_out = transformer(msg_in)
            assert msg_out.data.shape == data.shape
