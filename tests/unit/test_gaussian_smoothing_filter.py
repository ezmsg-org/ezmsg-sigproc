import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.gaussiansmoothing import (
    GaussianSmoothingFilterTransformer,
    GaussianSmoothingSettings,
    gaussian_smoothing_filter_design,
)


@pytest.mark.parametrize(
    "axis,sigma,width,kernel_size",
    [
        ("time", 1.5, 5, None),
        ("time", 2.0, 4, 21),
    ],
)
def test_gaussian_smoothing_filter_function(axis, sigma, width, kernel_size):
    """Test the gaussian_smoothing_filter convenience function."""
    transformer = GaussianSmoothingFilterTransformer(
        axis=axis,
        sigma=sigma,
        width=width,
        kernel_size=kernel_size,
    )

    assert isinstance(transformer, GaussianSmoothingFilterTransformer)
    assert transformer.settings.axis == axis
    assert transformer.settings.sigma == sigma
    assert transformer.settings.width == width
    assert transformer.settings.kernel_size == kernel_size


def test_gaussian_smoothing_settings_defaults():
    """Test the GaussianSmoothingSettings class with default values."""
    settings = GaussianSmoothingSettings()
    assert settings.sigma == 1.0
    assert settings.width == 4
    assert settings.kernel_size is None


def test_gaussian_smoothing_settings_custom():
    """Test the GaussianSmoothingSettings class with custom values."""
    settings = GaussianSmoothingSettings(
        sigma=2.5,
        width=6,
        kernel_size=21,
    )
    assert settings.sigma == 2.5
    assert settings.width == 6
    assert settings.kernel_size == 21


@pytest.mark.parametrize(
    "sigma,width,kernel_size",
    [
        (1.0, 4, None),
        (2.0, 6, None),
        (1.5, 5, 11),
        (1.5, 5, 17),  # Fixed kernel_size to be >= expected
    ],
)
def test_gaussian_smoothing_filter_design_parameters(sigma, width, kernel_size):
    """Test gaussian smoothing filter design across multiple parameter configurations."""
    coefs = gaussian_smoothing_filter_design(
        sigma=sigma,
        width=width,
        kernel_size=kernel_size,
    )
    assert coefs is not None
    assert isinstance(coefs, tuple)
    assert len(coefs) == 2  # b and a coefficients

    b, a = coefs
    assert b is not None and a is not None
    assert isinstance(b, np.ndarray) and isinstance(a, np.ndarray)
    assert np.all(b >= 0)  # positive
    assert np.allclose(b, b[::-1])  # symmetric
    assert np.isclose(np.sum(b), 1.0)  # normalized
    assert b[len(b) // 2] == np.max(b)  # center of kernel is peak
    assert len(a) == 1 and a[0] == 1.0  # default for gaussian window

    expected_kernel_size = int(2 * width * sigma + 1) if kernel_size is None else kernel_size
    assert len(b) == expected_kernel_size


def test_gaussian_smoothing_kernel_properties():
    """Test that larger sigma creates wider kernel"""
    coefs_small = gaussian_smoothing_filter_design(sigma=1.0)
    b_small, _ = coefs_small
    coefs_large = gaussian_smoothing_filter_design(sigma=3.0)
    b_large, _ = coefs_large

    assert len(b_large) >= len(b_small)  # wider kernel
    assert b_large[len(b_large) // 2] < b_small[len(b_small) // 2]  # lower peak


@pytest.mark.parametrize("sigma", [0.0, -1.0])
@pytest.mark.parametrize("width", [0.0, -1.0])
@pytest.mark.parametrize("kernel_size", [0, -1])
def test_gaussian_smoothing_filter_design_invalid_inputs(sigma, width, kernel_size):
    """Test the gaussian smoothing filter design function with invalid inputs."""
    with pytest.raises(ValueError):
        gaussian_smoothing_filter_design(sigma=sigma)
    with pytest.raises(ValueError):
        gaussian_smoothing_filter_design(width=width)
    with pytest.raises(ValueError):
        gaussian_smoothing_filter_design(kernel_size=kernel_size)


@pytest.mark.parametrize("data_shape", [(100,), (1, 100), (100, 2), (100, 2, 3)])
def test_gaussian_smoothing_filter_process(data_shape):
    """Test gaussian smoothing filter with different data shapes."""
    # Create test data
    data = np.arange(np.prod(data_shape)).reshape(data_shape)

    # Create appropriate dims and axes based on shape
    if len(data_shape) == 1:
        dims = ["time"]
        axes = {"time": AxisArray.TimeAxis(fs=100.0, offset=0)}
    elif len(data_shape) == 2:
        dims = ["time", "ch"]
        axes = {
            "time": AxisArray.TimeAxis(fs=100.0, offset=0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(data_shape[1]).astype(str), dims=["ch"]),
        }
    else:
        dims = ["freq", "time", "ch"]
        axes = {
            "freq": AxisArray.LinearAxis(unit="Hz", offset=0.0, gain=1.0),
            "time": AxisArray.TimeAxis(fs=100.0, offset=0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(data_shape[2]).astype(str), dims=["ch"]),
        }

    msg = AxisArray(data=data, dims=dims, axes=axes, key="test_gaussian_smoothing")

    # Instantiate transformer (not unit)
    transformer = GaussianSmoothingFilterTransformer(
        settings=GaussianSmoothingSettings(axis="time", sigma=2.0, width=4)
    )

    # Process message using __call__ method
    output_msg = transformer(msg)

    # Assertions
    assert isinstance(output_msg, AxisArray)
    assert output_msg.data.shape == data.shape
    assert np.isfinite(output_msg.data).all()


def test_gaussian_smoothing_edge_cases():
    """Test edge cases for gaussian smoothing filter."""
    # Test with very small sigma
    coefs_small = gaussian_smoothing_filter_design(sigma=0.01)
    b_small, a_small = coefs_small
    assert len(b_small) > 0
    assert np.isclose(np.sum(b_small), 1.0)

    # Test with very large sigma
    coefs_large = gaussian_smoothing_filter_design(sigma=100.0)
    b_large, a_large = coefs_large
    assert len(b_large) > 0
    assert np.isclose(np.sum(b_large), 1.0)

    # Test with very small width
    coefs_narrow = gaussian_smoothing_filter_design(width=1)
    b_narrow, a_narrow = coefs_narrow
    assert len(b_narrow) > 0

    # Test with very large width
    coefs_wide = gaussian_smoothing_filter_design(width=100)
    b_wide, a_wide = coefs_wide
    assert len(b_wide) > len(b_narrow)  # Wider kernel


def test_gaussian_smoothing_update_settings():
    """Test the update_settings functionality of the Gaussian smoothing filter."""
    # Setup parameters
    fs = 200.0
    dur = 1.0
    n_times = int(dur * fs)
    n_chans = 2

    # Create input data with high frequency noise
    t = np.arange(n_times) / fs
    # Create a signal with both low and high frequency components
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    in_dat = np.vstack([signal, signal + np.random.randn(n_times) * 0.1]).T

    # Create message
    msg_in = AxisArray(
        data=in_dat,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(n_chans).astype(str), dims=["ch"]),
        },
        key="test_gaussian_smoothing_update_settings",
    )

    def _calc_smoothing_effect(msg):
        """Calculate the smoothing effect by comparing variance."""
        return np.var(msg.data, axis=0)

    original_variance = _calc_smoothing_effect(msg_in)

    # Initialize filter with small sigma (minimal smoothing)
    proc = GaussianSmoothingFilterTransformer(
        axis="time",
        sigma=0.5,
        width=4,
        coef_type="ba",
    )

    # Process first message
    result1 = proc(msg_in)
    variance1 = _calc_smoothing_effect(result1)

    # Small sigma should have minimal effect
    assert np.allclose(variance1, original_variance, rtol=0.1)

    # Update settings - change to larger sigma (more smoothing)
    proc.update_settings(sigma=3.0)

    # Process the same message with new settings
    result2 = proc(msg_in)
    variance2 = _calc_smoothing_effect(result2)

    # Larger sigma should reduce variance (more smoothing)
    assert np.all(variance2 < variance1)

    # Test update_settings with complete new settings object
    new_settings = GaussianSmoothingSettings(
        axis="time",
        sigma=5.0,  # Even larger sigma
        width=6,
        kernel_size=None,
        coef_type="ba",
    )

    proc.update_settings(new_settings=new_settings)
    result3 = proc(msg_in)
    variance3 = _calc_smoothing_effect(result3)

    # Even larger sigma should reduce variance further
    assert np.all(variance3 < variance2)
