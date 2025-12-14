import copy

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.denormalize import (
    DenormalizeSettings,
    DenormalizeTransformer,
)
from tests.helpers.util import assert_messages_equal


@pytest.fixture
def basic_input_time_ch():
    """Create a basic input with time x ch dimensions (standard normalized data)."""
    n_times = 100
    n_chans = 4
    # Normalized data with mean ~0 and std ~1
    np.random.seed(42)
    data = np.random.randn(n_times, n_chans)
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
    )


@pytest.fixture
def basic_input_ch_time():
    """Create a basic input with ch x time dimensions."""
    n_times = 100
    n_chans = 4
    np.random.seed(42)
    data = np.random.randn(n_chans, n_times)
    return AxisArray(
        data=data,
        dims=["ch", "time"],
        axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
    )


class TestDenormalizeTransformer:
    def test_uniform_distribution(self, basic_input_time_ch):
        """Test denormalization with uniform distribution."""
        backup = [copy.deepcopy(basic_input_time_ch)]

        xformer = DenormalizeTransformer(low_rate=2.0, high_rate=40.0, distribution="uniform")
        output = xformer(basic_input_time_ch)

        # Output shape should match input shape
        assert output.data.shape == basic_input_time_ch.data.shape
        assert output.dims == basic_input_time_ch.dims

        # All output values should be non-negative (clipped)
        assert np.all(output.data >= 0)

        # Offsets should be within the specified range
        assert xformer.state.offsets is not None
        assert np.all(xformer.state.offsets >= 2.0)
        assert np.all(xformer.state.offsets <= 40.0)

        # Gains should be offsets / 3.29
        assert np.allclose(xformer.state.gains, xformer.state.offsets / 3.29)

        # Verify input wasn't modified
        assert_messages_equal([basic_input_time_ch], backup)

    def test_normal_distribution(self, basic_input_time_ch):
        """Test denormalization with normal distribution."""
        backup = [copy.deepcopy(basic_input_time_ch)]

        xformer = DenormalizeTransformer(low_rate=5.0, high_rate=35.0, distribution="normal")
        output = xformer(basic_input_time_ch)

        assert output.data.shape == basic_input_time_ch.data.shape
        assert np.all(output.data >= 0)

        # Offsets should be clipped to the specified range
        assert xformer.state.offsets is not None
        assert np.all(xformer.state.offsets >= 5.0)
        assert np.all(xformer.state.offsets <= 35.0)

        # Gains should be offsets / 3.29
        assert np.allclose(xformer.state.gains, xformer.state.offsets / 3.29)

        assert_messages_equal([basic_input_time_ch], backup)

    def test_constant_distribution(self, basic_input_time_ch):
        """Test denormalization with constant distribution."""
        backup = [copy.deepcopy(basic_input_time_ch)]

        low_rate = 10.0
        high_rate = 30.0
        expected_offset = (low_rate + high_rate) / 2.0  # 20.0

        xformer = DenormalizeTransformer(low_rate=low_rate, high_rate=high_rate, distribution="constant")
        output = xformer(basic_input_time_ch)

        assert output.data.shape == basic_input_time_ch.data.shape
        assert np.all(output.data >= 0)

        # All offsets should be exactly the midpoint
        assert xformer.state.offsets is not None
        assert np.allclose(xformer.state.offsets, expected_offset)

        # Gains should be offsets / 3.29
        expected_gain = expected_offset / 3.29
        assert np.allclose(xformer.state.gains, expected_gain)

        assert_messages_equal([basic_input_time_ch], backup)

    def test_invalid_distribution(self, basic_input_time_ch):
        """Test that invalid distribution raises ValueError."""
        xformer = DenormalizeTransformer(distribution="invalid_dist")

        with pytest.raises(ValueError, match="Invalid distribution"):
            xformer(basic_input_time_ch)

    def test_ch_time_axis_order(self, basic_input_ch_time):
        """Test denormalization with ch x time axis order."""
        backup = [copy.deepcopy(basic_input_ch_time)]

        xformer = DenormalizeTransformer(distribution="constant")
        output = xformer(basic_input_ch_time)

        assert output.data.shape == basic_input_ch_time.data.shape
        assert output.dims == basic_input_ch_time.dims
        assert np.all(output.data >= 0)

        # When ch is axis 0, shape should be (nch, 1)
        n_chans = basic_input_ch_time.data.shape[0]
        assert xformer.state.offsets.shape == (n_chans, 1)
        assert xformer.state.gains.shape == (n_chans, 1)

        assert_messages_equal([basic_input_ch_time], backup)

    def test_output_clipping(self):
        """Test that negative values are clipped to zero."""
        n_times = 50
        n_chans = 2
        # Create very negative input data that should result in negative output before clipping
        data = np.full((n_times, n_chans), -10.0)  # Very negative values
        msg_in = AxisArray(
            data=data,
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        xformer = DenormalizeTransformer(low_rate=2.0, high_rate=5.0, distribution="constant")
        output = xformer(msg_in)

        # All values should be >= 0 due to clipping
        assert np.all(output.data >= 0)
        # With very negative input, most/all should be clipped to 0
        assert np.sum(output.data == 0) > 0

    def test_multiple_messages_same_state(self, basic_input_time_ch):
        """Test that state is preserved across multiple messages."""
        xformer = DenormalizeTransformer(distribution="uniform")

        # First message initializes state
        _ = xformer(basic_input_time_ch)
        gains_after_first = xformer.state.gains.copy()
        offsets_after_first = xformer.state.offsets.copy()

        # Second message should use same state
        _ = xformer(basic_input_time_ch)

        assert np.array_equal(xformer.state.gains, gains_after_first)
        assert np.array_equal(xformer.state.offsets, offsets_after_first)

    def test_denormalization_formula(self):
        """Test that the denormalization formula is applied correctly."""
        n_times = 10
        n_chans = 2
        data = np.ones((n_times, n_chans))  # All ones for predictable output
        msg_in = AxisArray(
            data=data,
            dims=["time", "ch"],
            axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
        )

        low_rate = 20.0
        high_rate = 20.0  # Same as low to make offset = 20.0
        xformer = DenormalizeTransformer(low_rate=low_rate, high_rate=high_rate, distribution="constant")
        output = xformer(msg_in)

        # With constant distribution and equal rates, offset = 20.0
        # gain = offset / 3.29 = 20.0 / 3.29
        expected_offset = 20.0
        expected_gain = expected_offset / 3.29
        # output = data * gain + offset = 1.0 * gain + offset
        expected_output = 1.0 * expected_gain + expected_offset

        assert np.allclose(output.data, expected_output)

    def test_settings_defaults(self):
        """Test that DenormalizeSettings has correct defaults."""
        settings = DenormalizeSettings()
        assert settings.low_rate == 2.0
        assert settings.high_rate == 40.0
        assert settings.distribution == "uniform"

    def test_settings_custom(self):
        """Test custom DenormalizeSettings values."""
        settings = DenormalizeSettings(low_rate=5.0, high_rate=50.0, distribution="normal")
        assert settings.low_rate == 5.0
        assert settings.high_rate == 50.0
        assert settings.distribution == "normal"

    def test_state_initialization(self, basic_input_time_ch):
        """Test that state is properly initialized on first message."""
        xformer = DenormalizeTransformer(distribution="constant")

        # Before first message, state should have None values
        assert xformer.state.gains is None
        assert xformer.state.offsets is None

        # After first message, state should be initialized
        xformer(basic_input_time_ch)

        assert xformer.state.gains is not None
        assert xformer.state.offsets is not None

    def test_output_not_sharing_memory(self, basic_input_time_ch):
        """Test that output doesn't share memory with input."""
        xformer = DenormalizeTransformer(distribution="constant")
        output = xformer(basic_input_time_ch)

        assert not np.may_share_memory(output.data, basic_input_time_ch.data)
