import copy
import importlib.util

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.chunker import array_chunker
from frozendict import frozendict

from ezmsg.sigproc.scaler import (
    AdaptiveStandardScalerSettings,
    AdaptiveStandardScalerTransformer,
    scaler,
)
from tests.helpers.util import assert_messages_equal


@pytest.fixture
def fixture_arrays():
    # Test data values taken from river:
    # https://github.com/online-ml/river/blob/main/river/preprocessing/scale.py#L511-L536C17
    data = np.array([5.278, 5.050, 6.550, 7.446, 9.472, 10.353, 11.784, 11.173])
    expected_result = np.array([0.0, -0.816, 0.812, 0.695, 0.754, 0.598, 0.651, 0.124])
    return data, expected_result


@pytest.mark.skipif(importlib.util.find_spec("river") is None, reason="requires `river` package")
def test_adaptive_standard_scaler_river(fixture_arrays):
    data, expected_result = fixture_arrays

    test_input = AxisArray(
        np.tile(data, (2, 1)),
        dims=["ch", "time"],
        axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
    )

    backup = [copy.deepcopy(test_input)]

    # The River example used alpha = 0.6
    # tau = -gain / np.log(1 - alpha) and here we're using gain = 0.01
    tau = 0.010913566679372915
    _scaler = scaler(time_constant=tau, axis="time")
    output = _scaler.send(test_input)
    assert np.allclose(output.data[0], expected_result, atol=1e-3)
    assert_messages_equal([test_input], backup)


def test_scaler(fixture_arrays):
    data, expected_result = fixture_arrays
    chunker = array_chunker(data, 4, fs=100.0)
    test_input = list(chunker)
    backup = copy.deepcopy(test_input)
    tau = 0.010913566679372915

    xformer = AdaptiveStandardScalerTransformer(time_constant=tau, axis="time")
    outputs = []
    for chunk in test_input:
        outputs.append(xformer(chunk))
    output = AxisArray.concatenate(*outputs, dim="time")
    assert np.allclose(output.data, expected_result, atol=1e-3)
    assert_messages_equal(test_input, backup)


def _make_scaler_test_msg(data: np.ndarray, fs: float = 1000.0) -> AxisArray:
    """Helper to create test AxisArray messages."""
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs)},
    )


class TestAdaptiveStandardScalerAccumulate:
    """Tests for the accumulate setting on AdaptiveStandardScalerTransformer."""

    def test_settings_default_accumulate(self):
        """Test that AdaptiveStandardScalerSettings defaults to accumulate=True."""
        settings = AdaptiveStandardScalerSettings(time_constant=1.0)
        assert settings.accumulate is True

    def test_settings_accumulate_false(self):
        """Test that settings can be created with accumulate=False."""
        settings = AdaptiveStandardScalerSettings(time_constant=1.0, accumulate=False)
        assert settings.accumulate is False

    def test_accumulate_true_updates_state(self):
        """Test that accumulate=True updates internal EWMA states."""
        scaler = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=True)
        )

        # First message to initialize
        np.random.seed(42)
        msg1 = _make_scaler_test_msg(np.random.randn(100, 4))
        _ = scaler(msg1)
        zi1 = scaler._state.samps_ewma._state.zi.copy()

        # Second message with shifted mean
        msg2 = _make_scaler_test_msg(np.random.randn(100, 4) + 10.0)
        _ = scaler(msg2)
        zi2 = scaler._state.samps_ewma._state.zi.copy()

        # State should have changed
        assert not np.allclose(zi1, zi2)

    def test_accumulate_false_preserves_state(self):
        """Test that accumulate=False does not update internal EWMA states."""
        scaler = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=True)
        )

        # First message to initialize
        np.random.seed(42)
        msg1 = _make_scaler_test_msg(np.random.randn(100, 4))
        _ = scaler(msg1)
        zi1 = scaler._state.samps_ewma._state.zi.copy()

        # Switch to accumulate=False via property
        scaler.accumulate = False

        # Second message with very different values
        msg2 = _make_scaler_test_msg(np.random.randn(100, 4) + 100.0)
        _ = scaler(msg2)
        zi2 = scaler._state.samps_ewma._state.zi.copy()

        # State should be unchanged
        assert np.allclose(zi1, zi2)

    def test_accumulate_property_getter(self):
        """Test the accumulate property getter."""
        scaler = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=True)
        )
        assert scaler.accumulate is True

        scaler2 = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=False)
        )
        assert scaler2.accumulate is False

    def test_accumulate_property_setter_propagates_to_children(self):
        """Test that setting accumulate propagates to child EWMA transformers."""
        scaler = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=True)
        )

        # Initialize state by processing a message
        msg = _make_scaler_test_msg(np.random.randn(10, 2))
        _ = scaler(msg)

        # Verify initial state
        assert scaler._state.samps_ewma.settings.accumulate is True
        assert scaler._state.vars_sq_ewma.settings.accumulate is True

        # Change via property
        scaler.accumulate = False

        # Verify propagation
        assert scaler._state.samps_ewma.settings.accumulate is False
        assert scaler._state.vars_sq_ewma.settings.accumulate is False

        # Change back
        scaler.accumulate = True
        assert scaler._state.samps_ewma.settings.accumulate is True
        assert scaler._state.vars_sq_ewma.settings.accumulate is True

    def test_accumulate_false_still_produces_output(self):
        """Test that accumulate=False still produces valid z-scored output."""
        scaler = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=True)
        )

        # Initialize with some data
        np.random.seed(42)
        msg1 = _make_scaler_test_msg(np.random.randn(100, 4) * 5.0 + 10.0)
        _ = scaler(msg1)

        # Switch to accumulate=False
        scaler.accumulate = False

        # Process more data
        msg2 = _make_scaler_test_msg(np.random.randn(50, 4) * 5.0 + 10.0)
        out2 = scaler(msg2)

        # Output should have correct shape and be roughly z-scored
        assert out2.data.shape == msg2.data.shape
        # Z-scores should be reasonable (not NaN, not extreme)
        assert not np.any(np.isnan(out2.data))
        assert np.abs(out2.data).max() < 100  # Sanity check

    def test_accumulate_toggle(self):
        """Test toggling accumulate between True and False."""
        scaler = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=True)
        )

        # Initialize
        np.random.seed(42)
        msg1 = _make_scaler_test_msg(np.random.randn(50, 4))
        _ = scaler(msg1)

        # Accumulate more
        msg2 = _make_scaler_test_msg(np.random.randn(50, 4) + 5.0)
        _ = scaler(msg2)
        zi_after_accumulate = scaler._state.samps_ewma._state.zi.copy()

        # Freeze
        scaler.accumulate = False
        msg3 = _make_scaler_test_msg(np.random.randn(50, 4) + 100.0)
        _ = scaler(msg3)
        zi_after_frozen = scaler._state.samps_ewma._state.zi.copy()
        assert np.allclose(zi_after_accumulate, zi_after_frozen)

        # Resume accumulation
        scaler.accumulate = True
        msg4 = _make_scaler_test_msg(np.random.randn(50, 4) + 100.0)
        _ = scaler(msg4)
        zi_after_resume = scaler._state.samps_ewma._state.zi.copy()
        assert not np.allclose(zi_after_frozen, zi_after_resume)

    def test_initial_accumulate_false(self):
        """Test starting with accumulate=False from initialization."""
        scaler = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=False)
        )

        # First message initializes state but with accumulate=False
        msg1 = _make_scaler_test_msg(np.ones((50, 4)))
        _ = scaler(msg1)

        # Verify child EWMAs inherited the setting
        assert scaler._state.samps_ewma.settings.accumulate is False
        assert scaler._state.vars_sq_ewma.settings.accumulate is False
