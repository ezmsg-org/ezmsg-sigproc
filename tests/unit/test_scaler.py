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
    RiverAdaptiveStandardScalerSettings,
    RiverAdaptiveStandardScalerTransformer,
)
from tests.helpers.empty_time import check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg
from tests.helpers.util import assert_messages_equal, requires_mlx


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
    _scaler = RiverAdaptiveStandardScalerTransformer(
        settings=RiverAdaptiveStandardScalerSettings(time_constant=tau, axis="time")
    )
    output = _scaler(test_input)
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


@requires_mlx
def test_scaler_mlx_matches_numpy_and_stays_mlx():
    mx = pytest.importorskip("mlx.core")

    fs = 30_000.0
    tau = 0.5
    rng = np.random.default_rng(1)
    data = rng.normal(size=(1025, 16)).astype(np.float32)

    msg_np = AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs)},
    )
    msg_mx = AxisArray(
        data=mx.array(data),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs)},
    )

    scaler_np = AdaptiveStandardScalerTransformer(
        settings=AdaptiveStandardScalerSettings(time_constant=tau, axis="time")
    )
    scaler_mx = AdaptiveStandardScalerTransformer(
        settings=AdaptiveStandardScalerSettings(time_constant=tau, axis="time")
    )

    out_np = scaler_np(msg_np)
    out_mx = scaler_mx(msg_mx)

    assert isinstance(out_mx.data, mx.array)
    assert isinstance(scaler_mx._state.samps_ewma._state.zi, mx.array)
    assert isinstance(scaler_mx._state.vars_sq_ewma._state.zi, mx.array)
    out_mx_np = np.asarray(out_mx.data)
    # The first few post-initialization samples divide by a near-zero variance;
    # float32 MLX and float64 SciPy can differ by ~0.1% while still matching
    # once the EWMA variance has a few samples of support.
    np.testing.assert_allclose(out_mx_np[:4], out_np.data[:4], rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(out_mx_np[4:], out_np.data[4:], rtol=2e-4, atol=2e-4)


@requires_mlx
def test_scaler_mlx_accepts_numpy_initialized_state():
    mx = pytest.importorskip("mlx.core")

    fs = 30_000.0
    tau = 0.5
    rng = np.random.default_rng(3)
    data1 = rng.normal(size=(64, 4)).astype(np.float32)
    data2 = rng.normal(size=(257, 4)).astype(np.float32)

    msg1_np = AxisArray(data=data1, dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=fs)})
    msg2_np = AxisArray(data=data2, dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=fs)})
    msg2_mx = AxisArray(data=mx.array(data2), dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=fs)})

    ref = AdaptiveStandardScalerTransformer(settings=AdaptiveStandardScalerSettings(time_constant=tau, axis="time"))
    proc = AdaptiveStandardScalerTransformer(settings=AdaptiveStandardScalerSettings(time_constant=tau, axis="time"))

    _ = ref(msg1_np)
    expected = ref(msg2_np)

    _ = proc(msg1_np)
    assert isinstance(proc._state.samps_ewma._state.zi, np.ndarray)
    actual = proc(msg2_mx)

    assert isinstance(actual.data, mx.array)
    assert isinstance(proc._state.samps_ewma._state.zi, mx.array)
    assert isinstance(proc._state.vars_sq_ewma._state.zi, mx.array)
    np.testing.assert_allclose(np.asarray(actual.data), expected.data, rtol=2e-4, atol=2e-4)


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


class TestAdaptiveStandardScalerUpdateSettings:
    """Live settings updates via update_settings."""

    def test_accumulate_update_propagates_and_preserves_state(self):
        scaler = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=True)
        )

        np.random.seed(0)
        _ = scaler(_make_scaler_test_msg(np.random.randn(50, 4)))
        _ = scaler(_make_scaler_test_msg(np.random.randn(50, 4) + 5.0))
        zi_samps = scaler._state.samps_ewma._state.zi.copy()
        zi_vars = scaler._state.vars_sq_ewma._state.zi.copy()

        # accumulate-only change: should propagate to children without reset.
        scaler.update_settings(AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=False))
        assert scaler._hash != -1
        assert scaler._state.samps_ewma.settings.accumulate is False
        assert scaler._state.vars_sq_ewma.settings.accumulate is False

        # With accumulate=False, child zi values must not move.
        _ = scaler(_make_scaler_test_msg(np.random.randn(50, 4) + 100.0))
        assert np.allclose(scaler._state.samps_ewma._state.zi, zi_samps)
        assert np.allclose(scaler._state.vars_sq_ewma._state.zi, zi_vars)

    def test_time_constant_update_triggers_reset(self):
        scaler = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, accumulate=True)
        )
        _ = scaler(_make_scaler_test_msg(np.ones((10, 2))))
        samps_ewma_before = scaler._state.samps_ewma

        scaler.update_settings(AdaptiveStandardScalerSettings(time_constant=0.5, accumulate=True))
        assert scaler._hash == -1

        # Next message rebuilds the child EWMAs inside _reset_state.
        _ = scaler(_make_scaler_test_msg(np.ones((10, 2))))
        assert scaler._state.samps_ewma is not samps_ewma_before


def test_adaptive_scaler_np_empty_after_init():
    from ezmsg.sigproc.scaler import AdaptiveStandardScalerSettings, AdaptiveStandardScalerTransformer

    proc = AdaptiveStandardScalerTransformer(settings=AdaptiveStandardScalerSettings(time_constant=0.1, axis="time"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


@pytest.mark.skipif(
    importlib.util.find_spec("river") is None,
    reason="requires `river` package",
)
def test_adaptive_scaler_river_empty_after_init():
    from ezmsg.sigproc.scaler import (
        RiverAdaptiveStandardScalerSettings,
        RiverAdaptiveStandardScalerTransformer,
    )

    proc = RiverAdaptiveStandardScalerTransformer(
        settings=RiverAdaptiveStandardScalerSettings(time_constant=0.1, axis="time")
    )
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_adaptive_scaler_np_empty_first():
    from ezmsg.sigproc.scaler import AdaptiveStandardScalerSettings, AdaptiveStandardScalerTransformer

    proc = AdaptiveStandardScalerTransformer(settings=AdaptiveStandardScalerSettings(time_constant=0.1, axis="time"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


@pytest.mark.skipif(
    importlib.util.find_spec("river") is None,
    reason="requires `river` package",
)
def test_adaptive_scaler_river_empty_first():
    from ezmsg.sigproc.scaler import (
        RiverAdaptiveStandardScalerSettings,
        RiverAdaptiveStandardScalerTransformer,
    )

    proc = RiverAdaptiveStandardScalerTransformer(
        settings=RiverAdaptiveStandardScalerSettings(time_constant=0.1, axis="time")
    )
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)
