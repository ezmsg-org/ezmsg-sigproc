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
    warmup_data = rng.normal(size=(256, 16)).astype(np.float32)
    data = rng.normal(size=(1025, 16)).astype(np.float32)

    warmup_msg_np = AxisArray(
        data=warmup_data,
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs)},
    )
    warmup_msg_mx = AxisArray(
        data=mx.array(warmup_data),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs)},
    )
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

    _ = scaler_np(warmup_msg_np)
    _ = scaler_mx(warmup_msg_mx)
    out_np = scaler_np(msg_np)
    out_mx = scaler_mx(msg_mx)

    assert isinstance(out_mx.data, mx.array)
    assert isinstance(scaler_mx._state.samps_ewma._state.zi, mx.array)
    assert isinstance(scaler_mx._state.vars_sq_ewma._state.zi, mx.array)
    np.testing.assert_allclose(np.asarray(out_mx.data), out_np.data, rtol=2e-4, atol=2e-4)


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


class TestAdaptiveStandardScalerPassthrough:
    """Tests for the passthrough setting on AdaptiveStandardScalerTransformer."""

    def test_passthrough_identity(self):
        """passthrough=True returns the input unchanged — no scaling."""
        scaler = AdaptiveStandardScalerTransformer(
            settings=AdaptiveStandardScalerSettings(time_constant=0.1, passthrough=True)
        )

        msg = _make_scaler_test_msg(np.arange(20, dtype=float).reshape(10, 2) + 50.0)
        out = scaler(msg)

        assert out is msg
        # No state was ever initialized — the scaler was skipped entirely.
        assert scaler._state.samps_ewma is None

    def test_passthrough_toggle_preserves_state(self):
        """Toggling passthrough does not reset the child EWMA states."""
        scaler = AdaptiveStandardScalerTransformer(settings=AdaptiveStandardScalerSettings(time_constant=0.1))

        np.random.seed(42)
        _ = scaler(_make_scaler_test_msg(np.random.randn(100, 4)))
        zi_samps = scaler._state.samps_ewma._state.zi.copy()
        zi_vars = scaler._state.vars_sq_ewma._state.zi.copy()

        # Passthrough: identity output, state untouched even with wild input.
        scaler.update_settings(AdaptiveStandardScalerSettings(time_constant=0.1, passthrough=True))
        assert scaler._hash != -1
        msg = _make_scaler_test_msg(np.random.randn(50, 4) + 1000.0)
        assert scaler(msg) is msg
        assert np.allclose(scaler._state.samps_ewma._state.zi, zi_samps)
        assert np.allclose(scaler._state.vars_sq_ewma._state.zi, zi_vars)

        # Resume scaling: statistics pick up where they left off.
        scaler.update_settings(AdaptiveStandardScalerSettings(time_constant=0.1, passthrough=False))
        out = scaler(_make_scaler_test_msg(np.random.randn(50, 4)))
        assert out.data.shape == (50, 4)
        assert not np.any(np.isnan(out.data))


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


# ---------------------------------------------------------------------------
# init_mean / init_std: seed the scaler to avoid a transient first sample
# anchoring the running statistics (the "bad seed" cold-start problem).
# ---------------------------------------------------------------------------


def _scaler_first_sample_transient(fs=100.0, n=1000, transient=100.0, seed=0):
    """A 1-channel signal that opens with a big transient sample then settles to
    zero-mean unit-variance noise -- the classic filter-edge cold-start case."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1))
    x[0, 0] = transient
    msg = AxisArray(
        x,
        dims=["time", "ch"],
        axes=frozendict(
            {"time": AxisArray.TimeAxis(fs=fs), "ch": AxisArray.CoordinateAxis(data=np.array(["0"]), dims=["ch"])}
        ),
    )
    return x, msg


def test_scaler_default_seed_unchanged(fixture_arrays):
    """init_mean/init_std default to None and reproduce the river-matched
    first-sample seed (existing behavior)."""
    assert AdaptiveStandardScalerSettings().init_mean is None
    assert AdaptiveStandardScalerSettings().init_std is None
    data, expected_result = fixture_arrays
    test_input = AxisArray(
        data[:, None], dims=["time", "ch"], axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)})
    )
    tau = 0.010913566679372915
    out = AdaptiveStandardScalerTransformer(time_constant=tau, axis="time")(test_input)
    assert np.allclose(out.data[:, 0], expected_result, atol=1e-3)


def test_scaler_init_seed_first_output_is_zscore():
    """With init_mean/init_std the first output is a proper z-score from the
    seed (~ (x0 - mean)/std), not 0 (the unseeded var==0 first-sample case)."""
    _, msg = _scaler_first_sample_transient(transient=5.0)
    x0 = float(msg.data[0, 0])

    # Unseeded: on the first sample var == 0, so the output is 0.
    unseeded = AdaptiveStandardScalerTransformer(time_constant=1.0, axis="time")(msg)
    assert np.isclose(unseeded.data[0, 0], 0.0)

    # Seeded with a large tau (alpha ~ 0) so the first-sample update is
    # negligible: first output ~= (x0 - init_mean) / init_std = x0.
    seeded = AdaptiveStandardScalerTransformer(
        time_constant=100.0, axis="time", init_mean=0.0, init_std=1.0
    )(msg)
    assert np.isclose(seeded.data[0, 0], x0, atol=0.1)


def test_scaler_init_seed_avoids_cold_start_collapse():
    """Without a seed, a transient first sample anchors the running mean and
    the following samples collapse negative for ~3*tau. Seeding with the true
    baseline (0, 1) keeps the z-score well-behaved from the start."""
    _, msg = _scaler_first_sample_transient(fs=100.0, transient=100.0)

    unseeded = AdaptiveStandardScalerTransformer(time_constant=1.0, axis="time")(msg)
    seeded = AdaptiveStandardScalerTransformer(
        time_constant=1.0, axis="time", init_mean=0.0, init_std=1.0
    )(msg)

    # Post-transient window (still inside the ~3*tau = 3 s = 300-sample warmup).
    win = slice(10, 100)
    unseeded_mean = float(unseeded.data[win, 0].mean())
    seeded_mean = float(seeded.data[win, 0].mean())
    assert unseeded_mean < -0.8  # collapsed (running mean anchored high by transient)
    assert abs(seeded_mean) < 0.5  # de-biased: centered near 0, not collapsed
    assert seeded_mean - unseeded_mean > 0.8  # seeding clearly de-biases the warmup
    # NB: seeding fixes the mean-anchor collapse; the transient sample still
    # enters the variance EWMA (compressing the z-score) until it decays -- that
    # residual is removed upstream by not producing the transient (edge_scale_zi).


def test_scaler_init_seed_per_channel():
    """Per-channel array init_mean/init_std seed each channel independently, so
    the first output is a per-channel z-score against that channel's baseline."""
    fs = 100.0
    rng = np.random.default_rng(3)
    means = np.array([10.0, -5.0, 0.0])
    stds = np.array([2.0, 0.5, 1.0])
    x = rng.standard_normal((500, 3)) * stds + means
    msg = AxisArray(x, dims=["time", "ch"], axes=frozendict({"time": AxisArray.TimeAxis(fs=fs)}))

    # Large tau (alpha ~ 0) so the first-sample update is negligible and the
    # first output reflects the seed: (x0 - init_mean) / init_std per channel.
    seeded = AdaptiveStandardScalerTransformer(
        time_constant=100.0, axis="time", init_mean=means, init_std=stds
    )(msg)

    expected0 = (x[0] - means) / stds
    np.testing.assert_allclose(seeded.data[0], expected0, atol=1e-2)


def test_scaler_init_seed_accumulate_false():
    """With accumulate=False the seeded stats are applied but not advanced, so
    two identical messages produce identical z-scores from the seed (rather
    than the unseeded var==0 first output of 0)."""
    fs = 100.0
    x = np.full((10, 1), 7.0)

    def mk() -> AxisArray:
        return AxisArray(x.copy(), dims=["time", "ch"], axes=frozendict({"time": AxisArray.TimeAxis(fs=fs)}))

    proc = AdaptiveStandardScalerTransformer(
        time_constant=100.0, axis="time", accumulate=False, init_mean=0.0, init_std=1.0
    )
    out1 = proc(mk())
    out2 = proc(mk())

    # First output reflects the seed ((x0 - 0)/1 ~= 7), not the unseeded 0...
    assert out1.data[0, 0] > 5.0
    # ...and the state is frozen, so a second identical message is identical.
    assert np.allclose(out1.data, out2.data)
