from dataclasses import replace as dc_replace

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.ewma import (
    EWMASettings,
    EWMATransformer,
    _alpha_from_tau,
    _tau_from_alpha,
    ewma_step,
)
from tests.helpers.empty_time import check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg
from tests.helpers.util import requires_mlx


def test_tc_from_alpha():
    # np.log(1-0.6) = -dt / tau
    alpha = 0.6
    dt = 0.01
    tau = 0.010913566679372915
    assert np.isclose(_tau_from_alpha(alpha, dt), tau)
    assert np.isclose(_alpha_from_tau(tau, dt), alpha)


def test_ewma():
    time_constant = 0.010913566679372915
    fs = 100.0
    alpha = _alpha_from_tau(time_constant, 1 / fs)
    n_times = 100
    n_ch = 32
    n_feat = 4
    data = np.arange(1, n_times * n_ch * n_feat + 1, dtype=float).reshape(n_times, n_ch, n_feat)
    msg = AxisArray(
        data=data,
        dims=["time", "ch", "feat"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": AxisArray.CoordinateAxis(data=np.arange(n_ch).astype(str), dims=["ch"]),
            "feat": AxisArray.CoordinateAxis(data=np.arange(n_feat), dims=["feat"]),
        },
    )

    # Expected
    expected = [data[0]]
    for ix, dat in enumerate(data):
        expected.append(ewma_step(dat, expected[-1], alpha))
    expected = np.stack(expected)[1:]

    ewma = EWMATransformer(time_constant=time_constant, axis="time", accumulate=True)
    res = ewma(msg)
    assert np.allclose(res.data, expected)


@requires_mlx
def test_ewma_mlx_matches_numpy_and_stays_mlx():
    mx = pytest.importorskip("mlx.core")

    fs = 30_000.0
    time_constant = 0.25
    rng = np.random.default_rng(0)
    data = rng.normal(size=(257, 8)).astype(np.float32)

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

    ewma_np = EWMATransformer(time_constant=time_constant, axis="time", accumulate=True)
    ewma_mx = EWMATransformer(time_constant=time_constant, axis="time", accumulate=True)

    out_np = ewma_np(msg_np)
    out_mx = ewma_mx(msg_mx)

    assert isinstance(out_mx.data, mx.array)
    assert isinstance(ewma_mx._state.zi, mx.array)
    np.testing.assert_allclose(np.asarray(out_mx.data), out_np.data, rtol=1e-4, atol=1e-5)

    next_data = rng.normal(size=(129, 8)).astype(np.float32)
    next_np = replace_axisarray_data(msg_np, next_data)
    next_mx = replace_axisarray_data(msg_mx, mx.array(next_data))

    out_np = ewma_np(next_np)
    out_mx = ewma_mx(next_mx)

    assert isinstance(out_mx.data, mx.array)
    np.testing.assert_allclose(np.asarray(out_mx.data), out_np.data, rtol=1e-4, atol=1e-5)


@requires_mlx
def test_ewma_mlx_accepts_numpy_initialized_state():
    mx = pytest.importorskip("mlx.core")

    fs = 30_000.0
    time_constant = 0.25
    rng = np.random.default_rng(2)
    data1 = rng.normal(size=(64, 4)).astype(np.float32)
    data2 = rng.normal(size=(129, 4)).astype(np.float32)

    msg1_np = AxisArray(data=data1, dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=fs)})
    msg2_np = AxisArray(data=data2, dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=fs)})
    msg2_mx = AxisArray(data=mx.array(data2), dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=fs)})

    ref = EWMATransformer(time_constant=time_constant, axis="time", accumulate=True)
    proc = EWMATransformer(time_constant=time_constant, axis="time", accumulate=True)

    _ = ref(msg1_np)
    expected = ref(msg2_np)

    _ = proc(msg1_np)
    assert isinstance(proc._state.zi, np.ndarray)
    actual = proc(msg2_mx)

    assert isinstance(actual.data, mx.array)
    assert isinstance(proc._state.zi, mx.array)
    np.testing.assert_allclose(np.asarray(actual.data), expected.data, rtol=1e-4, atol=1e-5)


def replace_axisarray_data(message: AxisArray, data) -> AxisArray:
    return AxisArray(data=data, dims=message.dims, axes=message.axes, key=message.key)


def _make_ewma_test_msg(data: np.ndarray, fs: float = 1000.0) -> AxisArray:
    """Helper to create test AxisArray messages."""
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs)},
    )


def test_ewma_accumulate_true_updates_state():
    """Test that accumulate=True (default) updates EWMA state."""
    ewma = EWMATransformer(settings=EWMASettings(time_constant=0.1, accumulate=True))

    # First message
    msg1 = _make_ewma_test_msg(np.ones((10, 2)))
    _ = ewma(msg1)
    state_after_first = ewma._state.zi.copy()

    # Second message with different values
    msg2 = _make_ewma_test_msg(np.ones((10, 2)) * 5.0)
    _ = ewma(msg2)
    state_after_second = ewma._state.zi.copy()

    # State should have changed
    assert not np.allclose(state_after_first, state_after_second)


def test_ewma_accumulate_false_preserves_state():
    """Test that accumulate=False does not update EWMA state."""
    ewma = EWMATransformer(settings=EWMASettings(time_constant=0.1, accumulate=True))

    # First message to initialize state
    msg1 = _make_ewma_test_msg(np.ones((10, 2)))
    _ = ewma(msg1)
    state_after_first = ewma._state.zi.copy()

    # Switch to accumulate=False
    ewma.settings = dc_replace(ewma.settings, accumulate=False)

    # Second message with very different values
    msg2 = _make_ewma_test_msg(np.ones((10, 2)) * 100.0)
    _ = ewma(msg2)
    state_after_second = ewma._state.zi.copy()

    # State should be unchanged
    assert np.allclose(state_after_first, state_after_second)


def test_ewma_accumulate_false_still_produces_output():
    """Test that accumulate=False still produces valid output."""
    ewma = EWMATransformer(settings=EWMASettings(time_constant=0.1, accumulate=True))

    # Initialize with some data
    msg1 = _make_ewma_test_msg(np.ones((50, 2)) * 10.0)
    _ = ewma(msg1)

    # Switch to accumulate=False
    ewma.settings = dc_replace(ewma.settings, accumulate=False)

    # Process more data
    msg2 = _make_ewma_test_msg(np.ones((10, 2)) * 10.0)
    out2 = ewma(msg2)

    # Output should not be empty or all zeros
    assert out2.data.shape == msg2.data.shape
    assert not np.allclose(out2.data, 0.0)


def test_ewma_accumulate_toggle():
    """Test toggling accumulate between True and False."""
    ewma = EWMATransformer(settings=EWMASettings(time_constant=0.1, accumulate=True))

    # Initialize state
    msg1 = _make_ewma_test_msg(np.ones((10, 2)))
    _ = ewma(msg1)

    # Process with accumulate=True
    msg2 = _make_ewma_test_msg(np.ones((10, 2)) * 2.0)
    _ = ewma(msg2)
    state_after_accumulate = ewma._state.zi.copy()

    # Switch to accumulate=False
    ewma.settings = dc_replace(ewma.settings, accumulate=False)

    # Process - state should not change
    msg3 = _make_ewma_test_msg(np.ones((10, 2)) * 100.0)
    _ = ewma(msg3)
    state_after_frozen = ewma._state.zi.copy()
    assert np.allclose(state_after_accumulate, state_after_frozen)

    # Switch back to accumulate=True
    ewma.settings = dc_replace(ewma.settings, accumulate=True)

    # Process - state should change again
    msg4 = _make_ewma_test_msg(np.ones((10, 2)) * 100.0)
    _ = ewma(msg4)
    state_after_resume = ewma._state.zi.copy()
    assert not np.allclose(state_after_frozen, state_after_resume)


def test_ewma_settings_default_accumulate():
    """Test that EWMASettings defaults to accumulate=True."""
    settings = EWMASettings(time_constant=1.0)
    assert settings.accumulate is True


def test_ewma_settings_accumulate_false():
    """Test that EWMASettings can be created with accumulate=False."""
    settings = EWMASettings(time_constant=1.0, accumulate=False)
    assert settings.accumulate is False


def test_ewma_empty_after_init():
    proc = EWMATransformer(settings=EWMASettings(time_constant=0.1, axis="time"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_ewma_empty_accumulate_false():
    proc = EWMATransformer(settings=EWMASettings(time_constant=0.1, axis="time", accumulate=False))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_ewma_empty_first():
    """Empty message as first input triggers _reset_state on empty data."""
    proc = EWMATransformer(settings=EWMASettings(time_constant=0.1, axis="time"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


class TestEWMAUpdateSettings:
    """Live settings updates via BaseProcessor.update_settings."""

    def test_accumulate_toggle_preserves_state(self):
        proc = EWMATransformer(settings=EWMASettings(time_constant=0.1, accumulate=True))

        _ = proc(_make_ewma_test_msg(np.ones((10, 2))))
        _ = proc(_make_ewma_test_msg(np.ones((10, 2)) * 2.0))
        zi_before = proc._state.zi.copy()
        alpha_before = proc._state.alpha

        # `accumulate` is in NONRESET_SETTINGS_FIELDS — no reset queued.
        proc.update_settings(EWMASettings(time_constant=0.1, accumulate=False))
        assert proc._hash != -1
        assert np.allclose(proc._state.zi, zi_before)
        assert proc._state.alpha == alpha_before

        # accumulate=False must now gate state updates inside _process.
        _ = proc(_make_ewma_test_msg(np.ones((10, 2)) * 100.0))
        assert np.allclose(proc._state.zi, zi_before)

    def test_time_constant_change_recomputes_alpha(self):
        proc = EWMATransformer(settings=EWMASettings(time_constant=0.1, accumulate=True))

        msg = _make_ewma_test_msg(np.ones((10, 2)))
        _ = proc(msg)
        alpha_before = proc._state.alpha

        # `time_constant` is NOT in NONRESET_SETTINGS_FIELDS — reset queued.
        proc.update_settings(EWMASettings(time_constant=0.5, accumulate=True))
        assert proc._hash == -1

        # Next message triggers _reset_state, which recomputes alpha from the new time_constant.
        _ = proc(msg)
        alpha_after = proc._state.alpha
        assert alpha_after != alpha_before
        assert np.isclose(alpha_after, _alpha_from_tau(0.5, 1 / 1000.0))
