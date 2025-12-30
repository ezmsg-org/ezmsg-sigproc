from dataclasses import replace as dc_replace

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.ewma import (
    EWMASettings,
    EWMATransformer,
    _alpha_from_tau,
    _tau_from_alpha,
    ewma_step,
)


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
