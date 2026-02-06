from dataclasses import replace as dc_replace

import numpy as np
import scipy.signal as sps
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.detrend import DetrendTransformer
from ezmsg.sigproc.ewma import EWMASettings, _alpha_from_tau
from tests.helpers.empty_time import check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg

# --- Helpers ---


def _make_msg(data: np.ndarray, fs: float = 1000.0, dims=None, axes=None) -> AxisArray:
    if dims is None:
        dims = ["time", "ch"]
    if axes is None:
        axes = {"time": AxisArray.TimeAxis(fs=fs)}
    return AxisArray(data=data, dims=dims, axes=axes)


# --- Empty-time edge cases (pre-existing) ---


def test_detrend_empty_after_init():
    proc = DetrendTransformer(settings=EWMASettings(time_constant=0.1, axis="time"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_detrend_empty_first():
    """Empty message as first input triggers _reset_state on empty data."""
    proc = DetrendTransformer(settings=EWMASettings(time_constant=0.1, axis="time"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


# --- Basic functionality ---


def test_detrend_constant_signal():
    """Constant input should detrend to ~0 after initial transient."""
    fs = 1000.0
    tc = 0.01  # short time constant for fast convergence
    n_samples = 500
    n_ch = 2
    data = np.ones((n_samples, n_ch)) * 5.0
    msg = _make_msg(data, fs=fs)

    proc = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))
    result = proc(msg)

    # After the transient (last 50%), output should be ~0
    assert np.allclose(result.data[n_samples // 2 :], 0.0, atol=1e-3)


def test_detrend_removes_dc_offset():
    """Signal with DC offset: detrended output mean should be near 0."""
    fs = 1000.0
    tc = 0.01
    n_samples = 1000
    dc_offset = 10.0
    noise = np.random.RandomState(42).randn(n_samples, 1) * 0.1
    data = noise + dc_offset
    msg = _make_msg(data, fs=fs)

    proc = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))
    result = proc(msg)

    # Mean of the latter half should be close to 0
    assert abs(result.data[n_samples // 2 :].mean()) < 0.5


def test_detrend_preserves_oscillation():
    """High-freq sine + DC offset: detrend removes DC, preserves oscillation."""
    fs = 1000.0
    tc = 0.1  # long enough to not track the sine
    n_samples = 2000
    t = np.arange(n_samples) / fs
    freq = 50.0  # Hz — fast relative to time constant
    dc_offset = 5.0
    sine = np.sin(2 * np.pi * freq * t)
    data = (sine + dc_offset).reshape(-1, 1)
    msg = _make_msg(data, fs=fs)

    proc = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))
    result = proc(msg)

    # Latter half: DC should be removed, oscillation preserved
    latter = result.data[n_samples // 2 :, 0]
    assert abs(latter.mean()) < 0.5  # DC removed
    assert latter.std() > 0.5  # oscillation still present


# --- Numerical correctness ---


def test_detrend_matches_reference():
    """Compare output against manual data - lfilter(...) computation."""
    fs = 500.0
    tc = 0.05
    n_samples = 200
    n_ch = 3
    rng = np.random.RandomState(123)
    data = rng.randn(n_samples, n_ch) + 3.0
    msg = _make_msg(data, fs=fs)

    proc = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))
    result = proc(msg)

    # Compute reference manually
    alpha = _alpha_from_tau(tc, 1.0 / fs)
    zi = (1 - alpha) * data[:1, :]  # same as _reset_state
    means, _ = sps.lfilter(
        [alpha],
        [1.0, alpha - 1.0],
        data,
        axis=0,
        zi=zi,
    )
    expected = data - means

    assert np.allclose(result.data, expected, atol=1e-12)


# --- Multi-chunk streaming ---


def test_detrend_multi_chunk_vs_single():
    """Chunked processing should match single-chunk processing."""
    fs = 1000.0
    tc = 0.05
    n_samples = 300
    n_ch = 2
    rng = np.random.RandomState(7)
    data = rng.randn(n_samples, n_ch) + 2.0

    # Single chunk
    proc_single = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))
    result_single = proc_single(_make_msg(data, fs=fs))

    # Multi chunk
    proc_multi = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))
    chunk_sizes = [50, 100, 80, 70]  # sums to 300
    results = []
    for start, size in zip(np.cumsum([0] + chunk_sizes[:-1]), chunk_sizes):
        chunk = data[start : start + size]
        results.append(proc_multi(_make_msg(chunk, fs=fs)).data)
    result_multi = np.concatenate(results, axis=0)

    assert np.allclose(result_single.data, result_multi, atol=1e-12)


# --- Time constant behavior ---


def test_detrend_time_constant_effect():
    """Shorter time constant adapts faster (smaller residual after step change)."""
    fs = 1000.0
    # Step change: 10 samples at 0, then 40 samples at 10
    data = np.concatenate([np.zeros((10, 1)), np.ones((40, 1)) * 10.0])

    proc_short = DetrendTransformer(settings=EWMASettings(time_constant=0.01, axis="time"))
    proc_long = DetrendTransformer(settings=EWMASettings(time_constant=1.0, axis="time"))

    result_short = proc_short(_make_msg(data, fs=fs))
    result_long = proc_long(_make_msg(data, fs=fs))

    # At the end, shorter tc should have smaller residual (closer to 0)
    residual_short = abs(result_short.data[-1, 0])
    residual_long = abs(result_long.data[-1, 0])
    assert residual_short < residual_long


# --- Axis handling ---


def test_detrend_axis_default():
    """axis=None defaults to first dim."""
    fs = 1000.0
    tc = 0.05
    n_samples = 100
    n_ch = 2
    data = np.random.RandomState(0).randn(n_samples, n_ch) + 5.0
    msg = _make_msg(data, fs=fs)

    proc_default = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis=None))
    proc_explicit = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))

    result_default = proc_default(msg)
    result_explicit = proc_explicit(_make_msg(data, fs=fs))

    assert np.allclose(result_default.data, result_explicit.data, atol=1e-12)


def test_detrend_nondefault_axis():
    """Process along a non-time axis (channel axis)."""
    fs = 1000.0
    tc = 0.05
    n_time = 3
    n_ch = 10
    # Each row has a different offset — detrending along ch should remove per-row means
    rng = np.random.RandomState(42)
    data = rng.randn(n_time, n_ch)
    msg = AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": AxisArray.Axis(gain=1.0),
        },
    )

    proc = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="ch"))
    result = proc(msg)

    assert result.data.shape == data.shape
    # Verify output differs from input (detrend did something)
    assert not np.allclose(result.data, data)
    # Verify it processes along ch axis, not time: each row should be processed
    # independently, so results for row 0 don't depend on row 1.
    proc2 = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="ch"))
    result_row0 = proc2(
        _make_msg(
            data[:1, :],
            fs=fs,
            dims=["time", "ch"],
            axes={"time": AxisArray.TimeAxis(fs=fs), "ch": AxisArray.Axis(gain=1.0)},
        )
    )
    assert np.allclose(result.data[0], result_row0.data[0])


# --- State management ---


def test_detrend_state_reset_on_fs_change():
    """Different fs triggers state reset (hash changes due to gain change)."""
    tc = 0.05
    n_samples = 100
    n_ch = 2
    data = np.ones((n_samples, n_ch)) * 5.0

    proc = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))

    # Process at fs=1000
    _ = proc(_make_msg(data, fs=1000.0))
    alpha_first = proc._state.alpha

    # Process at fs=500 — different gain → different hash → reset
    _ = proc(_make_msg(data, fs=500.0))
    alpha_second = proc._state.alpha

    assert alpha_first != alpha_second
    # Alpha should correspond to the new fs
    expected_alpha = _alpha_from_tau(tc, 1.0 / 500.0)
    assert np.isclose(alpha_second, expected_alpha)


# --- Shape handling ---


def test_detrend_output_shape_matches_input():
    """Output shape == input shape for various dimensions."""
    fs = 1000.0
    tc = 0.05

    shapes_and_dims = [
        ((50,), ["time"]),
        ((50, 3), ["time", "ch"]),
        ((50, 3, 4), ["time", "ch", "feat"]),
    ]
    for shape, dims in shapes_and_dims:
        data = np.random.RandomState(0).randn(*shape)
        axes = {"time": AxisArray.TimeAxis(fs=fs)}
        msg = AxisArray(data=data, dims=dims, axes=axes)
        proc = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))
        result = proc(msg)
        assert result.data.shape == shape, f"Shape mismatch for input shape {shape}"


def test_detrend_multichannel_independence():
    """Each channel detrended independently (per-channel offsets removed)."""
    fs = 1000.0
    tc = 0.01
    n_samples = 500
    n_ch = 3
    rng = np.random.RandomState(99)
    offsets = np.array([1.0, 10.0, 100.0])
    data = rng.randn(n_samples, n_ch) * 0.01 + offsets
    msg = _make_msg(data, fs=fs)

    proc = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))
    result = proc(msg)

    # Latter half: each channel's mean should be near 0 despite different offsets
    latter = result.data[n_samples // 2 :]
    for ch in range(n_ch):
        assert abs(latter[:, ch].mean()) < 0.1, f"Channel {ch} not detrended"


# --- Edge cases ---


def test_detrend_single_sample():
    """Single time-step message processes without error."""
    fs = 1000.0
    tc = 0.05
    data = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3)
    msg = _make_msg(data, fs=fs)

    proc = DetrendTransformer(settings=EWMASettings(time_constant=tc, axis="time"))
    result = proc(msg)

    assert result.data.shape == (1, 3)
    # For a single sample, EWMA initialized from that sample, so detrend ≈ 0
    # zi = (1 - alpha) * data, means = alpha * data + zi = alpha*data + (1-alpha)*data = data
    # So result = data - data = 0... but lfilter with zi works slightly differently.
    # Just verify it doesn't error and produces finite output.
    assert np.all(np.isfinite(result.data))


# --- Accumulate flag ---


def test_detrend_accumulate_true_updates_state():
    """accumulate=True (default) updates EWMA state."""
    proc = DetrendTransformer(settings=EWMASettings(time_constant=0.1, accumulate=True))

    msg1 = _make_msg(np.ones((10, 2)))
    _ = proc(msg1)
    zi_after_first = proc._state.zi.copy()

    msg2 = _make_msg(np.ones((10, 2)) * 5.0)
    _ = proc(msg2)
    zi_after_second = proc._state.zi.copy()

    assert not np.allclose(zi_after_first, zi_after_second)


def test_detrend_accumulate_false_preserves_state():
    """accumulate=False does not update EWMA state."""
    proc = DetrendTransformer(settings=EWMASettings(time_constant=0.1, accumulate=True))

    # Initialize state
    msg1 = _make_msg(np.ones((10, 2)))
    _ = proc(msg1)
    zi_after_init = proc._state.zi.copy()

    # Switch to accumulate=False
    proc.settings = dc_replace(proc.settings, accumulate=False)

    # Process very different values — state should not change
    msg2 = _make_msg(np.ones((10, 2)) * 100.0)
    _ = proc(msg2)
    zi_after_frozen = proc._state.zi.copy()

    assert np.allclose(zi_after_init, zi_after_frozen)


def test_detrend_accumulate_false_still_produces_output():
    """accumulate=False still produces valid detrended output."""
    proc = DetrendTransformer(settings=EWMASettings(time_constant=0.1, accumulate=True))

    msg1 = _make_msg(np.ones((50, 2)) * 10.0)
    _ = proc(msg1)

    proc.settings = dc_replace(proc.settings, accumulate=False)

    msg2 = _make_msg(np.ones((10, 2)) * 10.0)
    out = proc(msg2)

    assert out.data.shape == msg2.data.shape
    assert np.all(np.isfinite(out.data))


def test_detrend_accumulate_toggle():
    """Toggling accumulate between True and False works correctly."""
    proc = DetrendTransformer(settings=EWMASettings(time_constant=0.1, accumulate=True))

    # Initialize
    _ = proc(_make_msg(np.ones((10, 2))))

    # Accumulate=True: state changes
    _ = proc(_make_msg(np.ones((10, 2)) * 2.0))
    zi_after_accum = proc._state.zi.copy()

    # Accumulate=False: state frozen
    proc.settings = dc_replace(proc.settings, accumulate=False)
    _ = proc(_make_msg(np.ones((10, 2)) * 100.0))
    zi_after_frozen = proc._state.zi.copy()
    assert np.allclose(zi_after_accum, zi_after_frozen)

    # Accumulate=True again: state changes
    proc.settings = dc_replace(proc.settings, accumulate=True)
    _ = proc(_make_msg(np.ones((10, 2)) * 100.0))
    zi_after_resume = proc._state.zi.copy()
    assert not np.allclose(zi_after_frozen, zi_after_resume)
