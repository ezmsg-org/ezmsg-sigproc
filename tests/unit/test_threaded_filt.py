"""Threaded scipy IIR filtering: bit-identity, gating, and state handling.

The load-bearing property is that splitting channels across threads changes
nothing about the arithmetic any one channel sees. These tests assert
``array_equal`` rather than ``allclose`` because "close" would defeat the point:
this exists to speed up offline runs of a pipeline whose output must match the
online run exactly.
"""

import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.filter import FilterSettings, FilterTransformer
from ezmsg.sigproc.util import threaded_filt

FS = 1000.0


def _sos(order=4):
    return scipy.signal.butter(order, [30.0, 200.0], btype="bandpass", fs=FS, output="sos")


def _ba(order=4):
    return scipy.signal.butter(order, [30.0, 200.0], btype="bandpass", fs=FS, output="ba")


def _zi_sos(sos, shape, axis_idx):
    zi = scipy.signal.sosfilt_zi(sos)
    n_tail = len(shape) - axis_idx - 1
    expand = (slice(None),) + (None,) * axis_idx + (slice(None),) + (None,) * n_tail
    tile = (1,) + shape[:axis_idx] + (1,) + shape[axis_idx + 1 :]
    return np.tile(zi[expand], tile)


def _zi_ba(b, a, shape, axis_idx):
    zi = scipy.signal.lfilter_zi(b, a)
    n_tail = len(shape) - axis_idx - 1
    expand = (None,) * axis_idx + (slice(None),) + (None,) * n_tail
    tile = shape[:axis_idx] + (1,) + shape[axis_idx + 1 :]
    return np.tile(zi[expand], tile)


# ---------------------------------------------------------------------------
# Bit-identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "shape, axis_idx",
    [
        ((4_000, 256), 0),  # time-first
        ((256, 4_000), 1),  # time-last
        ((8, 4_000, 32), 1),  # 3-D, time in the middle
        ((4_000, 257), 0),  # channel count not divisible by worker count
    ],
)
def test_sos_threaded_is_bit_identical(shape, axis_idx, dtype):
    sos = _sos()
    x = np.random.default_rng(0).standard_normal(shape).astype(dtype)
    zi = _zi_sos(sos, shape, axis_idx).astype(dtype)

    expected, expected_zf = scipy.signal.sosfilt(sos, x, axis=axis_idx, zi=zi)
    actual, actual_zf = threaded_filt.filt_threaded(
        scipy.signal.sosfilt, (sos,), x, axis_idx, zi, zi_axis_offset=1, min_bytes=1
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_zf, expected_zf)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ba_threaded_is_bit_identical(dtype):
    b, a = _ba()
    shape, axis_idx = (256, 4_000), 1
    x = np.random.default_rng(1).standard_normal(shape).astype(dtype)
    zi = _zi_ba(b, a, shape, axis_idx).astype(dtype)

    expected, expected_zf = scipy.signal.lfilter(b, a, x, axis=axis_idx, zi=zi)
    actual, actual_zf = threaded_filt.filt_threaded(
        scipy.signal.lfilter, (b, a), x, axis_idx, zi, zi_axis_offset=0, min_bytes=1
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_zf, expected_zf)


def test_threaded_streaming_matches_one_shot():
    """State carried across chunks must survive the split/reassemble round-trip."""
    sos = _sos()
    n_ch, total, chunk = 256, 8_192, 1_024
    x = np.random.default_rng(2).standard_normal((n_ch, total))
    zi0 = _zi_sos(sos, (n_ch, chunk), 1)

    expected, _ = scipy.signal.sosfilt(sos, x, axis=-1, zi=zi0)

    zi = zi0
    outs = []
    for s in range(0, total, chunk):
        y, zi = threaded_filt.filt_threaded(
            scipy.signal.sosfilt, (sos,), x[:, s : s + chunk], 1, zi, zi_axis_offset=1, min_bytes=1
        )
        outs.append(y)

    np.testing.assert_array_equal(np.concatenate(outs, axis=-1), expected)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def test_should_thread_gates_on_size():
    x_small = np.zeros((256, 30))  # 60 KB -- threading measured a 4x loss here
    x_large = np.zeros((256, 4_000))  # 8 MB
    assert not threaded_filt.should_thread(x_small, 1)
    assert threaded_filt.should_thread(x_large, 1)


def test_should_thread_disabled_by_zero():
    x = np.zeros((256, 40_000))
    assert threaded_filt.should_thread(x, 1)
    assert not threaded_filt.should_thread(x, 1, min_bytes=0)


def test_should_thread_declines_when_nothing_to_split():
    """1-D input, and input whose only other axis is degenerate."""
    assert not threaded_filt.should_thread(np.zeros(400_000), 0)
    assert not threaded_filt.should_thread(np.zeros((1, 400_000)), 1)


def test_should_thread_declines_with_one_worker(monkeypatch):
    monkeypatch.setattr(threaded_filt, "_worker_count", lambda: 1)
    assert not threaded_filt.should_thread(np.zeros((256, 4_000)), 1)


def test_should_thread_declines_non_numpy():
    class NotAnArray:
        nbytes = 1 << 30
        shape = (256, 40_000)

    assert not threaded_filt.should_thread(NotAnArray(), 1)


def test_below_threshold_takes_the_single_threaded_path(monkeypatch):
    """The gate must avoid the pool entirely, not just avoid the speedup."""
    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("pool must not be touched below the threshold")

    monkeypatch.setattr(threaded_filt, "get_pool", boom)

    sos = _sos()
    x = np.random.default_rng(3).standard_normal((256, 30))
    zi = _zi_sos(sos, x.shape, 1)
    expected, expected_zf = scipy.signal.sosfilt(sos, x, axis=1, zi=zi)
    actual, actual_zf = threaded_filt.filt_threaded(scipy.signal.sosfilt, (sos,), x, 1, zi, zi_axis_offset=1)

    assert not called
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_zf, expected_zf)


# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------


def test_pool_is_shared_and_restartable():
    a = threaded_filt.get_pool()
    b = threaded_filt.get_pool()
    assert a is b, "a new pool per call would defeat the point of pooling"
    threaded_filt.shutdown_pool()
    c = threaded_filt.get_pool()
    assert c is not a, "pool must be recreated after shutdown"
    assert c.submit(lambda: 42).result() == 42


def test_shutdown_is_idempotent():
    threaded_filt.get_pool()
    threaded_filt.shutdown_pool()
    threaded_filt.shutdown_pool()  # must not raise
    assert threaded_filt.get_pool().submit(lambda: 1).result() == 1


# ---------------------------------------------------------------------------
# End-to-end through FilterTransformer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coef_type", ["ba", "sos"])
@pytest.mark.parametrize("dims, axis", [(["time", "ch"], "time"), (["ch", "time"], "time")])
def test_transformer_threaded_matches_unthreaded(coef_type, dims, axis):
    n_ch, n_time = 256, 4_000
    coefs = _sos() if coef_type == "sos" else _ba()
    data = np.random.default_rng(4).standard_normal((n_time, n_ch))
    if dims[0] == "ch":
        data = np.ascontiguousarray(data.T)

    def run(thread_min_bytes):
        tf = FilterTransformer(
            FilterSettings(axis=axis, coefs=coefs, coef_type=coef_type, thread_min_bytes=thread_min_bytes)
        )
        msg = AxisArray(data, dims=dims, axes={"time": AxisArray.TimeAxis(fs=FS)}, key="t")
        return tf(msg).data

    np.testing.assert_array_equal(run(1), run(0))


def test_transformer_default_leaves_online_chunks_single_threaded(monkeypatch):
    """A realistic online chunk must not reach the pool under default settings."""
    monkeypatch.setattr(
        threaded_filt, "get_pool", lambda: (_ for _ in ()).throw(AssertionError("pool used for an online chunk"))
    )
    tf = FilterTransformer(FilterSettings(axis="time", coefs=_sos(), coef_type="sos"))
    data = np.random.default_rng(5).standard_normal((30, 256))  # 30 samples x 256 ch = 60 KB
    msg = AxisArray(data, dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=30_000.0)}, key="t")
    out = tf(msg)
    assert out.data.shape == data.shape
