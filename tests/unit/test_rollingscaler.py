import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.rollingscaler import (
    RollingScalerProcessor,
    RollingScalerSettings,
)
from ezmsg.sigproc.sampler import SampleMessage


def _axisarray_from_ndarray(x: np.ndarray, fs: float = 100.0, t0: float = 0.0) -> AxisArray:
    """
    x shape: (time, ch)
    """
    T, C = x.shape
    return AxisArray(
        data=x,
        dims=["time", "ch"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=fs, offset=t0),
                "ch": AxisArray.CoordinateAxis(data=np.arange(C), dims=["ch"]),
            }
        ),
    )


def test_warmup_identity():
    proc = RollingScalerProcessor(settings=RollingScalerSettings())
    x = np.random.randn(32, 4)
    msg = _axisarray_from_ndarray(x)
    proc._reset_state(msg)
    out = proc._process(msg)
    np.testing.assert_allclose(out.data, x, atol=1e-12)


def test_partial_fit_updates_and_process_uses_snapshot():
    proc = RollingScalerProcessor(settings=RollingScalerSettings())
    init = _axisarray_from_ndarray(np.zeros((1, 2)))
    proc._reset_state(init)

    epoch = np.array(
        [
            [1.0, 3.0],
            [3.0, 3.0],
            [5.0, 3.0],
        ]
    )
    proc.partial_fit(SampleMessage(trigger=None, sample=_axisarray_from_ndarray(epoch)))

    x = np.array([[1.0, 3.0], [3.0, 3.0], [5.0, 3.0]])
    msg = _axisarray_from_ndarray(x)
    out = proc._process(msg)

    mu = epoch.mean(axis=0)
    ex2 = (epoch**2).mean(axis=0)
    var = np.maximum(ex2 - mu**2, 0.0)
    std = np.sqrt(np.maximum(var, 1e-12))
    expected = (x - mu) / std
    np.testing.assert_allclose(out.data, expected, rtol=1e-6, atol=1e-6)


def test_rolling_window_drops_oldest():
    settings = RollingScalerSettings(k_samples=3)
    proc = RollingScalerProcessor(settings=settings)
    proc._reset_state(_axisarray_from_ndarray(np.zeros((1, 1))))

    def add_epoch(vals):
        arr = np.array(vals, dtype=float).reshape(-1, 1)
        proc.partial_fit(SampleMessage(trigger=None, sample=_axisarray_from_ndarray(arr)))

    add_epoch([1, 1, 1, 1])
    add_epoch([3, 3])
    add_epoch([5, 5, 5])
    mu_before = proc.state.mean.copy()

    add_epoch([7, 7, 7, 7])
    mu_after = proc.state.mean.copy()
    np.testing.assert_allclose(mu_before[0], 25 / 9, atol=1e-12)
    np.testing.assert_allclose(mu_after[0], 49 / 9, atol=1e-12)


def test_window_size_converts_to_k_samples_and_min_seconds():
    fs = 200.0
    msg0 = _axisarray_from_ndarray(np.zeros((1, 2)), fs=fs)

    settings = RollingScalerSettings(
        window_size=0.101,
        min_seconds=0.055,
    )
    proc = RollingScalerProcessor(settings=settings)
    proc(msg0)

    assert proc.state.k_samples == 21
    assert proc.state.min_samples == 11


def test_under_threshold_passthrough_without_update():
    fs = 100.0
    settings = RollingScalerSettings(window_size=1.0, min_seconds=1.0, update_with_signal=False)
    proc = RollingScalerProcessor(settings=settings)

    x = np.ones((50, 1))
    msg = _axisarray_from_ndarray(x, fs=fs)
    out = proc(msg)
    np.testing.assert_allclose(out.data, x, atol=1e-12)


def test_update_with_signal_influences_next_chunk():
    fs = 100.0
    settings = RollingScalerSettings(window_size=1.0, min_seconds=0.0, update_with_signal=True)
    proc = RollingScalerProcessor(settings=settings)

    x1 = np.ones((60, 1))
    msg1 = _axisarray_from_ndarray(x1, fs=fs)
    out1 = proc(msg1)
    np.testing.assert_allclose(out1.data, x1, atol=1e-12)
    assert proc.state.N == 60

    x2 = np.ones((40, 1))
    msg2 = _axisarray_from_ndarray(x2, fs=fs)
    out2 = proc(msg2)
    np.testing.assert_allclose(out2.data, 0.0, atol=1e-9)


def test_artifact_rejection_excludes_rows_from_update():
    fs = 100.0
    settings = RollingScalerSettings(
        window_size=1.0,
        min_seconds=0.0,
        update_with_signal=True,
        artifact_z_thresh=5.0,
    )
    proc = RollingScalerProcessor(settings=settings)

    x1 = np.zeros((100, 1))
    msg1 = _axisarray_from_ndarray(x1, fs=fs)
    _ = proc(msg1)
    assert proc.state.N == 100

    x2 = np.vstack([np.zeros((50, 1)), 10 * np.ones((50, 1))])
    msg2 = _axisarray_from_ndarray(x2, fs=fs)
    _ = proc(msg2)

    assert proc.state.N == 150
    np.testing.assert_allclose(proc.state.mean, 0.0, atol=1e-12)
