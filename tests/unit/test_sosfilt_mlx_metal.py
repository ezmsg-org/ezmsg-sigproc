import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.butterworthfilter import ButterworthFilterSettings, ButterworthFilterTransformer
from tests.helpers.util import requires_mlx


@requires_mlx
def test_sosfilt_mlx_metal_low_highpass_matches_scipy_float32():
    import mlx.core as mx

    from ezmsg.sigproc.util.sosfilt_mlx_metal import sosfilt_mlx_metal

    fs = 30_000.0
    sos = scipy.signal.butter(4, 3.0, btype="highpass", fs=fs, output="sos").astype(np.float32)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 4096)).astype(np.float32)

    zi = np.zeros((sos.shape[0], data.shape[0], 2), dtype=np.float32)
    expected, expected_zf = scipy.signal.sosfilt(sos, data, axis=-1, zi=zi)

    actual, actual_zf = sosfilt_mlx_metal(mx.array(sos), mx.array(data))
    mx.eval(actual, actual_zf)

    actual_np = np.asarray(actual)
    assert np.isfinite(actual_np).all()
    assert np.allclose(actual_np, expected, rtol=1e-5, atol=2e-5)
    assert np.allclose(np.asarray(actual_zf), expected_zf, rtol=1e-5, atol=2e-5)


@requires_mlx
def test_sosfilt_mlx_metal_rejects_float32_unstable_highpass():
    import mlx.core as mx

    from ezmsg.sigproc.util.sosfilt_mlx_metal import sosfilt_mlx_metal

    fs = 30_000.0
    sos = scipy.signal.butter(4, 0.3, btype="highpass", fs=fs, output="sos").astype(np.float32)
    data = np.zeros((1, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="float32 quantization"):
        sosfilt_mlx_metal(mx.array(sos), mx.array(data))


@requires_mlx
def test_butterworth_mlx_float32_unstable_highpass_uses_scipy_numpy_state():
    import mlx.core as mx

    fs = 30_000.0
    n_samples = 4096
    n_channels = 2
    rng = np.random.default_rng(1)
    data = rng.standard_normal((n_samples, n_channels)).astype(np.float32)
    msg = AxisArray(
        mx.array(data),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0.0)},
        key="low-highpass",
    )

    transformer = ButterworthFilterTransformer(
        ButterworthFilterSettings(
            axis="time",
            order=4,
            cuton=0.3,
            cutoff=None,
            coef_type="sos",
            use_mlx_metal=True,
        )
    )

    result = transformer(msg)
    mx.eval(result.data)

    assert transformer.state.filter.state.sos_method == "scipy_numpy"
    assert isinstance(transformer.state.filter.state.zi, np.ndarray)

    sos = scipy.signal.butter(4, 0.3, btype="highpass", fs=fs, output="sos")
    zi = scipy.signal.sosfilt_zi(sos)[:, :, None] + np.zeros((1, 1, n_channels))
    expected, _ = scipy.signal.sosfilt(sos, data, axis=0, zi=zi)

    actual = np.asarray(result.data)
    assert np.isfinite(actual).all()
    assert np.allclose(actual, expected.astype(np.float32), rtol=1e-5, atol=1e-5)
