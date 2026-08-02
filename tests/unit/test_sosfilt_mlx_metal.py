import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.butterworthfilter import ButterworthFilterSettings, ButterworthFilterTransformer
from tests.helpers.util import requires_mlx


def test_sos_component_default_chunk_sizes():
    assert ButterworthFilterSettings().mlx_metal_chunk_sizes == (512,)


@requires_mlx
def test_sosfilt_mlx_metal_low_highpass_matches_scipy_float32():
    import mlx.core as mx

    from ezmsg.sigproc.util.sosfilt_mlx_metal import sosfilt_mlx_metal

    fs = 30_000.0
    sos = scipy.signal.butter(4, 3.0, btype="highpass", fs=fs, output="sos").astype(np.float32)
    rng = np.random.default_rng(0)
    # One full MAX_CHUNK_SIZE span plus a one-sample tail exercises runtime
    # valid-length state emission in the numerically delicate serial kernel.
    data = rng.standard_normal((3, 4097)).astype(np.float32)

    zi = np.zeros((sos.shape[0], data.shape[0], 2), dtype=np.float32)
    expected, expected_zf = scipy.signal.sosfilt(sos, data, axis=-1, zi=zi)

    actual, actual_zf = sosfilt_mlx_metal(mx.array(sos), mx.array(data))
    mx.eval(actual, actual_zf)

    actual_np = np.asarray(actual)
    assert np.isfinite(actual_np).all()
    assert np.allclose(actual_np, expected, rtol=1e-5, atol=2e-5)
    assert np.allclose(np.asarray(actual_zf), expected_zf, rtol=1e-5, atol=2e-5)


@requires_mlx
def test_sosfilt_mlx_metal_variable_tails_preserve_streaming_state():
    import mlx.core as mx

    from ezmsg.sigproc.util.sosfilt_mlx_metal import _sosfilt_mlx_metal_unfused, sosfilt_mlx_metal

    fs = 1_000.0
    chunk_size = 128
    sos = scipy.signal.butter(4, [30.0, 100.0], btype="bandpass", fs=fs, output="sos").astype(np.float32)
    rng = np.random.default_rng(187)
    blocks = [
        rng.standard_normal((3, 257)).astype(np.float32),  # tail 1
        rng.standard_normal((3, 259)).astype(np.float32),  # tail 3
    ]

    expected_state = np.zeros((sos.shape[0], 3, 2), dtype=np.float32)
    actual_state = None
    for block in blocks:
        expected, expected_state = scipy.signal.sosfilt(sos, block, axis=-1, zi=expected_state)
        actual, actual_state = sosfilt_mlx_metal(
            mx.array(sos),
            mx.array(block),
            zi=actual_state,
            chunk_size=chunk_size,
        )
        mx.eval(actual, actual_state)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=2e-5)
        np.testing.assert_allclose(np.asarray(actual_state), expected_state, rtol=1e-5, atol=2e-5)

    unfused, unfused_state = _sosfilt_mlx_metal_unfused(
        mx.array(sos),
        mx.array(blocks[-1]),
        chunk_size=chunk_size,
    )
    fused, fused_state = sosfilt_mlx_metal(mx.array(sos), mx.array(blocks[-1]), chunk_size=chunk_size)
    mx.eval(unfused, unfused_state, fused, fused_state)
    np.testing.assert_array_equal(np.asarray(fused), np.asarray(unfused))
    np.testing.assert_array_equal(np.asarray(fused_state), np.asarray(unfused_state))


@requires_mlx
def test_sosfilt_mlx_metal_dispatches_only_configured_chunk_sizes(monkeypatch):
    import mlx.core as mx

    import ezmsg.sigproc.util.sosfilt_mlx_metal as sosfilt_module

    sos = scipy.signal.butter(4, [30.0, 100.0], btype="bandpass", fs=1_000.0, output="sos").astype(np.float32)
    rng = np.random.default_rng(188)
    data = rng.standard_normal((3, 1_054)).astype(np.float32)
    expected = scipy.signal.sosfilt(sos, data, axis=-1)

    launches = []
    original_launch = sosfilt_module._launch_fused_kernel

    def recording_launch(x_chunk, sos_flat, state, n_channels, n_sections, cs, valid_length):
        launches.append((cs, int(valid_length.item())))
        return original_launch(x_chunk, sos_flat, state, n_channels, n_sections, cs, valid_length)

    monkeypatch.setattr(sosfilt_module, "_launch_fused_kernel", recording_launch)
    actual, actual_state = sosfilt_module.sosfilt_mlx_metal(
        mx.array(sos),
        mx.array(data),
        chunk_sizes=(512, 32, 32),
    )
    mx.eval(actual, actual_state)

    assert launches == [(512, 512), (512, 512), (32, 30)]
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=2e-5)


@requires_mlx
@pytest.mark.parametrize("chunk_sizes", [(), (0,), (513,), (32, 1.5)])
def test_sosfilt_mlx_metal_rejects_invalid_chunk_sizes(chunk_sizes):
    import mlx.core as mx

    from ezmsg.sigproc.util.sosfilt_mlx_metal import sosfilt_mlx_metal

    sos = mx.array(scipy.signal.butter(2, 100.0, fs=1_000.0, output="sos").astype(np.float32))
    with pytest.raises(ValueError, match="chunk_sizes"):
        sosfilt_mlx_metal(sos, mx.zeros((2, 30)), chunk_sizes=chunk_sizes)


@requires_mlx
def test_butterworth_mlx_dispatches_component_chunk_sizes(monkeypatch):
    import mlx.core as mx

    import ezmsg.sigproc.util.sosfilt_mlx_metal as sosfilt_module

    fs = 1_000.0
    rng = np.random.default_rng(189)
    data = rng.standard_normal((300, 4)).astype(np.float32)
    settings = ButterworthFilterSettings(
        axis="time",
        order=4,
        cuton=30.0,
        cutoff=100.0,
        coef_type="sos",
        mlx_metal_chunk_sizes=(32, 128),
    )
    msg_np = AxisArray(data, dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=fs)})
    msg_mx = AxisArray(mx.array(data), dims=msg_np.dims, axes=msg_np.axes)
    proc_np = ButterworthFilterTransformer(settings)
    proc_mx = ButterworthFilterTransformer(settings)

    launches = []
    original_launch = sosfilt_module._launch_fused_kernel

    def recording_launch(x_chunk, sos_flat, state, n_channels, n_sections, cs, valid_length):
        launches.append((cs, int(valid_length.item())))
        return original_launch(x_chunk, sos_flat, state, n_channels, n_sections, cs, valid_length)

    monkeypatch.setattr(sosfilt_module, "_launch_fused_kernel", recording_launch)
    expected = proc_np(msg_np)
    actual = proc_mx(msg_mx)
    mx.eval(actual.data)

    assert proc_mx.state.filter.settings.mlx_metal_chunk_sizes == (32, 128)
    assert launches == [(128, 128), (128, 128), (128, 44)]
    np.testing.assert_allclose(np.asarray(actual.data), expected.data, rtol=1e-5, atol=2e-5)


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
    # FilterTransformer edge-scales the steady-state zi by the first sample.
    zi = zi * data[0]
    expected, _ = scipy.signal.sosfilt(sos, data, axis=0, zi=zi)

    actual = np.asarray(result.data)
    assert np.isfinite(actual).all()
    assert np.allclose(actual, expected.astype(np.float32), rtol=1e-5, atol=1e-5)
