import platform
import time

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.butterworthfilter import ButterworthFilterSettings
from ezmsg.sigproc.downsample import DownsampleSettings
from ezmsg.sigproc.singlebandpow import (
    RMSBandPowerSettings,
    RMSBandPowerTransformer,
    SquareLawBandPowerSettings,
    SquareLawBandPowerTransformer,
)


def _make_sinusoid(
    freq: float = 50.0,
    amplitude: float = 1.0,
    fs: float = 1000.0,
    duration: float = 2.0,
    n_channels: int = 2,
) -> AxisArray:
    """Generate a multi-channel sinusoid as an AxisArray."""
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    data = np.column_stack([signal] * n_channels)
    return AxisArray(
        data,
        dims=["time", "ch"],
        axes={"time": AxisArray.LinearAxis(gain=1.0 / fs, offset=0.0)},
    )


def test_rms_bandpower():
    """RMS band power of a sinusoid should approximate A / sqrt(2)."""
    freq = 50.0
    amplitude = 2.0
    fs = 1000.0
    duration = 2.0
    bin_duration = 0.1
    n_channels = 2

    msg_in = _make_sinusoid(freq=freq, amplitude=amplitude, fs=fs, duration=duration, n_channels=n_channels)

    xformer = RMSBandPowerTransformer(
        RMSBandPowerSettings(
            bandpass=ButterworthFilterSettings(order=4, coef_type="sos", cuton=30.0, cutoff=70.0),
            bin_duration=bin_duration,
            apply_sqrt=True,
        )
    )

    # Process in chunks to exercise stateful behavior
    chunk_size = 100
    outputs = []
    for i in range(0, msg_in.data.shape[0], chunk_size):
        chunk_data = msg_in.data[i : i + chunk_size]
        chunk = AxisArray(
            chunk_data,
            dims=["time", "ch"],
            axes={"time": AxisArray.LinearAxis(gain=1.0 / fs, offset=i / fs)},
        )
        result = xformer(chunk)
        if result.data.size > 0:
            outputs.append(result)

    assert len(outputs) > 0

    all_data = np.concatenate([o.data for o in outputs], axis=0)

    # Output should have dims (time, ch)
    assert all_data.ndim == 2
    assert all_data.shape[1] == n_channels

    # Check the output axis is "time" (renamed from "bin")
    assert "time" in outputs[-1].dims

    # After the filter settles, RMS of sinusoid should be ~ A / sqrt(2)
    expected_rms = amplitude / np.sqrt(2)
    # Use the second half of the output to let the filter settle
    settled = all_data[all_data.shape[0] // 2 :]
    mean_rms = np.mean(settled)
    assert abs(mean_rms - expected_rms) < 0.15 * expected_rms, f"Expected RMS ~{expected_rms:.3f}, got {mean_rms:.3f}"


def test_rms_bandpower_no_sqrt():
    """With apply_sqrt=False, output should be mean-square power ~ A^2 / 2."""
    freq = 50.0
    amplitude = 2.0
    fs = 1000.0
    duration = 2.0
    bin_duration = 0.1

    msg_in = _make_sinusoid(freq=freq, amplitude=amplitude, fs=fs, duration=duration, n_channels=1)

    xformer = RMSBandPowerTransformer(
        RMSBandPowerSettings(
            bandpass=ButterworthFilterSettings(order=4, coef_type="sos", cuton=30.0, cutoff=70.0),
            bin_duration=bin_duration,
            apply_sqrt=False,
        )
    )

    chunk_size = 100
    outputs = []
    for i in range(0, msg_in.data.shape[0], chunk_size):
        chunk_data = msg_in.data[i : i + chunk_size]
        chunk = AxisArray(
            chunk_data,
            dims=["time", "ch"],
            axes={"time": AxisArray.LinearAxis(gain=1.0 / fs, offset=i / fs)},
        )
        result = xformer(chunk)
        if result.data.size > 0:
            outputs.append(result)

    assert len(outputs) > 0

    all_data = np.concatenate([o.data for o in outputs], axis=0)

    # Mean-square power of sinusoid: A^2 / 2
    expected_ms = amplitude**2 / 2
    settled = all_data[all_data.shape[0] // 2 :]
    mean_ms = np.mean(settled)
    assert (
        abs(mean_ms - expected_ms) < 0.15 * expected_ms
    ), f"Expected mean-square ~{expected_ms:.3f}, got {mean_ms:.3f}"


def test_squarelaw_bandpower():
    """Square-law band power should track signal power and downsample correctly."""
    freq = 50.0
    amplitude = 3.0
    fs = 1000.0
    duration = 2.0
    target_rate = 100.0
    n_channels = 2

    msg_in = _make_sinusoid(freq=freq, amplitude=amplitude, fs=fs, duration=duration, n_channels=n_channels)

    xformer = SquareLawBandPowerTransformer(
        SquareLawBandPowerSettings(
            bandpass=ButterworthFilterSettings(order=4, coef_type="sos", cuton=30.0, cutoff=70.0),
            lowpass=ButterworthFilterSettings(order=4, coef_type="sos", cutoff=10.0),
            downsample=DownsampleSettings(target_rate=target_rate),
        )
    )

    chunk_size = 100
    outputs = []
    for i in range(0, msg_in.data.shape[0], chunk_size):
        chunk_data = msg_in.data[i : i + chunk_size]
        chunk = AxisArray(
            chunk_data,
            dims=["time", "ch"],
            axes={"time": AxisArray.LinearAxis(gain=1.0 / fs, offset=i / fs)},
        )
        result = xformer(chunk)
        if result.data.size > 0:
            outputs.append(result)

    assert len(outputs) > 0

    all_data = np.concatenate([o.data for o in outputs], axis=0)

    # Output should have dims (time, ch) and be downsampled
    assert all_data.ndim == 2
    assert all_data.shape[1] == n_channels

    # Check output rate: should be approximately target_rate
    out_axis = outputs[-1].get_axis("time")
    out_rate = 1.0 / out_axis.gain
    assert abs(out_rate - target_rate) < 1.0, f"Expected rate ~{target_rate}, got {out_rate}"

    # After settling, the mean power should track A^2/2
    expected_ms = amplitude**2 / 2
    settled = all_data[all_data.shape[0] // 2 :]
    mean_power = np.mean(settled)
    assert (
        abs(mean_power - expected_ms) < 0.25 * expected_ms
    ), f"Expected power ~{expected_ms:.3f}, got {mean_power:.3f}"


requires_apple_silicon = pytest.mark.skipif(
    platform.machine() != "arm64" or platform.system() != "Darwin",
    reason="Requires Apple Silicon for MLX",
)


@requires_apple_silicon
def test_rms_bandpower_mlx_benchmark():
    """
    Benchmark RMSBandPowerTransformer with numpy vs MLX backends.

    Both paths receive numpy input. The numpy-backend transformer keeps data as
    numpy throughout, while the mlx-backend transformer converts to MLX after
    the bandpass filter (via the asarray step), running the remaining pipeline
    stages (square, window, aggregate, sqrt) on MLX arrays.
    """
    import mlx.core as mx

    from ezmsg.sigproc.asarray import ArrayBackend

    freq, amplitude, fs = 50.0, 2.0, 30_000.0
    chunk_samples = 100
    n_channels = 684
    n_chunks = 500

    bandpass = ButterworthFilterSettings(order=4, coef_type="sos", cuton=30.0, cutoff=70.0)

    settings_np = RMSBandPowerSettings(
        bandpass=bandpass,
        backend=ArrayBackend.numpy,
        bin_duration=0.05,
        apply_sqrt=True,
    )
    settings_mx = RMSBandPowerSettings(
        bandpass=bandpass,
        backend=ArrayBackend.mlx,
        bin_duration=0.05,
        apply_sqrt=True,
    )

    # Pre-generate all chunk messages as numpy
    np_chunks = []
    for i in range(n_chunks + 1):  # +1 for warmup
        t = (np.arange(chunk_samples) + i * chunk_samples) / fs
        data = amplitude * np.sin(2 * np.pi * freq * t[:, None] * np.ones((1, n_channels)))
        np_chunks.append(
            AxisArray(
                data,
                dims=["time", "ch"],
                axes={"time": AxisArray.LinearAxis(gain=1.0 / fs, offset=i * chunk_samples / fs)},
            )
        )

    # --- Numpy backend ---
    xformer_np = RMSBandPowerTransformer(settings_np)
    xformer_np(np_chunks[0])  # Warmup

    t0 = time.perf_counter()
    np_outputs = [xformer_np(chunk) for chunk in np_chunks[1:]]
    t_numpy = time.perf_counter() - t0

    # --- MLX backend ---
    xformer_mx = RMSBandPowerTransformer(settings_mx)
    xformer_mx(np_chunks[0])  # Warmup

    t0 = time.perf_counter()
    mx_outputs = [xformer_mx(chunk) for chunk in np_chunks[1:]]
    t_mlx = time.perf_counter() - t0

    # Correctness: both backends should produce equivalent results.
    for np_out, mx_out in zip(np_outputs, mx_outputs):
        if np_out.data.size > 0 and np.asarray(mx_out.data).size > 0:
            np.testing.assert_allclose(np.asarray(mx_out.data), np_out.data, rtol=1e-5)

    # Verify output array types match the configured backend.
    last_np = next(o for o in reversed(np_outputs) if o.data.size > 0)
    last_mx = next(o for o in reversed(mx_outputs) if np.asarray(o.data).size > 0)
    assert isinstance(last_np.data, np.ndarray)
    assert isinstance(
        last_mx.data, mx.array
    ), f"Expected mlx.core.array output, got {type(last_mx.data).__module__}.{type(last_mx.data).__name__}"

    print(
        f"\n  RMSBandPower benchmark ({n_chunks} chunks, {chunk_samples}×{n_channels}):"
        f"\n    numpy backend: {t_numpy:.4f}s ({t_numpy / n_chunks * 1000:.2f} ms/chunk)"
        f"\n    mlx   backend: {t_mlx:.4f}s ({t_mlx / n_chunks * 1000:.2f} ms/chunk)"
        f"\n    ratio (mlx/numpy): {t_mlx / t_numpy:.2f}x"
    )
