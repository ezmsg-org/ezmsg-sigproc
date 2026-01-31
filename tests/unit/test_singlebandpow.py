import numpy as np
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
