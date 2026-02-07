import platform
import time

import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.firfilter import (
    FIRFilterSettings,
    FIRFilterTransformer,
    firwin_design_fun,
)
from tests.helpers.empty_time import check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg

requires_apple_silicon = pytest.mark.skipif(
    platform.machine() != "arm64" or platform.system() != "Darwin",
    reason="Requires Apple Silicon for MLX",
)


@pytest.mark.parametrize(
    "cutoff, pass_zero",
    [
        (30.0, True),  # lowpass
        (30.0, False),  # highpass
        ([30.0, 45.0], True),  # bandpass
        ([30.0, 45.0], False),  # bandstop
        ([30.0, 45.0], "bandpass"),  # explicit bandpass
        ([30.0, 45.0], "bandstop"),  # explicit bandstop
    ],
)
@pytest.mark.parametrize("order", [11, 21, 51])  # Odd numbers for FIR
@pytest.mark.parametrize("window", ["hamming", "hann", "blackman"])
def test_firwin_design_fun(cutoff, pass_zero, order, window):
    """Test the FIR filter design function with various parameters."""
    fs = 200.0
    result = firwin_design_fun(
        fs=fs,
        order=order,
        cutoff=cutoff,
        window=window,
        pass_zero=pass_zero,
        wn_hz=True,
    )

    assert result is not None
    b, a = result
    assert len(b) == order
    assert len(a) == 1
    assert a[0] == 1.0
    # FIR filters are always stable (no feedback, only zeros)


def test_firwin_design_fun_zero_order():
    """Test that zero order returns None (no filter)."""
    result = firwin_design_fun(
        fs=200.0,
        order=0,
        cutoff=30.0,
        window="hamming",
        pass_zero=True,
        wn_hz=True,
    )
    assert result is None


@pytest.mark.parametrize(
    "cutoff, pass_zero",
    [
        (30.0, True),  # lowpass
        (30.0, False),  # highpass
        ([30.0, 45.0], "bandpass"),  # bandpass
        ([30.0, 45.0], "bandstop"),  # bandstop
    ],
)
@pytest.mark.parametrize("order", [0, 11, 21])  # Include 0 for passthrough
@pytest.mark.parametrize("fs", [200.0])
@pytest.mark.parametrize("n_chans", [3])
@pytest.mark.parametrize("n_dims, time_ax", [(1, 0), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_firfilter_transformer(
    cutoff,
    pass_zero,
    order,
    fs,
    n_chans,
    n_dims,
    time_ax,
    coef_type,
):
    """Test FIR filter transformer with various configurations."""
    dur = 2.0
    n_freqs = 5
    n_splits = 4

    n_times = int(dur * fs)
    if n_dims == 1:
        dat_shape = [n_times]
        dat_dims = ["time"]
        other_axes = {}
    else:
        dat_shape = [n_freqs, n_chans]
        dat_shape.insert(time_ax, n_times)
        dat_dims = ["freq", "ch"]
        dat_dims.insert(time_ax, "time")
        other_axes = {
            "freq": AxisArray.LinearAxis(unit="Hz", offset=0.0, gain=1.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(n_chans).astype(str), dims=["ch"]),
        }
    in_dat = np.arange(np.prod(dat_shape), dtype=float).reshape(*dat_shape)

    # Calculate Expected Result
    if order > 0:
        coefs = firwin_design_fun(
            fs=fs,
            order=order,
            cutoff=cutoff,
            window="hamming",
            pass_zero=pass_zero,
            scale=True,
            wn_hz=True,
        )
        if coef_type == "sos":
            # Convert ba to sos for comparison
            b, a = coefs
            coefs = scipy.signal.tf2sos(b, a)
        tmp_dat = np.moveaxis(in_dat, time_ax, -1)

        if coef_type == "ba":
            b, a = coefs
            # FIR filters use zero initial conditions (lfiltic with empty arrays)
            zi = scipy.signal.lfiltic(b, a, [])
            if n_dims == 3:
                zi = np.tile(zi[None, None, :], (n_freqs, n_chans, 1))
            out_dat, _ = scipy.signal.lfilter(b, a, tmp_dat, zi=zi)
        elif coef_type == "sos":
            # SOS representation uses sosfilt_zi for initial conditions
            zi = scipy.signal.sosfilt_zi(coefs)
            if n_dims == 3:
                zi = np.tile(zi[:, None, None, :], (1, n_freqs, n_chans, 1))
            out_dat, _ = scipy.signal.sosfilt(coefs, tmp_dat, zi=zi)
        expected = np.moveaxis(out_dat, -1, time_ax)
    else:
        # Zero order = passthrough
        expected = in_dat

    # Split the data into multiple messages
    n_seen = 0
    messages = []
    for split_dat in np.array_split(in_dat, n_splits, axis=time_ax):
        _time_axis = AxisArray.TimeAxis(fs=fs, offset=n_seen / fs)
        messages.append(
            AxisArray(
                split_dat,
                dims=dat_dims,
                axes=frozendict({**other_axes, "time": _time_axis}),
                key="test_firfilter",
            )
        )
        n_seen += split_dat.shape[time_ax]

    # Create transformer
    axis_name = "time" if time_ax != 0 else None
    settings = FIRFilterSettings(
        axis=axis_name,
        order=order,
        cutoff=cutoff,
        window="hamming",
        pass_zero=pass_zero,
        scale=True,
        wn_hz=True,
        coef_type=coef_type,
    )
    transformer = FIRFilterTransformer(settings=settings)

    # Process messages
    result = []
    for msg in messages:
        out_msg = transformer(msg)
        result.append(out_msg.data)
    result = np.concatenate(result, axis=time_ax)

    assert np.allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_firfilter_empty_msg():
    """Test FIR filter with empty message."""
    settings = FIRFilterSettings(
        axis="time",
        order=21,
        cutoff=30.0,
        window="hamming",
        pass_zero=True,
        coef_type="ba",
    )
    transformer = FIRFilterTransformer(settings=settings)

    msg_in = AxisArray(
        data=np.zeros((0, 2)),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(2).astype(str), dims=["ch"]),
        },
        key="test_firfilter_empty",
    )

    result = transformer(msg_in)
    assert result.data.size == 0


def test_firfilter_normalized_frequencies():
    """Test FIR filter with normalized frequencies (wn_hz=False)."""
    fs = 200.0
    dur = 1.0
    n_times = int(dur * fs)

    # Create input signal
    t = np.arange(n_times) / fs
    # Mix of 10Hz and 60Hz sine waves
    in_dat = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)

    msg = AxisArray(
        data=in_dat,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_normalized",
    )

    # Design lowpass filter at 0.3 (normalized, = 30Hz)
    settings = FIRFilterSettings(
        axis="time",
        order=51,
        cutoff=0.3,  # Normalized cutoff (30Hz / Nyquist(100Hz))
        window="hamming",
        pass_zero=True,
        wn_hz=False,  # Use normalized frequencies
        coef_type="ba",
    )
    transformer = FIRFilterTransformer(settings=settings)

    result = transformer(msg)

    # Verify output shape matches input
    assert result.data.shape == in_dat.shape

    # Check that 10Hz component is preserved and 60Hz is attenuated
    fft_in = np.abs(np.fft.rfft(in_dat))
    fft_out = np.abs(np.fft.rfft(result.data))
    freqs = np.fft.rfftfreq(n_times, 1 / fs)

    idx_10hz = np.argmin(np.abs(freqs - 10))
    idx_60hz = np.argmin(np.abs(freqs - 60))

    # 10Hz should be mostly preserved (> 80% of input)
    assert fft_out[idx_10hz] > 0.8 * fft_in[idx_10hz]
    # 60Hz should be significantly attenuated (< 20% of input)
    assert fft_out[idx_60hz] < 0.2 * fft_in[idx_60hz]


def test_firfilter_kaiser_width():
    """Test FIR filter using Kaiser window specified via width parameter."""
    fs = 200.0
    dur = 1.0
    n_times = int(dur * fs)

    # Create test signal
    t = np.arange(n_times) / fs
    in_dat = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)

    msg = AxisArray(
        data=in_dat,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_kaiser_width",
    )

    # When width is specified, window parameter is ignored and Kaiser is used
    settings = FIRFilterSettings(
        axis="time",
        order=51,
        cutoff=30.0,
        width=10.0,  # Transition width
        window="hamming",  # Will be ignored
        pass_zero=True,
        wn_hz=True,
        coef_type="ba",
    )
    transformer = FIRFilterTransformer(settings=settings)

    result = transformer(msg)
    assert result.data.shape == in_dat.shape


def test_fir_empty_after_init():
    from ezmsg.sigproc.firfilter import FIRFilterSettings, FIRFilterTransformer

    proc = FIRFilterTransformer(FIRFilterSettings(order=10, cutoff=10.0, wn_hz=True, axis="time"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_fir_empty_first():
    from ezmsg.sigproc.firfilter import FIRFilterSettings, FIRFilterTransformer

    proc = FIRFilterTransformer(FIRFilterSettings(order=10, cutoff=10.0, wn_hz=True, axis="time"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


@requires_apple_silicon
@pytest.mark.parametrize(
    "cutoff, pass_zero",
    [
        (30.0, True),  # lowpass
        (30.0, False),  # highpass
        ([30.0, 45.0], "bandpass"),  # bandpass
        ([30.0, 45.0], "bandstop"),  # bandstop
    ],
)
@pytest.mark.parametrize("order", [11, 21])
@pytest.mark.parametrize("n_dims, time_ax", [(1, 0), (3, 0), (3, 1), (3, 2)])
def test_firfilter_mlx(cutoff, pass_zero, order, n_dims, time_ax):
    """Test FIR filter with MLX arrays: correctness vs numpy and output type."""
    import mlx.core as mx

    fs = 200.0
    dur = 2.0
    n_freqs = 5
    n_chans = 3
    n_splits = 4

    n_times = int(dur * fs)
    if n_dims == 1:
        dat_shape = [n_times]
        dat_dims = ["time"]
        other_axes = {}
    else:
        dat_shape = [n_freqs, n_chans]
        dat_shape.insert(time_ax, n_times)
        dat_dims = ["freq", "ch"]
        dat_dims.insert(time_ax, "time")
        other_axes = {
            "freq": AxisArray.LinearAxis(unit="Hz", offset=0.0, gain=1.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(n_chans).astype(str), dims=["ch"]),
        }
    in_dat = np.arange(np.prod(dat_shape), dtype=np.float32).reshape(*dat_shape)

    # --- Numpy reference ---
    np_settings = FIRFilterSettings(
        axis="time" if time_ax != 0 else None,
        order=order,
        cutoff=cutoff,
        window="hamming",
        pass_zero=pass_zero,
        scale=True,
        wn_hz=True,
        coef_type="ba",
    )
    np_xformer = FIRFilterTransformer(settings=np_settings)

    n_seen = 0
    np_results = []
    for split_dat in np.array_split(in_dat, n_splits, axis=time_ax):
        _time_axis = AxisArray.TimeAxis(fs=fs, offset=n_seen / fs)
        msg = AxisArray(
            split_dat,
            dims=dat_dims,
            axes=frozendict({**other_axes, "time": _time_axis}),
            key="test_fir_mlx",
        )
        np_results.append(np_xformer(msg).data)
        n_seen += split_dat.shape[time_ax]
    np_result = np.concatenate(np_results, axis=time_ax)

    # --- MLX path ---
    xp_settings = FIRFilterSettings(
        axis="time" if time_ax != 0 else None,
        order=order,
        cutoff=cutoff,
        window="hamming",
        pass_zero=pass_zero,
        scale=True,
        wn_hz=True,
        coef_type="ba",
    )
    xp_xformer = FIRFilterTransformer(settings=xp_settings)

    n_seen = 0
    mx_results = []
    for split_dat in np.array_split(in_dat, n_splits, axis=time_ax):
        _time_axis = AxisArray.TimeAxis(fs=fs, offset=n_seen / fs)
        mx_dat = mx.array(split_dat)
        msg = AxisArray(
            mx_dat,
            dims=dat_dims,
            axes=frozendict({**other_axes, "time": _time_axis}),
            key="test_fir_mlx",
        )
        out = xp_xformer(msg)
        mx_results.append(out.data)
        n_seen += split_dat.shape[time_ax]

    # Verify output is MLX array
    assert isinstance(mx_results[0], mx.array), "Output should be mx.array, not numpy"

    mx_result = mx.concatenate(mx_results, axis=time_ax)
    mx_result_np = np.array(mx_result)

    # Correctness: FFT convolution should match scipy lfilter
    # Tolerances account for float32 FFT rounding vs scipy's float64 lfilter.
    np.testing.assert_allclose(mx_result_np, np_result, rtol=5e-3, atol=5e-3)


@requires_apple_silicon
def test_firfilter_mlx_benchmark():
    """Benchmark FIR filter: numpy (scipy lfilter) vs MLX (FFT convolution)."""
    import mlx.core as mx

    fs = 1000.0
    chunk_samples = 256
    n_channels = 32
    n_chunks = 100
    order = 51

    np_settings = FIRFilterSettings(
        axis="time",
        order=order,
        cutoff=100.0,
        window="hamming",
        pass_zero=True,
        scale=True,
        wn_hz=True,
        coef_type="ba",
    )

    # Build chunks
    rng = np.random.default_rng(42)
    np_chunks = []
    mx_chunks = []
    for i in range(n_chunks):
        d = rng.standard_normal((chunk_samples, n_channels)).astype(np.float32)
        _time_axis = AxisArray.TimeAxis(fs=fs, offset=i * chunk_samples / fs)
        axes = frozendict(
            {
                "time": _time_axis,
                "ch": AxisArray.CoordinateAxis(data=np.arange(n_channels).astype(str), dims=["ch"]),
            }
        )
        np_chunks.append(AxisArray(d, dims=["time", "ch"], axes=axes, key="bench"))
        mx_chunks.append(AxisArray(mx.array(d), dims=["time", "ch"], axes=axes, key="bench"))

    # Warm up both transformers
    np_xformer = FIRFilterTransformer(settings=np_settings)
    mx_xformer = FIRFilterTransformer(settings=np_settings)
    _ = np_xformer(np_chunks[0])
    _ = mx_xformer(mx_chunks[0])
    mx.eval(mx_xformer(mx_chunks[0]).data)  # force MLX compile

    # Numpy timing
    np_xformer = FIRFilterTransformer(settings=np_settings)
    _ = np_xformer(np_chunks[0])
    t0 = time.perf_counter()
    np_outputs = [np_xformer(chunk) for chunk in np_chunks[1:]]
    t_numpy = time.perf_counter() - t0

    # MLX timing
    mx_xformer = FIRFilterTransformer(settings=np_settings)
    _ = mx_xformer(mx_chunks[0])
    t0 = time.perf_counter()
    mx_outputs = [mx_xformer(chunk) for chunk in mx_chunks[1:]]
    mx.eval(mx_outputs[-1].data)  # force evaluation
    t_mlx = time.perf_counter() - t0

    # Correctness check
    for np_out, mx_out in zip(np_outputs, mx_outputs):
        if np_out.data.size > 0:
            np.testing.assert_allclose(np.array(mx_out.data), np_out.data, rtol=1e-4, atol=1e-4)

    # Verify MLX output type
    assert isinstance(mx_outputs[0].data, mx.array), "MLX output should remain mx.array"

    print(
        f"\n  FIR filter benchmark ({n_chunks} chunks, {chunk_samples}x{n_channels}, order={order}):"
        f"\n    numpy (scipy lfilter): {t_numpy:.4f}s ({t_numpy / n_chunks * 1000:.2f} ms/chunk)"
        f"\n    mlx   (FFT convolve):  {t_mlx:.4f}s ({t_mlx / n_chunks * 1000:.2f} ms/chunk)"
        f"\n    ratio (mlx/numpy):     {t_mlx / t_numpy:.2f}x"
    )
