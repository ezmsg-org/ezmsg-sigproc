import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.butterworthzerophase import (
    ButterworthBackwardFilterTransformer,
    ButterworthZeroPhaseSettings,
    ButterworthZeroPhaseTransformer,
)


def _compute_pad_length(
    order: int,
    coef_type: str,
    fs: float,
    cutoff: float | None = None,
    cuton: float | None = None,
    settle_cutoff: float = 0.01,
    max_pad_duration: float | None = None,
) -> int:
    """Helper to compute expected pad_length using impulse response settling."""
    settings = ButterworthZeroPhaseSettings(
        order=order,
        coef_type=coef_type,
        cutoff=cutoff,
        cuton=cuton,
        settle_cutoff=settle_cutoff,
        max_pad_duration=max_pad_duration,
    )
    backward = ButterworthBackwardFilterTransformer(settings)
    return backward._compute_pad_length(fs)


@pytest.mark.parametrize(
    "cutoff, cuton",
    [
        (30.0, None),  # lowpass
        (None, 30.0),  # highpass
        (45.0, 30.0),  # bandpass
        (30.0, 45.0),  # bandstop
    ],
)
@pytest.mark.parametrize("order", [2, 4, 8])
def test_butterworth_zp_filter_specs(cutoff, cuton, order):
    """Zero-phase settings inherit filter_specs logic from ButterworthFilterSettings."""
    btype, Wn = ButterworthZeroPhaseSettings(order=order, cuton=cuton, cutoff=cutoff).filter_specs()
    if cuton is None:
        assert btype == "lowpass" and Wn == cutoff
    elif cutoff is None:
        assert btype == "highpass" and Wn == cuton
    elif cuton <= cutoff:
        assert btype == "bandpass" and Wn == (cuton, cutoff)
    else:
        assert btype == "bandstop" and Wn == (cutoff, cuton)


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_pad_length_computation(order, coef_type):
    """Verify pad_length is computed based on impulse response settling."""
    fs = 1000.0  # Use a moderate fs for this test
    settings = ButterworthZeroPhaseSettings(order=order, cutoff=30.0, coef_type=coef_type)
    backward = ButterworthBackwardFilterTransformer(settings)
    pad_length = backward._compute_pad_length(fs)

    # Verify pad_length is at least the scipy heuristic minimum
    if coef_type == "ba":
        min_length = 3 * (order + 1)
    else:
        n_sections = (order + 1) // 2
        min_length = 3 * n_sections * 2
    assert pad_length >= min_length

    # Verify pad_length is reasonable (not excessively large for this filter)
    # At fs=1000, cutoff=30 Hz gives normalized freq = 0.06, should settle quickly
    assert pad_length < 500  # Sanity check


def test_settle_cutoff_affects_pad_length():
    """Larger settle_cutoff should result in shorter pad_length."""
    fs = 1000.0
    order = 4
    cutoff = 30.0

    # Default settle_cutoff = 0.01
    settings_default = ButterworthZeroPhaseSettings(order=order, cutoff=cutoff, settle_cutoff=0.01)
    backward_default = ButterworthBackwardFilterTransformer(settings_default)
    pad_default = backward_default._compute_pad_length(fs)

    # Larger settle_cutoff = 0.1 (10% of peak instead of 1%)
    settings_larger = ButterworthZeroPhaseSettings(order=order, cutoff=cutoff, settle_cutoff=0.1)
    backward_larger = ButterworthBackwardFilterTransformer(settings_larger)
    pad_larger = backward_larger._compute_pad_length(fs)

    # Smaller settle_cutoff = 0.001 (0.1% of peak)
    settings_smaller = ButterworthZeroPhaseSettings(order=order, cutoff=cutoff, settle_cutoff=0.001)
    backward_smaller = ButterworthBackwardFilterTransformer(settings_smaller)
    pad_smaller = backward_smaller._compute_pad_length(fs)

    # Larger cutoff threshold should give shorter pad length
    assert pad_larger < pad_default
    # Smaller cutoff threshold should give longer pad length
    assert pad_smaller > pad_default


def test_max_pad_duration_caps_pad_length():
    """max_pad_duration should cap the pad_length."""
    fs = 1000.0
    order = 4
    cutoff = 1.0  # 1 Hz lowpass - very long impulse response (~2292 samples)

    # Without cap
    settings_uncapped = ButterworthZeroPhaseSettings(order=order, cutoff=cutoff)
    backward_uncapped = ButterworthBackwardFilterTransformer(settings_uncapped)
    pad_uncapped = backward_uncapped._compute_pad_length(fs)

    # With cap of 0.5 seconds = 500 samples at 1 kHz
    max_duration = 0.5
    settings_capped = ButterworthZeroPhaseSettings(order=order, cutoff=cutoff, max_pad_duration=max_duration)
    backward_capped = ButterworthBackwardFilterTransformer(settings_capped)
    pad_capped = backward_capped._compute_pad_length(fs)

    expected_max = int(max_duration * fs)

    # Uncapped should be longer than the cap
    assert pad_uncapped > expected_max, f"Expected uncapped {pad_uncapped} > {expected_max}"
    # Capped should be at most the expected max
    assert pad_capped <= expected_max
    # Capped should be exactly the max (since uncapped exceeds it)
    assert pad_capped == expected_max


def test_max_pad_duration_no_effect_when_not_limiting():
    """max_pad_duration should have no effect when pad_length is already shorter."""
    fs = 1000.0
    order = 4
    cutoff = 100.0  # Higher cutoff = faster settling

    # Get natural pad length (should be short)
    settings_natural = ButterworthZeroPhaseSettings(order=order, cutoff=cutoff)
    backward_natural = ButterworthBackwardFilterTransformer(settings_natural)
    pad_natural = backward_natural._compute_pad_length(fs)

    # With generous cap of 1 second = 1000 samples
    settings_with_cap = ButterworthZeroPhaseSettings(order=order, cutoff=cutoff, max_pad_duration=1.0)
    backward_with_cap = ButterworthBackwardFilterTransformer(settings_with_cap)
    pad_with_cap = backward_with_cap._compute_pad_length(fs)

    # Both should be equal since natural pad is well under 1 second
    assert pad_natural == pad_with_cap


def _make_message(data, dims, fs, time_axis_name="time"):
    """Helper to create AxisArray messages with frozendict axes to detect mutation."""
    axes = {}
    for i, dim in enumerate(dims):
        if dim == time_axis_name:
            axes[dim] = AxisArray.TimeAxis(fs=fs, offset=0.0)
        elif dim == "ch":
            axes[dim] = AxisArray.CoordinateAxis(data=np.arange(data.shape[i]).astype(str), dims=[dim])
        else:
            axes[dim] = AxisArray.LinearAxis(unit="", offset=0.0, gain=1.0)
    return AxisArray(data=data, dims=dims, axes=frozendict(axes), key="test")


@pytest.mark.parametrize(
    "cutoff, cuton",
    [
        (500.0, None),  # lowpass
        (None, 250.0),  # highpass
        (7500.0, 300.0),  # bandpass
        (3000.0, 6000.0),  # bandstop
    ],
)
@pytest.mark.parametrize("order", [4, 8])
@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_single_chunk_matches_reference(cutoff, cuton, order, coef_type):
    """
    Single large chunk output should be highly correlated with scipy filtfilt.

    Note: The streaming implementation initializes zi differently than scipy's
    filtfilt, so exact numerical match is not expected. Instead, we verify:
    1. Output shape is correct
    2. Output is highly correlated with reference
    3. Values are all finite
    """
    fs = 30000.0
    n_times = int(2.0 * fs)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n_times, 3))

    msg = _make_message(x, ["time", "ch"], fs)

    transformer = ButterworthZeroPhaseTransformer(
        ButterworthZeroPhaseSettings(
            axis="time",
            order=order,
            cuton=cuton,
            cutoff=cutoff,
            coef_type=coef_type,
        )
    )

    result = transformer(msg)
    pad_length = _compute_pad_length(order, coef_type, fs, cutoff=cutoff, cuton=cuton)

    # Output should be n_times - pad_length samples
    assert result.data.shape[0] == n_times - pad_length
    assert result.data.shape[1] == 3

    # All values should be finite
    assert np.isfinite(result.data).all()

    # Compute reference using scipy filtfilt on the same data (no padding)
    btype, Wn = ButterworthZeroPhaseSettings(order=order, cuton=cuton, cutoff=cutoff).filter_specs()
    if coef_type == "ba":
        b, a = scipy.signal.butter(order, Wn, btype=btype, fs=fs, output="ba")
        ref = scipy.signal.filtfilt(b, a, x, axis=0, padtype=None, padlen=0)
    else:
        sos = scipy.signal.butter(order, Wn, btype=btype, fs=fs, output="sos")
        ref = scipy.signal.sosfiltfilt(sos, x, axis=0, padtype=None, padlen=0)

    # Trim reference to match output length
    ref_trimmed = ref[: n_times - pad_length]

    # Outputs should be highly correlated (r > 0.99)
    for ch in range(result.data.shape[1]):
        # Note: Skip first pad_length samples to account for differences in initialization
        r = np.corrcoef(result.data[pad_length:, ch], ref_trimmed[pad_length:, ch])[0, 1]
        assert r > 0.999, f"Correlation {r} too low for channel {ch}"


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_streaming_chunked_processing(order, coef_type):
    """
    Verify streaming chunked processing produces valid output.

    Note: Chunked processing won't produce exactly identical results to
    single-chunk processing because the backward filter sees different amounts
    of future context at chunk boundaries. This test verifies:
    1. Output shape is correct
    2. All values are finite
    3. Output is reasonably correlated with single-chunk output
    """
    fs = 30000.0
    cuton = 300.0
    cutoff = None
    n_times = int(2.0 * fs)
    chunk_size = 48
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n_times, 3))

    # Single chunk processing (reference)
    single_transformer = ButterworthZeroPhaseTransformer(
        ButterworthZeroPhaseSettings(axis="time", order=order, cuton=cuton, cutoff=cutoff, coef_type=coef_type)
    )
    single_msg = _make_message(x, ["time", "ch"], fs)
    single_result = single_transformer(single_msg)

    # Chunked processing
    chunked_transformer = ButterworthZeroPhaseTransformer(
        ButterworthZeroPhaseSettings(axis="time", order=order, cuton=cuton, cutoff=cutoff, coef_type=coef_type)
    )

    outputs = []
    for i in range(0, n_times, chunk_size):
        chunk = x[i : i + chunk_size]
        msg = _make_message(chunk, ["time", "ch"], fs)
        result = chunked_transformer(msg)
        if result.data.size > 0:
            outputs.append(result.data)

    chunked_output = np.concatenate(outputs, axis=0)

    # Both should have the same output length
    assert chunked_output.shape == single_result.data.shape

    # All values should be finite
    assert np.isfinite(chunked_output).all()

    # Outputs should be highly correlated (r > 0.98)
    # Note: Correlation may be slightly lower for low-order filters with small chunks
    for ch in range(chunked_output.shape[1]):
        r = np.corrcoef(chunked_output[:, ch], single_result.data[:, ch])[0, 1]
        assert r > 0.98, f"Correlation {r} too low for channel {ch}"


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_warmup_returns_empty(order, coef_type):
    """During warmup (< pad_length samples), output should be empty."""
    fs = 200.0
    cutoff = 30.0
    pad_length = _compute_pad_length(order, coef_type, fs, cutoff=cutoff)

    transformer = ButterworthZeroPhaseTransformer(
        ButterworthZeroPhaseSettings(axis="time", order=order, cutoff=cutoff, coef_type=coef_type)
    )

    # Send chunks smaller than pad_length
    chunk_size = max(1, pad_length // 3)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((chunk_size, 2))
    msg = _make_message(x, ["time", "ch"], fs)

    # First chunk - should return empty but preserve non-time dimensions
    result1 = transformer(msg)
    assert result1.data.shape[0] == 0
    assert result1.data.shape[1] == 2  # Channel dimension preserved

    # Second chunk - still in warmup, non-time dimensions still preserved
    result2 = transformer(msg)
    assert result2.data.shape[0] == 0
    assert result2.data.shape[1] == 2  # Channel dimension preserved

    # After enough chunks, should start outputting
    total_sent = 2 * chunk_size
    while total_sent <= pad_length:
        result = transformer(msg)
        total_sent += chunk_size
        if total_sent <= pad_length:
            assert result.data.shape[0] == 0

    # Next chunk should produce output
    result = transformer(msg)
    assert result.data.shape[0] > 0


def test_empty_message():
    """Empty input should return empty output."""
    transformer = ButterworthZeroPhaseTransformer(
        ButterworthZeroPhaseSettings(axis="time", order=4, cutoff=30.0, coef_type="sos")
    )
    msg = AxisArray(
        data=np.zeros((0, 2)),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
            "ch": AxisArray.CoordinateAxis(data=np.array(["0", "1"]), dims=["ch"]),
        },
        key="empty",
    )
    result = transformer(msg)
    assert result.data.size == 0


def test_order_zero_passthrough():
    """Order 0 should pass data through unchanged and return same object."""
    fs = 200.0
    rng = np.random.default_rng(42)
    x = rng.standard_normal((100, 3))

    transformer = ButterworthZeroPhaseTransformer(ButterworthZeroPhaseSettings(axis="time", order=0, cutoff=30.0))
    msg = _make_message(x, ["time", "ch"], fs)
    result = transformer(msg)

    assert result is msg


@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_zero_phase_property(coef_type):
    """
    Zero-phase filter should not introduce phase delay for sinusoids
    in the passband.
    """
    fs = 1000.0
    duration = 2.0
    n_times = int(duration * fs)
    t = np.arange(n_times) / fs
    f0 = 20.0  # Test frequency in passband

    # Pure sinusoid
    x = np.sin(2 * np.pi * f0 * t)

    order = 4
    cuton = 5.0
    cutoff = 50.0
    transformer = ButterworthZeroPhaseTransformer(
        ButterworthZeroPhaseSettings(
            axis="time",
            order=order,
            cuton=cuton,
            cutoff=cutoff,  # Bandpass 5-50 Hz
            coef_type=coef_type,
        )
    )
    pad_length = _compute_pad_length(order, coef_type, fs, cutoff=cutoff, cuton=cuton)

    msg = _make_message(x.reshape(-1, 1), ["time", "ch"], fs)
    result = transformer(msg)
    y = result.data.flatten()

    # Compare in the interior region where both signals are valid
    edge = 100
    n_output = n_times - pad_length
    xi = x[edge : n_output - edge]
    yi = y[edge : n_output - edge]

    # Cross-correlation to find lag
    corr = np.correlate(yi, xi, mode="full")
    lag = np.argmax(corr) - (len(xi) - 1)

    # Zero-phase means zero lag (within 1 sample tolerance)
    assert abs(lag) <= 1


@pytest.mark.parametrize("n_dims, time_ax", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_different_axis_positions(n_dims, time_ax, coef_type):
    """Filter should work correctly with time axis in different positions."""
    fs = 200.0
    n_times = 200
    order = 4
    rng = np.random.default_rng(42)

    if n_dims == 1:
        shape = [n_times]
        dims = ["time"]
        axis_name = None
    elif n_dims == 2:
        shape = [3, 5]
        shape[time_ax] = n_times
        dims = ["ch", "freq"]
        dims[time_ax] = "time"
        axis_name = "time"
    else:
        shape = [3, 5, 7]
        shape[time_ax] = n_times
        dims = ["ch", "freq", "other"]
        dims[time_ax] = "time"
        axis_name = "time"

    x = rng.standard_normal(shape)
    msg = _make_message(x, dims, fs)

    cutoff = 30.0
    transformer = ButterworthZeroPhaseTransformer(
        ButterworthZeroPhaseSettings(axis=axis_name, order=order, cutoff=cutoff, coef_type=coef_type)
    )
    pad_length = _compute_pad_length(order, coef_type, fs, cutoff=cutoff)

    result = transformer(msg)

    # Check output shape
    expected_shape = list(shape)
    expected_shape[time_ax] = n_times - pad_length
    assert list(result.data.shape) == expected_shape

    # Check output is finite
    assert np.isfinite(result.data).all()


def test_offset_accumulates_correctly():
    """Time axis offset should be handled correctly across chunks."""
    fs = 100.0
    chunk_size = 50
    n_chunks = 5
    order = 2
    coef_type = "ba"
    cutoff = 20.0
    pad_length = _compute_pad_length(order, coef_type, fs, cutoff=cutoff)

    transformer = ButterworthZeroPhaseTransformer(
        ButterworthZeroPhaseSettings(axis="time", order=order, cutoff=cutoff, coef_type=coef_type)
    )

    rng = np.random.default_rng(42)
    total_output_samples = 0

    for i in range(n_chunks):
        x = rng.standard_normal((chunk_size, 2))
        msg = AxisArray(
            data=x,
            dims=["time", "ch"],
            axes={
                "time": AxisArray.TimeAxis(fs=fs, offset=i * chunk_size / fs),
                "ch": AxisArray.CoordinateAxis(data=np.array(["0", "1"]), dims=["ch"]),
            },
            key="test",
        )
        result = transformer(msg)

        if result.data.shape[0] > 0:
            total_output_samples += result.data.shape[0]

    # After all chunks, total output should be total_input - pad_length
    expected_output = n_chunks * chunk_size - pad_length
    assert total_output_samples == expected_output


@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_filter_actually_filters(coef_type):
    """Verify the filter actually attenuates out-of-band frequencies."""
    fs = 1000.0
    duration = 2.0
    n_times = int(duration * fs)
    t = np.arange(n_times) / fs

    # Signal: 10 Hz (in passband) + 200 Hz (out of passband for LP at 50 Hz)
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 200 * t)

    transformer = ButterworthZeroPhaseTransformer(
        ButterworthZeroPhaseSettings(axis="time", order=4, cutoff=50.0, coef_type=coef_type)
    )

    msg = _make_message(x.reshape(-1, 1), ["time", "ch"], fs)
    result = transformer(msg)
    y = result.data.flatten()

    # Check power spectrum
    fft_in = np.abs(np.fft.rfft(x))
    fft_out = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    freqs_out = np.fft.rfftfreq(len(y), 1 / fs)

    idx_10_in = np.argmin(np.abs(freqs - 10))
    idx_200_in = np.argmin(np.abs(freqs - 200))
    idx_10_out = np.argmin(np.abs(freqs_out - 10))
    idx_200_out = np.argmin(np.abs(freqs_out - 200))

    # 10 Hz should be mostly preserved
    assert fft_out[idx_10_out] > 0.5 * fft_in[idx_10_in]

    # 200 Hz should be heavily attenuated (order 4 = 80 dB/decade)
    assert fft_out[idx_200_out] < 0.01 * fft_in[idx_200_in]


def test_composite_structure():
    """Verify the composite processor has the expected structure."""
    transformer = ButterworthZeroPhaseTransformer(ButterworthZeroPhaseSettings(axis="time", order=4, cutoff=30.0))

    # Should have forward and backward processors
    assert "forward" in transformer._procs
    assert "backward" in transformer._procs

    # Forward should be ButterworthFilterTransformer
    from ezmsg.sigproc.butterworthfilter import ButterworthFilterTransformer

    assert isinstance(transformer._procs["forward"], ButterworthFilterTransformer)

    # Backward should be ButterworthBackwardFilterTransformer
    assert isinstance(transformer._procs["backward"], ButterworthBackwardFilterTransformer)
