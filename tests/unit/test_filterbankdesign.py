import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.filterbank import FilterbankMode, MinPhaseMode
from ezmsg.sigproc.filterbankdesign import (
    FilterbankDesignSettings,
    FilterbankDesignTransformer,
)
from ezmsg.sigproc.kaiser import KaiserFilterSettings, kaiser_design_fun


@pytest.mark.parametrize("n_filters", [1, 3, 5])
def test_calculate_kernels_basic(n_filters):
    """Test the _calculate_kernels method with varying numbers of filters."""
    fs = 200.0

    # Create filter settings for different frequency bands
    filters = []
    for i in range(n_filters):
        cutoff = 10.0 + i * 10.0  # 10Hz, 20Hz, 30Hz, etc.
        filters.append(
            KaiserFilterSettings(
                cutoff=cutoff,
                ripple=60.0,
                width=5.0,
                pass_zero=True,
                wn_hz=True,
            )
        )

    settings = FilterbankDesignSettings(filters=filters)
    transformer = FilterbankDesignTransformer(settings=settings)

    # Calculate kernels
    kernels = transformer._calculate_kernels(fs)

    assert len(kernels) == n_filters
    for kernel in kernels:
        assert isinstance(kernel, np.ndarray)
        assert len(kernel) > 0
        # Kaiser filters should have odd number of taps
        assert len(kernel) % 2 == 1


@pytest.mark.parametrize("mode", [FilterbankMode.CONV, FilterbankMode.FFT, FilterbankMode.AUTO])
def test_filterbankdesign_transformer_modes(mode):
    """Test FilterbankDesignTransformer with different processing modes."""
    fs = 200.0
    dur = 5.0
    n_times = int(dur * fs)

    # Create test signal
    t = np.arange(n_times) / fs
    # Signal with 10Hz, 40Hz, and 80Hz components
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 40 * t) + np.sin(2 * np.pi * 80 * t)

    # Create filterbank with 3 bandpass filters
    filters = [
        # 5-15 Hz (should pass 10Hz)
        KaiserFilterSettings(
            cutoff=[5.0, 15.0],
            ripple=60.0,
            width=5.0,
            pass_zero="bandpass",
            wn_hz=True,
        ),
        # 30-50 Hz (should pass 40Hz)
        KaiserFilterSettings(
            cutoff=[30.0, 50.0],
            ripple=60.0,
            width=5.0,
            pass_zero="bandpass",
            wn_hz=True,
        ),
        # 70-90 Hz (should pass 80Hz)
        KaiserFilterSettings(
            cutoff=[70.0, 90.0],
            ripple=60.0,
            width=5.0,
            pass_zero="bandpass",
            wn_hz=True,
        ),
    ]

    settings = FilterbankDesignSettings(filters=filters, mode=mode, axis="time")
    transformer = FilterbankDesignTransformer(settings=settings)

    # For FFT mode, use larger chunks to avoid windowing issues
    # For CONV and AUTO, split into smaller messages to test streaming
    if mode == FilterbankMode.FFT:
        # Single large message for FFT mode
        messages = [
            AxisArray(
                data=signal,
                dims=["time"],
                axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
                key="test_filterbankdesign",
            )
        ]
    else:
        # Split into multiple messages for CONV/AUTO modes
        n_splits = 4
        messages = []
        for split_dat in np.array_split(signal, n_splits):
            offset = len(messages) * len(split_dat) / fs
            messages.append(
                AxisArray(
                    data=split_dat,
                    dims=["time"],
                    axes={"time": AxisArray.TimeAxis(fs=fs, offset=offset)},
                    key="test_filterbankdesign",
                )
            )

    # Process messages
    result_msgs = [transformer(msg) for msg in messages]
    if len(result_msgs) > 1:
        result = AxisArray.concatenate(*result_msgs, dim="time")
    else:
        result = result_msgs[0]

    # Verify output structure
    assert "kernel" in result.dims
    assert "time" in result.dims
    assert result.data.shape[0] == 3  # 3 filters

    # Verify frequency selectivity using FFT
    # FFT mode has delay/windowing that reduces output length, so check we have enough data
    if result.data.shape[1] > 200:  # Need at least 200 samples for meaningful FFT
        transient = 100
        fft_in = np.abs(np.fft.rfft(signal[transient : transient + result.data.shape[1] - transient]))
        freqs = np.fft.rfftfreq(result.data.shape[1] - transient, 1 / fs)

        for filter_idx, target_freq in enumerate([10.0, 40.0, 80.0]):
            fft_out = np.abs(np.fft.rfft(result.data[filter_idx, transient:]))

            idx_target = np.argmin(np.abs(freqs - target_freq))

            # Target frequency should have significant power
            assert fft_out[idx_target] > 0.3 * fft_in[idx_target]


@pytest.mark.parametrize("n_chans", [1, 3])
@pytest.mark.parametrize("time_ax", [0, 1])
def test_filterbankdesign_multidim(n_chans, time_ax):
    """Test FilterbankDesignTransformer with multi-dimensional data."""
    fs = 200.0
    dur = 1.0
    n_times = int(dur * fs)

    # Create test data
    if n_chans == 1:
        data_shape = [n_times]
        dims = ["time"]
        axes = {"time": AxisArray.TimeAxis(fs=fs, offset=0)}
    else:
        if time_ax == 0:
            data_shape = [n_times, n_chans]
            dims = ["time", "ch"]
        else:
            data_shape = [n_chans, n_times]
            dims = ["ch", "time"]
        axes = {
            "time": AxisArray.TimeAxis(fs=fs, offset=0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(n_chans).astype(str), dims=["ch"]),
        }

    data = np.random.randn(*data_shape)

    # Create simple lowpass filter
    filters = [
        KaiserFilterSettings(
            cutoff=30.0,
            ripple=60.0,
            width=10.0,
            pass_zero=True,
            wn_hz=True,
        )
    ]

    settings = FilterbankDesignSettings(filters=filters, axis="time", mode=FilterbankMode.CONV)
    transformer = FilterbankDesignTransformer(settings=settings)

    msg = AxisArray(data=data, dims=dims, axes=axes, key="test_multidim")
    result = transformer(msg)

    # Verify output dimensions
    assert "kernel" in result.dims
    assert "time" in result.dims
    if n_chans > 1:
        assert "ch" in result.dims


@pytest.mark.skip(reason="Empty messages are not supported by filterbank")
def test_filterbankdesign_empty_message():
    """Test FilterbankDesignTransformer with empty message.

    Note: Empty messages are not supported by the filterbank transformer.
    This test is skipped as it's not a valid use case.
    """
    fs = 200.0

    filters = [
        KaiserFilterSettings(
            cutoff=30.0,
            ripple=60.0,
            width=10.0,
            pass_zero=True,
            wn_hz=True,
        )
    ]

    settings = FilterbankDesignSettings(filters=filters, axis="time", mode=FilterbankMode.CONV)
    transformer = FilterbankDesignTransformer(settings=settings)

    msg = AxisArray(
        data=np.zeros((0,)),
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_empty",
    )

    result = transformer(msg)
    assert result.data.size == 0
    assert "kernel" in result.dims


def test_filterbankdesign_normalized_frequencies():
    """Test FilterbankDesignTransformer with normalized frequencies."""
    fs = 200.0
    dur = 1.0
    n_times = int(dur * fs)

    # Create test signal
    t = np.arange(n_times) / fs
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)

    # Create lowpass filter with normalized frequency
    # Cutoff at 0.3 (30Hz / 100Hz Nyquist)
    filters = [
        KaiserFilterSettings(
            cutoff=0.3,
            ripple=60.0,
            width=0.1,  # normalized width
            pass_zero=True,
            wn_hz=False,  # Use normalized frequencies
        )
    ]

    settings = FilterbankDesignSettings(filters=filters, axis="time", mode=FilterbankMode.CONV)
    transformer = FilterbankDesignTransformer(settings=settings)

    msg = AxisArray(
        data=signal,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_normalized",
    )

    result = transformer(msg)

    # Verify frequency response
    transient = 50
    fft_in = np.abs(np.fft.rfft(signal[transient:]))
    fft_out = np.abs(np.fft.rfft(result.data[0, transient:]))
    freqs = np.fft.rfftfreq(len(signal[transient:]), 1 / fs)

    idx_10hz = np.argmin(np.abs(freqs - 10))
    idx_60hz = np.argmin(np.abs(freqs - 60))

    # 10Hz should be mostly preserved
    assert fft_out[idx_10hz] > 0.7 * fft_in[idx_10hz]
    # 60Hz should be attenuated
    assert fft_out[idx_60hz] < 0.1 * fft_in[idx_60hz]


@pytest.mark.parametrize("min_phase", [MinPhaseMode.NONE, MinPhaseMode.HILBERT, MinPhaseMode.HOMOMORPHIC])
def test_filterbankdesign_min_phase(min_phase):
    """Test FilterbankDesignTransformer with different minimum phase modes."""
    fs = 200.0
    dur = 1.0
    n_times = int(dur * fs)

    # Create test signal
    signal = np.random.randn(n_times)

    # Create filter
    filters = [
        KaiserFilterSettings(
            cutoff=30.0,
            ripple=60.0,
            width=10.0,
            pass_zero=True,
            wn_hz=True,
        )
    ]

    settings = FilterbankDesignSettings(filters=filters, axis="time", min_phase=min_phase, mode=FilterbankMode.CONV)
    transformer = FilterbankDesignTransformer(settings=settings)

    msg = AxisArray(
        data=signal,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_min_phase",
    )

    result = transformer(msg)

    # Should process successfully
    assert result.data.shape[0] == 1  # 1 filter
    assert result.data.shape[1] == n_times


def test_filterbankdesign_update_settings():
    """Test update_settings functionality."""
    fs = 200.0
    dur = 2.0
    n_times = int(dur * fs)

    # Create test signal
    t = np.arange(n_times) / fs
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)

    msg = AxisArray(
        data=signal,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_update",
    )

    # Initial filter - lowpass at 30Hz
    filters_low = [
        KaiserFilterSettings(
            cutoff=30.0,
            ripple=60.0,
            width=10.0,
            pass_zero=True,
            wn_hz=True,
        )
    ]

    settings = FilterbankDesignSettings(filters=filters_low, axis="time", mode=FilterbankMode.CONV)
    transformer = FilterbankDesignTransformer(settings=settings)

    # Process first message
    result1 = transformer(msg)

    # Update to highpass at 40Hz
    filters_high = [
        KaiserFilterSettings(
            cutoff=40.0,
            ripple=60.0,
            width=10.0,
            pass_zero=False,  # highpass
            wn_hz=True,
        )
    ]

    new_settings = FilterbankDesignSettings(filters=filters_high, axis="time", mode=FilterbankMode.CONV)
    transformer.update_settings(new_settings=new_settings)

    # Process second message with updated settings
    result2 = transformer(msg)

    # Results should be different
    assert not np.allclose(result1.data, result2.data)

    # Verify frequency characteristics changed
    transient = 50
    fft_in = np.abs(np.fft.rfft(signal[transient:]))
    freqs = np.fft.rfftfreq(len(signal[transient:]), 1 / fs)

    fft_out1 = np.abs(np.fft.rfft(result1.data[0, transient:]))
    fft_out2 = np.abs(np.fft.rfft(result2.data[0, transient:]))

    idx_10hz = np.argmin(np.abs(freqs - 10))
    idx_60hz = np.argmin(np.abs(freqs - 60))

    # Result1 (lowpass) should pass 10Hz, attenuate 60Hz
    assert fft_out1[idx_10hz] > 0.7 * fft_in[idx_10hz]
    assert fft_out1[idx_60hz] < 0.1 * fft_in[idx_60hz]

    # Result2 (highpass) should attenuate 10Hz, pass 60Hz
    assert fft_out2[idx_10hz] < 0.1 * fft_in[idx_10hz]
    assert fft_out2[idx_60hz] > 0.7 * fft_in[idx_60hz]


def test_filterbankdesign_different_filter_types():
    """Test FilterbankDesignTransformer with various filter types."""
    fs = 200.0
    dur = 2.0
    n_times = int(dur * fs)

    # Create test signal with multiple frequency components
    t = np.arange(n_times) / fs
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 40 * t) + np.sin(2 * np.pi * 80 * t)

    # Create different types of filters
    filters = [
        # Lowpass
        KaiserFilterSettings(
            cutoff=20.0,
            ripple=60.0,
            width=5.0,
            pass_zero=True,
            wn_hz=True,
        ),
        # Highpass
        KaiserFilterSettings(
            cutoff=60.0,
            ripple=60.0,
            width=5.0,
            pass_zero=False,
            wn_hz=True,
        ),
        # Bandpass
        KaiserFilterSettings(
            cutoff=[30.0, 50.0],
            ripple=60.0,
            width=5.0,
            pass_zero="bandpass",
            wn_hz=True,
        ),
        # Bandstop
        KaiserFilterSettings(
            cutoff=[35.0, 45.0],
            ripple=60.0,
            width=5.0,
            pass_zero="bandstop",
            wn_hz=True,
        ),
    ]

    settings = FilterbankDesignSettings(filters=filters, axis="time", mode=FilterbankMode.CONV)
    transformer = FilterbankDesignTransformer(settings=settings)

    msg = AxisArray(
        data=signal,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_filter_types",
    )

    result = transformer(msg)

    # Verify output structure
    assert result.data.shape[0] == 4  # 4 filters
    assert result.data.shape[1] == n_times

    # Basic sanity check - outputs should be different from each other
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.allclose(result.data[i], result.data[j])


def test_filterbankdesign_streaming():
    """Test FilterbankDesignTransformer with streaming data (multiple messages)."""
    fs = 200.0
    dur = 5.0
    n_times = int(dur * fs)

    # Create long test signal
    t = np.arange(n_times) / fs
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

    # Create filter
    filters = [
        KaiserFilterSettings(
            cutoff=30.0,
            ripple=60.0,
            width=10.0,
            pass_zero=True,
            wn_hz=True,
        )
    ]

    settings = FilterbankDesignSettings(filters=filters, axis="time", mode=FilterbankMode.CONV)
    transformer = FilterbankDesignTransformer(settings=settings)

    # Split into many small messages
    chunk_size = 50
    messages = []
    for i in range(0, n_times, chunk_size):
        chunk = signal[i : i + chunk_size]
        messages.append(
            AxisArray(
                data=chunk,
                dims=["time"],
                axes={"time": AxisArray.TimeAxis(fs=fs, offset=i / fs)},
                key="test_streaming",
            )
        )

    # Process all messages
    results = [transformer(msg) for msg in messages]
    result = AxisArray.concatenate(*results, dim="time")

    # Compare to reference implementation
    # Design the same filter manually
    coefs = kaiser_design_fun(
        fs=fs,
        cutoff=30.0,
        ripple=60.0,
        width=10.0,
        pass_zero=True,
        wn_hz=True,
    )
    b, a = coefs
    zi = scipy.signal.lfiltic(b, a, [])
    expected, _ = scipy.signal.lfilter(b, a, signal, zi=zi)

    # Results should match (after initial transient)
    transient = len(b)
    assert np.allclose(result.data[0, transient:], expected[transient:], rtol=1e-5, atol=1e-8)


def test_filterbankdesign_comparison_with_filterbank():
    """Verify FilterbankDesignTransformer produces same results as FilterbankTransformer with
    manually designed kernels."""
    fs = 200.0
    dur = 2.0
    n_times = int(dur * fs)

    # Create test signal
    signal = np.random.randn(n_times)

    # Design filters manually
    filter_settings = [
        KaiserFilterSettings(
            cutoff=20.0,
            ripple=60.0,
            width=10.0,
            pass_zero=True,
            wn_hz=True,
        ),
        KaiserFilterSettings(
            cutoff=50.0,
            ripple=60.0,
            width=10.0,
            pass_zero=False,
            wn_hz=True,
        ),
    ]

    # Calculate kernels manually
    kernels = []
    for filt_set in filter_settings:
        coefs = kaiser_design_fun(
            fs=fs,
            cutoff=filt_set.cutoff,
            ripple=filt_set.ripple,
            width=filt_set.width,
            pass_zero=filt_set.pass_zero,
            wn_hz=filt_set.wn_hz,
        )
        kernels.append(coefs[0])

    # Use FilterbankTransformer directly
    from ezmsg.sigproc.filterbank import FilterbankSettings, FilterbankTransformer

    filterbank_settings = FilterbankSettings(
        kernels=kernels,
        mode=FilterbankMode.CONV,
        axis="time",
    )
    filterbank_transformer = FilterbankTransformer(settings=filterbank_settings)

    # Use FilterbankDesignTransformer
    design_settings = FilterbankDesignSettings(
        filters=filter_settings,
        mode=FilterbankMode.CONV,
        axis="time",
    )
    design_transformer = FilterbankDesignTransformer(settings=design_settings)

    # Create message
    msg = AxisArray(
        data=signal,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_comparison",
    )

    # Process with both transformers
    result_filterbank = filterbank_transformer(msg)
    result_design = design_transformer(msg)

    # Results should be identical
    assert np.allclose(result_filterbank.data, result_design.data, rtol=1e-10, atol=1e-12)


def test_filterbankdesign_ripple_width_variations():
    """Test FilterbankDesignTransformer with varying ripple and width parameters."""
    fs = 200.0
    dur = 1.0
    n_times = int(dur * fs)

    signal = np.random.randn(n_times)

    # Test different ripple values (affects filter steepness)
    for ripple in [40.0, 60.0, 80.0]:
        filters = [
            KaiserFilterSettings(
                cutoff=30.0,
                ripple=ripple,
                width=10.0,
                pass_zero=True,
                wn_hz=True,
            )
        ]

        settings = FilterbankDesignSettings(filters=filters, axis="time", mode=FilterbankMode.CONV)
        transformer = FilterbankDesignTransformer(settings=settings)

        msg = AxisArray(
            data=signal,
            dims=["time"],
            axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
            key=f"test_ripple_{ripple}",
        )

        result = transformer(msg)
        assert result.data.shape == (1, n_times)

    # Test different width values (affects transition width)
    for width in [5.0, 10.0, 20.0]:
        filters = [
            KaiserFilterSettings(
                cutoff=30.0,
                ripple=60.0,
                width=width,
                pass_zero=True,
                wn_hz=True,
            )
        ]

        settings = FilterbankDesignSettings(filters=filters, axis="time", mode=FilterbankMode.CONV)
        transformer = FilterbankDesignTransformer(settings=settings)

        msg = AxisArray(
            data=signal,
            dims=["time"],
            axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
            key=f"test_width_{width}",
        )

        result = transformer(msg)
        assert result.data.shape == (1, n_times)
