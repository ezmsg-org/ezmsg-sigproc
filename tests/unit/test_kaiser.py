import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.kaiser import (
    KaiserFilterSettings,
    KaiserFilterTransformer,
    kaiser_design_fun,
)


@pytest.mark.parametrize(
    "cutoff, pass_zero",
    [
        (30.0, True),  # lowpass
        (30.0, False),  # highpass
        ([30.0, 45.0], "bandpass"),  # bandpass
        ([30.0, 45.0], "bandstop"),  # bandstop
    ],
)
@pytest.mark.parametrize("ripple", [40.0, 60.0, 80.0])  # dB attenuation
@pytest.mark.parametrize("width", [5.0, 10.0])  # Transition width in Hz
def test_kaiser_design_fun(cutoff, pass_zero, ripple, width):
    """Test the Kaiser filter design function with various parameters."""
    fs = 200.0
    result = kaiser_design_fun(
        fs=fs,
        cutoff=cutoff,
        ripple=ripple,
        width=width,
        pass_zero=pass_zero,
        wn_hz=True,
    )

    assert result is not None
    b, a = result
    # Kaiser filter should have odd number of taps
    assert len(b) % 2 == 1
    assert len(a) == 1
    assert a[0] == 1.0
    # Higher ripple (more attenuation) should require more taps
    # (This is a general trend but not strict)
    assert len(b) > 5


def test_kaiser_design_fun_missing_params():
    """Test that missing parameters return None."""
    fs = 200.0

    # Missing ripple
    result = kaiser_design_fun(
        fs=fs,
        cutoff=30.0,
        ripple=None,
        width=10.0,
        pass_zero=True,
        wn_hz=True,
    )
    assert result is None

    # Missing width
    result = kaiser_design_fun(
        fs=fs,
        cutoff=30.0,
        ripple=60.0,
        width=None,
        pass_zero=True,
        wn_hz=True,
    )
    assert result is None

    # Missing cutoff
    result = kaiser_design_fun(
        fs=fs,
        cutoff=None,
        ripple=60.0,
        width=10.0,
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
@pytest.mark.parametrize("ripple", [60.0])
@pytest.mark.parametrize("width", [10.0])
@pytest.mark.parametrize("fs", [200.0])
@pytest.mark.parametrize("n_chans", [3])
@pytest.mark.parametrize("n_dims, time_ax", [(1, 0), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_kaiser_filter_transformer(
    cutoff,
    pass_zero,
    ripple,
    width,
    fs,
    n_chans,
    n_dims,
    time_ax,
    coef_type,
):
    """Test Kaiser filter transformer with various configurations."""
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

    # Calculate Expected Result using scipy directly
    coefs = kaiser_design_fun(
        fs=fs,
        cutoff=cutoff,
        ripple=ripple,
        width=width,
        pass_zero=pass_zero,
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
                key="test_kaiser",
            )
        )
        n_seen += split_dat.shape[time_ax]

    # Create transformer
    axis_name = "time" if time_ax != 0 else None
    settings = KaiserFilterSettings(
        axis=axis_name,
        cutoff=cutoff,
        ripple=ripple,
        width=width,
        pass_zero=pass_zero,
        wn_hz=True,
        coef_type=coef_type,
    )
    transformer = KaiserFilterTransformer(settings=settings)

    # Process messages
    result = []
    for msg in messages:
        out_msg = transformer(msg)
        result.append(out_msg.data)
    result = np.concatenate(result, axis=time_ax)

    assert np.allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_kaiser_filter_empty_msg():
    """Test Kaiser filter with empty message."""
    settings = KaiserFilterSettings(
        axis="time",
        cutoff=30.0,
        ripple=60.0,
        width=10.0,
        pass_zero=True,
        coef_type="ba",
    )
    transformer = KaiserFilterTransformer(settings=settings)

    msg_in = AxisArray(
        data=np.zeros((0, 2)),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(2).astype(str), dims=["ch"]),
        },
        key="test_kaiser_empty",
    )

    result = transformer(msg_in)
    assert result.data.size == 0


def test_kaiser_filter_frequency_response():
    """Test Kaiser filter frequency response characteristics."""
    fs = 200.0
    dur = 2.0
    n_times = int(dur * fs)

    # Create test signal with multiple frequency components
    t = np.arange(n_times) / fs
    # 10Hz (should pass), 40Hz (transition), 60Hz (should stop)
    in_dat = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 40 * t) + np.sin(2 * np.pi * 60 * t)

    msg = AxisArray(
        data=in_dat,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_kaiser_freq_response",
    )

    # Design lowpass filter at 30Hz with 10Hz transition width
    # Should pass 10Hz, attenuate 60Hz
    settings = KaiserFilterSettings(
        axis="time",
        cutoff=30.0,
        ripple=60.0,  # 60dB attenuation in stopband
        width=10.0,  # 10Hz transition width (30Hz to 40Hz)
        pass_zero=True,
        wn_hz=True,
        coef_type="ba",
    )
    transformer = KaiserFilterTransformer(settings=settings)

    result = transformer(msg)

    # Analyze frequency content
    fft_in = np.abs(np.fft.rfft(in_dat))
    fft_out = np.abs(np.fft.rfft(result.data))
    freqs = np.fft.rfftfreq(n_times, 1 / fs)

    idx_10hz = np.argmin(np.abs(freqs - 10))
    idx_60hz = np.argmin(np.abs(freqs - 60))

    # 10Hz should be mostly preserved (passband)
    assert fft_out[idx_10hz] > 0.85 * fft_in[idx_10hz]
    # 60Hz should be highly attenuated (stopband)
    # With 60dB ripple specification, attenuation should be significant
    assert fft_out[idx_60hz] < 0.01 * fft_in[idx_60hz]


def test_kaiser_filter_normalized_frequencies():
    """Test Kaiser filter with normalized frequencies (wn_hz=False)."""
    fs = 200.0
    dur = 1.0
    n_times = int(dur * fs)

    # Create input signal
    t = np.arange(n_times) / fs
    in_dat = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)

    msg = AxisArray(
        data=in_dat,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_normalized",
    )

    # Design filter with normalized frequencies
    # Cutoff at 0.3 (30Hz / 100Hz Nyquist)
    # Width of 0.1 (10Hz / 100Hz Nyquist)
    settings = KaiserFilterSettings(
        axis="time",
        cutoff=0.3,
        ripple=60.0,
        width=0.1,
        pass_zero=True,
        wn_hz=False,  # Use normalized frequencies
        coef_type="ba",
    )
    transformer = KaiserFilterTransformer(settings=settings)

    result = transformer(msg)

    # Verify frequency response
    fft_in = np.abs(np.fft.rfft(in_dat))
    fft_out = np.abs(np.fft.rfft(result.data))
    freqs = np.fft.rfftfreq(n_times, 1 / fs)

    idx_10hz = np.argmin(np.abs(freqs - 10))
    idx_60hz = np.argmin(np.abs(freqs - 60))

    # 10Hz should be mostly preserved (passband)
    assert fft_out[idx_10hz] > 0.80 * fft_in[idx_10hz]
    # 60Hz should be highly attenuated (stopband)
    assert fft_out[idx_60hz] < 0.01 * fft_in[idx_60hz]


def test_kaiser_filter_vs_fir_with_kaiser_window():
    """Verify Kaiser filter produces similar results to FIR with Kaiser window."""
    from ezmsg.sigproc.firfilter import FIRFilterSettings, FIRFilterTransformer

    fs = 200.0
    dur = 1.0
    n_times = int(dur * fs)

    # Create test signal
    t = np.arange(n_times) / fs
    in_dat = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)

    msg = AxisArray(
        data=in_dat,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=fs, offset=0)},
        key="test_comparison",
    )

    # Kaiser filter
    kaiser_settings = KaiserFilterSettings(
        axis="time",
        cutoff=30.0,
        ripple=60.0,
        width=10.0,
        pass_zero=True,
        wn_hz=True,
        coef_type="ba",
    )
    kaiser_transformer = KaiserFilterTransformer(settings=kaiser_settings)

    # Get the designed filter parameters
    coefs = kaiser_design_fun(
        fs=fs,
        cutoff=30.0,
        ripple=60.0,
        width=10.0,
        pass_zero=True,
        wn_hz=True,
    )
    b_kaiser, a_kaiser = coefs
    n_taps = len(b_kaiser)

    # Calculate beta from ripple
    ripple_db = 60.0
    beta = scipy.signal.kaiser_beta(ripple_db)

    # FIR filter with equivalent Kaiser window
    fir_settings = FIRFilterSettings(
        axis="time",
        order=n_taps,
        cutoff=30.0,
        window=("kaiser", beta),
        pass_zero=True,
        scale=False,  # Kaiser filter uses scale=False
        wn_hz=True,
        coef_type="ba",
    )
    fir_transformer = FIRFilterTransformer(settings=fir_settings)

    result_kaiser = kaiser_transformer(msg)
    result_fir = fir_transformer(msg)

    # Results should be very similar
    assert np.allclose(result_kaiser.data, result_fir.data, rtol=1e-3, atol=1e-5)


def test_kaiser_higher_ripple_more_taps():
    """Verify that higher ripple requirements result in more filter taps."""
    fs = 200.0

    # Lower ripple (less stringent) should use fewer taps
    coefs_low = kaiser_design_fun(
        fs=fs,
        cutoff=30.0,
        ripple=40.0,
        width=10.0,
        pass_zero=True,
        wn_hz=True,
    )

    # Higher ripple (more stringent) should use more taps
    coefs_high = kaiser_design_fun(
        fs=fs,
        cutoff=30.0,
        ripple=80.0,
        width=10.0,
        pass_zero=True,
        wn_hz=True,
    )

    b_low, _ = coefs_low
    b_high, _ = coefs_high

    # Higher ripple should require more taps
    assert len(b_high) > len(b_low)
