import numpy as np
import pytest
from scipy import signal
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.combfilter import (
    comb_design_fun,
    CombFilterTransformer,
)


def test_comb_filter_design():
    """Test that comb filters are correctly designed with the expected number of notches/peaks."""
    fs = 1000.0
    fund_freq = 50.0
    num_harmonics = 3

    # Test notch filter
    sos_coeffs = comb_design_fun(
        fs=fs,
        fundamental_freq=fund_freq,
        num_harmonics=num_harmonics,
        filter_type="notch",
        coef_type="sos",
    )

    # Check that we got the expected number of SOS sections
    assert sos_coeffs.shape == (num_harmonics, 6)

    # Test peak filter
    sos_coeffs = comb_design_fun(
        fs=fs,
        fundamental_freq=fund_freq,
        num_harmonics=num_harmonics,
        filter_type="peak",
        coef_type="sos",
    )

    # Check that we got the expected number of SOS sections
    assert sos_coeffs.shape == (num_harmonics, 6)

    # Test ba coefficients
    ba_coeffs = comb_design_fun(
        fs=fs,
        fundamental_freq=fund_freq,
        num_harmonics=num_harmonics,
        filter_type="notch",
        coef_type="ba",
    )

    # Check that ba coefficients are returned as a tuple of arrays
    assert isinstance(ba_coeffs, tuple)
    assert len(ba_coeffs) == 2
    assert isinstance(ba_coeffs[0], np.ndarray)
    assert isinstance(ba_coeffs[1], np.ndarray)


def _debug_plot(sos, fs, N=8000):
    """Helper function to plot frequency response of a filter."""
    freqs, resp = signal.sosfreqz(sos, worN=N, fs=fs)
    mag = np.abs(resp)
    phase = np.angle(resp)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(freqs, mag)
    axs[0].set_ylabel("Magnitude")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_title("Frequency Response")

    axs[1].plot(freqs, phase)
    axs[1].set_ylabel("Phase")
    axs[1].set_xlabel("Frequency (Hz)")

    plt.show()


def test_comb_filter_frequency_response():
    """Test that the comb filter has the expected frequency response."""
    fs = 1000.0
    fund_freq = 100.0
    num_harmonics = 3
    q_factor = 35.0

    # Create notch filter
    notch_sos = comb_design_fun(
        fs=fs,
        fundamental_freq=fund_freq,
        num_harmonics=num_harmonics,
        q_factor=q_factor,
        filter_type="notch",
        coef_type="sos",
    )
    # _debug_plot(notch_sos, fs)

    # Create peak filter
    peak_sos = comb_design_fun(
        fs=fs,
        fundamental_freq=fund_freq,
        num_harmonics=num_harmonics,
        q_factor=q_factor,
        filter_type="peak",
        coef_type="sos",
    )
    # _debug_plot(peak_sos, fs)

    # Compute frequency responses
    freqs, notch_resp = signal.sosfreqz(notch_sos, worN=8000, fs=fs)
    _, peak_resp = signal.sosfreqz(peak_sos, worN=8000, fs=fs)

    notch_mag = np.abs(notch_resp)
    peak_mag = np.abs(peak_resp)

    # Check attenuation at harmonic frequencies for notch filter
    for i in range(1, num_harmonics + 1):
        harmonic_freq = fund_freq * i
        idx = np.argmin(np.abs(freqs - harmonic_freq))

        # Notch should attenuate at harmonics
        assert notch_mag[idx] < 0.5  # significant attenuation

        # Peak should amplify at harmonics
        assert peak_mag[idx] > (
            1.5 * peak_mag[0]
        )  # passed peak is larger than stopped non-peak.


def test_nyquist_limitation():
    """Test that frequencies above Nyquist are properly skipped."""
    fs = 300.0  # Low sample rate
    fund_freq = 60.0
    num_harmonics = 5  # This would create harmonics above Nyquist

    sos_coeffs = comb_design_fun(
        fs=fs,
        fundamental_freq=fund_freq,
        num_harmonics=num_harmonics,
        filter_type="notch",
        coef_type="sos",
    )

    # Should only have harmonics below Nyquist (60, 120 Hz), not 180, 240, 300 Hz
    assert sos_coeffs.shape == (2, 6)


def test_bandwidth_methods():
    """Test that different bandwidth methods affect Q factors correctly."""
    fs = 1000.0
    fund_freq = 50.0
    num_harmonics = 3
    q_factor = 10.0

    # Constant bandwidth
    const_sos = comb_design_fun(
        fs=fs,
        fundamental_freq=fund_freq,
        num_harmonics=num_harmonics,
        q_factor=q_factor,
        quality_scaling="constant",
        filter_type="notch",
        coef_type="sos",
    )
    # _debug_plot(const_sos, fs)

    # Proportional bandwidth
    prop_sos = comb_design_fun(
        fs=fs,
        fundamental_freq=fund_freq,
        num_harmonics=num_harmonics,
        q_factor=q_factor,
        quality_scaling="proportional",
        filter_type="notch",
        coef_type="sos",
    )
    # _debug_plot(prop_sos, fs)

    # Compute frequency responses
    freqs, const_resp = signal.sosfreqz(const_sos, worN=8000, fs=fs)
    _, prop_resp = signal.sosfreqz(prop_sos, worN=8000, fs=fs)

    const_mag = np.abs(const_resp)
    prop_mag = np.abs(prop_resp)

    # For harmonic 3 (150 Hz), proportional bandwidth should be narrower
    # than constant bandwidth, resulting in a steeper transition
    harmonic_idx = np.argmin(np.abs(freqs - 150))

    # Find points 5 Hz away from the harmonic
    sidx1 = np.argmin(np.abs(freqs - 145))
    sidx2 = np.argmin(np.abs(freqs - 155))

    # Calculate average transition steepness
    const_steep = (const_mag[sidx1] + const_mag[sidx2]) / 2 - const_mag[harmonic_idx]
    prop_steep = (prop_mag[sidx1] + prop_mag[sidx2]) / 2 - prop_mag[harmonic_idx]

    # Proportional bandwidth should give steeper transitions for higher harmonics
    assert prop_steep > const_steep


def test_comb_transformer():
    """Test that CombFilterTransformer correctly initializes and uses settings."""
    transformer = CombFilterTransformer(
        axis="time",
        fundamental_freq=60.0,
        num_harmonics=4,
        q_factor=25.0,
        filter_type="notch",
        coef_type="sos",
        quality_scaling="constant",
    )

    # Check that transformer is correctly initialized
    assert isinstance(transformer, CombFilterTransformer)
    assert transformer.settings.fundamental_freq == 60.0
    assert transformer.settings.num_harmonics == 4
    assert transformer.settings.q_factor == 25.0
    assert transformer.settings.filter_type == "notch"
    assert transformer.settings.coef_type == "sos"
    assert transformer.settings.quality_scaling == "constant"


def test_invalid_coef_type():
    """Test that an invalid coefficient type raises an error."""
    with pytest.raises(ValueError):
        comb_design_fun(
            fs=1000.0,
            fundamental_freq=50.0,
            num_harmonics=3,
            coef_type="invalid",  # Invalid coefficient type
        )


def test_empty_harmonics_return():
    """Test that empty harmonics return None."""
    # All harmonics above Nyquist
    result = comb_design_fun(
        fs=20.0,  # Very low sample rate
        fundamental_freq=15.0,  # Above Nyquist
        num_harmonics=3,
    )

    assert result is None


def test_comb_filter_update_settings():
    """Test the update_settings functionality of the Comb filter."""
    # Setup parameters
    fs = 500.0
    dur = 2.0
    n_times = int(dur * fs)
    n_chans = 2

    # Create input data with sine waves at fundamental frequency and harmonics
    t = np.arange(n_times) / fs
    # Create 50Hz fundamental and its harmonics (50Hz, 100Hz) for ch0
    # And 60Hz fundamental and harmonics (60Hz, 120Hz) for ch1
    in_data = np.zeros((n_times, n_chans))
    in_data[:, 0] = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
    in_data[:, 1] = np.sin(2 * np.pi * 60 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

    # Create message
    msg_in = AxisArray(
        data=in_data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=0),
            "ch": AxisArray.CoordinateAxis(
                data=np.arange(n_chans).astype(str), dims=["ch"]
            ),
        },
        key="test_comb_filter_update_settings",
    )

    def _calc_power(msg, fund_freqs, n_harmonics=2):
        """Calculate power at specified frequencies for each channel."""
        fft_result = np.abs(np.fft.rfft(msg.data, axis=0)) ** 2
        fft_freqs = np.fft.rfftfreq(len(msg.data), 1 / fs)

        # Return dims: (fund_freqs, harmonics, channels)
        return np.array(
            [
                [
                    fft_result[np.argmin(np.abs(fft_freqs - (i * fund)))]
                    for i in range(1, n_harmonics + 1)
                ]
                for fund in fund_freqs
            ]
        )

    # Calculate power in original signal
    power0 = _calc_power(msg_in, fund_freqs=[50, 60], n_harmonics=2)

    # Initialize filter - notch comb at 50Hz (should attenuate 50Hz and 100Hz in ch0)
    proc = CombFilterTransformer(
        axis="time",
        fundamental_freq=50.0,
        num_harmonics=2,
        q_factor=35.0,
        filter_type="notch",
        coef_type="sos",
        quality_scaling="constant",
    )

    # Process first message
    result1 = proc(msg_in)

    # Check that 50Hz components are attenuated in ch0 but 60Hz in ch1 is less affected
    power1 = _calc_power(result1, fund_freqs=[50, 60], n_harmonics=2)
    assert np.all(
        (power1[0, :, 0] / power0[0, :, 0]) < 0.02
    )  # 50Hz and harmonics should be attenuated
    assert np.all(
        (power1[1, :, 1] / power0[1, :, 1]) > 0.9
    )  # 60Hz and harmonics should be mostly preserved

    # Update settings - change to target 60Hz
    proc.update_settings(fundamental_freq=60.0)

    # Process the same message with new settings
    result2 = proc(msg_in)

    # Calculate power after update
    power2 = _calc_power(result2, fund_freqs=[50, 60], n_harmonics=2)

    # Verify that the filter behavior changed correctly
    assert np.all(
        (power2[0, :, 0] / power0[0, :, 0]) > 0.9
    )  # 50Hz and harmonics should be mostly preserved
    assert np.all(
        (power2[1, :, 1] / power0[1, :, 1]) < 0.02
    )  # 60Hz and harmonics should be attenuated
