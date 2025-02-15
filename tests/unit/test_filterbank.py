import pytest
import numpy as np
import scipy.signal as sps
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.filterbank import filterbank, FilterbankMode

from tests.helpers.util import gaussian, make_chirp


def gen_signal(fs, dur):
    # generate signal
    f_gains = [9, 5]  # frequency will scale as square of this value * time.
    t_offsets = [0.3 * dur, 0.1 * dur]  # When the chirp starts.
    time = np.arange(int(dur * fs)) / fs
    # TODO: Replace with sps.chirp?
    chirp1, frequency1 = make_chirp(time, t_offsets[0], f_gains[0])
    chirp2, frequency2 = make_chirp(time, t_offsets[1], f_gains[1])
    chirp = chirp1 + 0.6 * chirp2
    chirp *= gaussian(time, 0.5, 0.2)
    return chirp, time


def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
    atten = sps.kaiser_atten(ntaps, width / (0.5 * fs))
    beta = sps.kaiser_beta(atten)
    taps = sps.firwin(
        ntaps,
        [lowcut, highcut],
        fs=fs,
        pass_zero="bandpass",
        window=("kaiser", beta),
        scale=False,
    )
    return taps


@pytest.mark.parametrize(
    "mode", [FilterbankMode.CONV, FilterbankMode.FFT, FilterbankMode.AUTO]
)
@pytest.mark.parametrize("kernel_type", ["kaiser", "brickwall"])
def test_filterbank(mode: str, kernel_type: str):
    # Generate test signal
    fs = 1000
    dur = 5.0
    chirp, tvec = gen_signal(fs, dur)
    chirp = np.vstack((chirp, np.roll(chirp, fs)))

    # Split signal into messages
    step_size = 100
    in_messages = []
    for idx in range(0, len(tvec), step_size):
        in_messages.append(
            AxisArray(
                data=chirp[:, idx : idx + step_size],
                dims=["ch", "time"],
                axes={"time": AxisArray.TimeAxis(offset=tvec[idx], fs=fs)},
                key="test_filterbank",
            )
        )

    # Get kernels for 2 bandpass filters. 3-30 and 30-100 Hz.
    kernels = []
    bands = [(3, 10), (10, 30), (30, 50), (50, 70), (70, 100), (100, 300)]
    if kernel_type == "kaiser":
        # Decent FIR filter for a given number of taps
        ntaps = 101
        for band in bands:
            k = bandpass_kaiser(ntaps, band[0], band[1], fs, 2)
            kernels.append(k)
    elif kernel_type == "brickwall":
        # brickwall filter. Not very useful. Lots of transients. TODO: Use this to implement sparsity in the filterbank.
        ntaps = fs // 2
        fvec = np.arange(0, ntaps, 1)
        for band in bands:
            fft_kernel = np.zeros(ntaps)
            fft_kernel[np.logical_and(fvec >= band[0], fvec <= band[1])] = 1
            k = np.fft.irfft(fft_kernel, n=ntaps)
            kernels.append(k)

    # Prep filterbank
    gen = filterbank(kernels=kernels, mode=mode, axis="time")

    # Pass the messages
    out_messages = [gen.send(msg_in) for msg_in in in_messages]
    result = AxisArray.concatenate(*out_messages, dim="time")
    assert result.key == "test_filterbank"

    # Compare to sps.oaconvolve(chirp), with the following differences:
    #  - conv has transients at the beginning that we need to skip over
    #  - oaconvolve assumes the data is finished so it returns the trailing windows,
    #    but filterbank keeps the tail assuming more data is coming.
    expected = np.stack(
        [sps.oaconvolve(chirp, _[None, :], axes=1) for _ in kernels], axis=1
    )
    idx0 = ntaps if mode in [FilterbankMode.CONV, FilterbankMode.AUTO] else 0
    assert np.allclose(
        result.data[..., idx0:], expected[..., idx0 : result.data.shape[-1]]
    )

    if False:
        # Debug visualize result
        import matplotlib.pyplot as plt

        # tmp = result.data
        tmp = expected
        nch = tmp.shape[0]
        fig, axes = plt.subplots(3, nch)
        for ch_ix in range(nch):
            axes[0, ch_ix].plot(tvec, chirp[ch_ix])
            axes[1, ch_ix].imshow(tmp[ch_ix], aspect="auto", origin="lower")
            axes[2, ch_ix].imshow(np.abs(tmp[ch_ix]), aspect="auto", origin="lower")
        plt.show()
