import numpy as np
import pytest
import scipy.signal as sps
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.filterbank import FilterbankMode, FilterbankSettings, FilterbankTransformer, MinPhaseMode
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


@pytest.mark.parametrize("mode", [FilterbankMode.CONV, FilterbankMode.FFT, FilterbankMode.AUTO])
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
    proc = FilterbankTransformer(settings=FilterbankSettings(kernels=kernels, mode=mode, axis="time"))

    # Pass the messages
    out_messages = [proc(msg_in) for msg_in in in_messages]
    result = AxisArray.concatenate(*out_messages, dim="time")
    assert result.key == "test_filterbank"

    # Compare to sps.oaconvolve(chirp), with the following differences:
    #  - conv has transients at the beginning that we need to skip over
    #  - oaconvolve assumes the data is finished so it returns the trailing windows,
    #    but filterbank keeps the tail assuming more data is coming.
    expected = np.stack([sps.oaconvolve(chirp, _[None, :], axes=1) for _ in kernels], axis=1)
    idx0 = ntaps if mode in [FilterbankMode.CONV, FilterbankMode.AUTO] else 0
    assert np.allclose(result.data[..., idx0:], expected[..., idx0 : result.data.shape[-1]])

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


def test_template_axes_cached_and_owned():
    """Filterbank reuses processor-owned non-target axes across messages."""
    fs = 100.0
    n_time = 20
    shared_ch_axis = AxisArray.CoordinateAxis(data=np.array(["Ch0", "Ch1"]), dims=["ch"])
    messages = [
        AxisArray(
            data=np.arange(2 * n_time, dtype=float).reshape(2, n_time) + i,
            dims=["ch", "time"],
            axes={
                "ch": shared_ch_axis,
                "time": AxisArray.TimeAxis(fs=fs, offset=i * n_time / fs),
            },
            key="test_filterbank_axes",
        )
        for i in range(3)
    ]
    assert all(message.axes["ch"] is shared_ch_axis for message in messages)

    kernels = [np.array([0.5, 0.5]), np.array([1.0, -1.0])]
    proc = FilterbankTransformer(settings=FilterbankSettings(kernels=kernels, mode=FilterbankMode.CONV, axis="time"))
    outputs = [proc(message) for message in messages]

    assert all(output.axes["ch"] is outputs[0].axes["ch"] for output in outputs)
    assert all(output.axes["ch"] is not message.axes["ch"] for output, message in zip(outputs, messages))


@pytest.mark.parametrize("mode", [FilterbankMode.CONV, FilterbankMode.FFT])
def test_filterbank_complex_min_phase_chunk_invariant(mode):
    """Complex (analytic) kernels can be min-phased; one buffer and many small chunks agree."""
    fs = 500.0
    n = np.arange(101) - 50
    lp = sps.firwin(101, 10.0, fs=fs)
    kernels = [lp * np.exp(2j * np.pi * fc * n / fs) for fc in (30.0, 60.0, 120.0)]
    data = np.random.default_rng(0).standard_normal((2000, 3))

    def run(step):
        proc = FilterbankTransformer(
            FilterbankSettings(kernels=kernels, mode=mode, min_phase=MinPhaseMode.HOMOMORPHICFULL, axis="time")
        )
        outs = [
            proc(
                AxisArray(
                    data=data[i : i + step],
                    dims=["time", "ch"],
                    axes={"time": AxisArray.TimeAxis(fs=fs, offset=i / fs)},
                    key="test_fb_minphase",
                )
            )
            for i in range(0, data.shape[0], step)
        ]
        return AxisArray.concatenate(*outs, dim="time").data

    one_shot = run(data.shape[0])
    assert np.iscomplexobj(one_shot) and one_shot.shape[-1] > 0
    np.testing.assert_allclose(run(37), one_shot, rtol=1e-10, atol=1e-12)
