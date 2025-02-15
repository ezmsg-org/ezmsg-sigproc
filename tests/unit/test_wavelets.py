import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
import pywt

from ezmsg.sigproc.wavelets import cwt, MinPhaseMode

from tests.helpers.util import gaussian, make_chirp


def scratch():
    scales = np.geomspace(4, 256, num=35)
    wavelets = [
        f"cmor{x:.1f}-{y:.1f}" for x in [0.5, 1.5, 2.5] for y in [0.5, 1.0, 1.5]
    ]
    wavelet = wavelets[1]

    # Generate test signal
    fs = 1000
    dur = 2.0
    tvec = np.arange(int(dur * fs)) / fs
    chirp1, frequency1 = make_chirp(tvec, 0.2, 9)
    chirp2, frequency2 = make_chirp(tvec, 0.1, 5)
    chirp = chirp1 + 0.6 * chirp2
    chirp *= gaussian(tvec, 0.5, 0.2)
    chirp = np.vstack((chirp, np.roll(chirp, fs)))
    # TODO: Replace with sps.chirp?

    precision = 10
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
    int_psi = np.conj(int_psi)
    int_psi = np.asarray(int_psi, dtype="complex")
    x = np.asarray(x, dtype=chirp.real.dtype)

    wave_range = x[-1] - x[0]
    step = x[1] - x[0]
    int_psi_scales = []
    for scale in scales:
        reix = (np.arange(scale * wave_range + 1) / (scale * step)).astype(int)
        if reix[-1] >= int_psi.size:
            reix = np.extract(reix < int_psi.size, reix)
        int_psi_scales.append(int_psi[reix][::-1])

    # Try different kinds of convolutions.
    # np.convolve: 1d, 1d only
    conv = np.convolve(chirp[0], int_psi_scales[-1])

    import matplotlib.pyplot as plt

    plt.plot(chirp[0])
    plt.plot(int_psi_scales[-1])
    plt.plot(conv)
    plt.show()
    print("DONE")


def test_cwt():
    frequencies = np.geomspace(4, 200, num=50)
    # wavelets = [f"cmor{x:.1f}-{y:.1f}" for x in [0.5, 1.5, 2.5] for y in [0.5, 1.0, 1.5]]
    # wavelet = wavelets[4]
    wavelet = "morl"

    # Generate test signal
    fs = 1000
    dur = 2.0
    tvec = np.arange(int(dur * fs)) / fs
    chirp1, frequency1 = make_chirp(tvec, 0.2, 9)
    chirp2, frequency2 = make_chirp(tvec, 0.1, 5)
    chirp = chirp1 + 0.6 * chirp2
    chirp *= gaussian(tvec, 0.5, 0.2)
    chirp = np.vstack((chirp, np.roll(chirp, fs)))
    # TODO: Replace with sps.chirp?
    # scales = np.geomspace(4, 256, num=70)[::-1]
    scales = pywt.frequency2scale(wavelet, frequencies / fs, precision=10)

    # Split signal into messages
    step_size = 100
    in_messages = []
    for idx in range(0, len(tvec), step_size):
        in_messages.append(
            AxisArray(
                data=chirp[:, idx : idx + step_size],
                dims=["ch", "time"],
                axes={
                    "ch": AxisArray.CoordinateAxis(
                        data=np.array([f"Ch{_}" for _ in range(2)]), dims=["ch"]
                    ),
                    "time": AxisArray.TimeAxis(offset=tvec[idx], fs=fs),
                },
                key="test_cwt",
            )
        )

    # Prepare expected output from pywt.cwt
    expected, freqs = pywt.cwt(chirp, scales, wavelet, 1 / fs, method="conv", axis=-1)
    # Swap scales and channels -> ch, freqs, time
    expected = np.swapaxes(expected, 0, 1)

    # Prep filterbank
    gen = cwt(
        frequencies=frequencies,
        wavelet=wavelet,
        min_phase=MinPhaseMode.HOMOMORPHIC,
        axis="time",
    )

    # Pass the messages
    out_messages = [gen.send(in_messages[0])]
    out_messages += [gen.send(msg_in) for msg_in in in_messages[1:]]
    result = AxisArray.concatenate(*out_messages, dim="time")
    assert result.key == "test_cwt"

    # Compare result to expected
    ez_freqs = result.axes["freq"].data
    assert np.allclose(ez_freqs, freqs)
    # It is impossible to check the data result because pywt.cwt uses filtfilt and different coeffs (not minphase).
    # Use the below snippet to visualize the result and compare to expected.

    """
    # Debug visualize result
    import matplotlib.pyplot as plt

    tmp = result.data
    title = "ezmsg minphase homomorphic"
    # tmp = expected
    # title = "pywavelets"
    nch = tmp.shape[0]
    fig, axes = plt.subplots(3, nch, figsize=(8, 8), tight_layout=True)
    for ch_ix in range(nch):
        axes[0, ch_ix].set_title(f"Channel {ch_ix}")
        axes[0, ch_ix].plot(tvec, chirp[ch_ix])

        axes[1, ch_ix].set_title(title)
        _ = axes[1, ch_ix].pcolormesh(
            tvec[: tmp.shape[-1]], freqs, np.abs(tmp[ch_ix, :-1, :-1])
        )
        axes[1, ch_ix].set_yscale("log")
        axes[1, ch_ix].set_xlabel("Time (s)")
        axes[1, ch_ix].set_ylabel("Frequency (Hz)")

        axes[2, ch_ix].set_title("pywt.cwt filtfilt")
        _ = axes[2, ch_ix].pcolormesh(
            tvec[: expected.shape[-1]], freqs, np.abs(expected[ch_ix, :-1, :-1])
        )
        axes[2, ch_ix].set_yscale("log")
    plt.show()
    """
