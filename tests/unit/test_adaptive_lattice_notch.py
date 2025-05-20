import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, replace

from ezmsg.sigproc.adaptive_lattice_notch import (
    AdaptiveLatticeNotchFilterTransformer,
)
from tests.helpers.util import (
    create_messages_with_periodic_signal,
)


class ReferenceALNF:
    def __init__(
        self, gamma: float = 0.9, mu: float = 0.95, eta: float = 0.90, n_chans: int = 3
    ):
        self.gamma = gamma  # Pole-zero contraction factor
        self.mu = mu  # Smoothing factor
        self.eta = eta  # Forgetting factor
        self.k1 = [-0.99] * n_chans
        # State variables
        self.s_n_1 = [0] * n_chans
        self.s_n_2 = [0] * n_chans
        self.p = [0] * n_chans
        self.q = [0] * n_chans

    def update(self, msg: AxisArray):
        """Receive new samples and perform adaptive notch filtering"""
        fs = 1 / msg.axes["time"].gain
        results = []
        for sample in msg.data:
            sample_res = []
            for ch_ix, x_n in enumerate(sample):
                s_n = (
                    x_n
                    - self.k1[ch_ix] * (1 + self.gamma) * self.s_n_1[ch_ix]
                    - self.gamma * self.s_n_2[ch_ix]
                )
                # y_n = s_n + 2 * self.k1[ch_ix] * self.s_n_1[ch_ix] + self.s_n_2[ch_ix]

                self.p[ch_ix] = self.eta * self.p[ch_ix] + (1 - self.eta) * (
                    self.s_n_1[ch_ix] * (s_n + self.s_n_2[ch_ix])
                )
                self.q[ch_ix] = self.eta * self.q[ch_ix] + (1 - self.eta) * (
                    2 * (self.s_n_1[ch_ix] * self.s_n_1[ch_ix])
                )

                k1_n_1 = self.k1[ch_ix]
                self.k1[ch_ix] = -self.p[ch_ix] / (
                    self.q[ch_ix] + 1e-8
                )  # Avoid division by zero
                self.k1[ch_ix] = max(
                    -1, min(1, self.k1[ch_ix])
                )  # Clip to prevent instability

                self.k1[ch_ix] = self.mu * k1_n_1 + (1 - self.mu) * self.k1[ch_ix]

                omega_n = np.arccos(
                    -self.k1[ch_ix]
                )  # Compute omega_n using equation 13 from the paper

                self.s_n_2[ch_ix] = self.s_n_1[ch_ix]
                self.s_n_1[ch_ix] = s_n

                sample_res.append((omega_n * fs) / (2 * np.pi))
            results.append(sample_res)

        return replace(msg, data=np.array(results))


def debug_plot(t, ppg, hrs, hr_freq, rrs):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, ppg)
    plt.title("Synthetic PPG Signal")
    plt.xlabel("Time (s)")

    plt.subplot(3, 1, 2)
    plt.plot(t, hrs)
    plt.axhline(y=hr_freq, color="r", linestyle="--", label="True HR")
    plt.title("Estimated Heart Rate")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    # plt.ylim(0.5, 2.5)

    plt.subplot(3, 1, 3)
    plt.plot(t, rrs)
    plt.axhline(y=0.25, color="r", linestyle="--", label="True RR")
    plt.title("Estimated Respiratory Rate")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    # plt.ylim(0.1, 0.5)

    plt.tight_layout()
    plt.show()


def debug_alnf_plot(msgs):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    concat = AxisArray.concatenate(*msgs, dim="time")
    fs = 1 / concat.axes["time"].gain
    y = np.hstack([msg.axes["freq"].data[:, 0] for msg in msgs])
    tvec = concat.axes["time"].value(np.arange(concat.shape[0]))

    sgram, freqs, t, im = plt.specgram(concat.data[:, 0], NFFT=512, Fs=fs, noverlap=128)
    plt.plot(tvec, y, "r--", label="Notch Frequency")
    plt.show()


def test_adaptive_lattice_notch_transformer():
    # Generate synthetic PPG signal
    fs = 50.0  # Sampling frequency
    dur = 60.0  # Duration in seconds
    sin_params = [
        {"f": 0.25, "a": 2.0, "dur": dur, "offset": 0.0, "p": np.pi / 2},  # Resp
        {"f": 1.0, "a": 10.0, "dur": dur / 3, "offset": 0.0},
        {"f": 2.0, "a": 10.0, "dur": dur / 3, "offset": dur / 3},
        {"f": 1.0, "a": 10.0, "dur": dur / 3, "offset": 2 * dur / 3},
        {"f": 7.5, "a": 2.0, "dur": dur, "offset": 0.0},
    ]
    messages = create_messages_with_periodic_signal(
        sin_params=sin_params,
        fs=fs,
        msg_dur=0.4,
        win_step_dur=None,
        n_ch=3,
    )

    # Process signal
    alnf = AdaptiveLatticeNotchFilterTransformer(
        gamma=0.9,
        mu=0.95,  # Smoothing factor
        eta=0.90,  # Forgetting factor
        init_notch_freq=1.1263353411103032,
        chunkwise=False,
    )
    result = [alnf(msg) for msg in messages]
    result_f = np.vstack([_.axes["freq"].data for _ in result])

    ref_proc = ReferenceALNF(
        gamma=0.9,
        mu=0.95,
        eta=0.90,
    )
    ref_result = [ref_proc.update(msg) for msg in messages]
    ref_concat = AxisArray.concatenate(*ref_result, dim="time")

    assert np.allclose(ref_concat.data, result_f, atol=1e-9)

    # debug_alnf_plot(result)
