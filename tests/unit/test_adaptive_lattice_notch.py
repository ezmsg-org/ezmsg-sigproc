import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.adaptive_lattice_notch import (
    AdaptiveLatticeNotchFilterTransformer,
    AdaptiveLatticeNotchFilterSettings,
)
from tests.helpers.util import (
    create_messages_with_periodic_signal,
    assert_messages_equal,
)


def debug_plot():
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


def test_adaptive_lattice_notch_transformer():
    # Generate synthetic PPG signal
    fs = 50.0  # Sampling frequency
    dur = 60.0  # Duration in seconds
    sin_params = [
        {"f": 0.25, "a": 2.0, "dur": dur, "offset": 0.0, "p": np.pi/2},  # Resp
        {"f": 2.5, "a": 10.0, "dur": dur, "offset": 0.0},
        {"f": 5.0, "a": 5.0, "dur": dur, "offset": 0.0},
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
    alnf = AdaptiveLatticeNotchFilterTransformer()
    result = []
    for msg in messages:
        result.append(alnf(msg))

    concat = AxisArray.concatenate(*result, dim="time")
    print(concat)
