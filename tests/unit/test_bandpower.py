import copy

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.bandpower import bandpower, SpectrogramSettings, AggregationFunction

from tests.helpers.util import (
    create_messages_with_periodic_signal,
    assert_messages_equal,
)


def _debug_plot(result):
    import matplotlib.pyplot as plt

    t_vec = result.axes["time"].value(np.arange(result.data.shape[0]))
    plt.plot(t_vec, result.data[..., 0])


def test_bandpower():
    win_dur = 1.0
    fs = 1000.0
    bands = [(9, 11), (70, 90), (134, 136)]

    sin_params = [
        {"f": 10.0, "a": 3.0, "dur": 4.0, "offset": 1.0},
        {"f": 10.0, "a": 1.0, "dur": 3.0, "offset": 5.0},
        {"f": 135.0, "a": 4.0, "dur": 4.0, "offset": 1.0},
        {"f": 135.0, "a": 2.0, "dur": 3.0, "offset": 5.0},
    ]
    messages = create_messages_with_periodic_signal(
        sin_params=sin_params,
        fs=fs,
        msg_dur=0.4,
        win_step_dur=None,  # The spectrogram will do the windowing
    )

    # Grab a deepcopy backup of the inputs, so we can check the inputs didn't change
    #  while being processed.
    backup = [copy.deepcopy(_) for _ in messages]

    gen = bandpower(
        spectrogram_settings=SpectrogramSettings(
            window_dur=win_dur,
            window_shift=0.1,
        ),
        bands=bands,
        aggregation=AggregationFunction.MEAN,
    )
    results = [gen.send(_) for _ in messages]

    assert_messages_equal(messages, backup)

    result = AxisArray.concatenate(*results, dim="time")
    # _debug_plot(result)

    # Check the amplitudes at the midpoints of each of our sinusoids.
    t_vec = result.axes["time"].value(np.arange(result.data.shape[0]))
    mags = []
    for s_p in sin_params[:2]:
        ix = np.argmin(np.abs(t_vec - (s_p["offset"] + s_p["dur"] / 2)))
        mags.append(result.data[ix, 0, 0])
    for s_p in sin_params[2:]:
        ix = np.argmin(np.abs(t_vec - (s_p["offset"] + s_p["dur"] / 2)))
        mags.append(result.data[ix, 2, 0])
    # The sorting of the measured magnitudes should match the sorting of the parameter magnitudes.
    assert np.array_equal(np.argsort(mags), np.argsort([_["a"] for _ in sin_params]))
