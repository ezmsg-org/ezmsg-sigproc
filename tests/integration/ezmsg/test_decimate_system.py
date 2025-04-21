import pytest
import numpy as np
import scipy.signal
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.terminate import TerminateOnTotal

from ezmsg.sigproc.decimate import Decimate
from ezmsg.sigproc.synth import EEGSynth

from tests.helpers.util import get_test_fn


@pytest.mark.parametrize("target_rate", [100.0, 500.0])
def test_decimate_system(target_rate: float):
    test_filename = get_test_fn()
    test_filename_raw = test_filename.parent / (
        test_filename.stem + "raw" + test_filename.suffix
    )

    fs = 500.0
    n_ch = 8
    n_time = 100
    n_total = int(fs / n_time)  # 1 second of messages

    comps = {
        "SRC": EEGSynth(n_time=n_time, fs=fs, n_ch=n_ch, alpha_freq=10.5),
        "DECIMATE": Decimate(axis="time", target_rate=target_rate),
        "LOGRAW": MessageLogger(output=test_filename_raw),
        "LOGFILT": MessageLogger(output=test_filename),
        "TERM": TerminateOnTotal(n_total),
    }
    conns = (
        (comps["SRC"].OUTPUT_SIGNAL, comps["DECIMATE"].INPUT_SIGNAL),
        (comps["DECIMATE"].OUTPUT_SIGNAL, comps["LOGFILT"].INPUT_MESSAGE),
        (comps["SRC"].OUTPUT_SIGNAL, comps["LOGRAW"].INPUT_MESSAGE),
        (comps["LOGFILT"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)
    # Unfortunately, we can't test the factor < 1 error because MessageLogger raises its own 0-msg error.

    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    assert len(messages) >= n_total
    inputs = [_ for _ in message_log(test_filename_raw)]
    inputs = AxisArray.concatenate(*inputs, dim="time")
    outputs = AxisArray.concatenate(*messages, dim="time")

    expected_factor: int = int(fs // target_rate)
    if expected_factor == 1:
        expected = inputs.data
    else:
        b, a = scipy.signal.cheby1(8, 0.05, 0.8 / expected_factor)
        b, a = scipy.signal.normalize(b, a)
        zi = scipy.signal.lfilter_zi(b, a)[:, None]
        antialiased, _ = scipy.signal.lfilter(b, a, inputs.data, axis=0, zi=zi)
        expected = antialiased[:: int(expected_factor)]

    assert np.allclose(outputs.data, expected)
