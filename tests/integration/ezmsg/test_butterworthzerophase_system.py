import tempfile
from pathlib import Path

import ezmsg.core as ez
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal

from ezmsg.sigproc.butterworthzerophase import ButterworthZeroPhase
from tests.helpers.synth import EEGSynth


def test_butterworth_zero_phase_system():
    fs = 1000.0
    n_time = 50
    n_total = 10
    n_channels = 96
    test_filename = Path(tempfile.gettempdir())
    test_filename = test_filename / Path("test_butterworth_zero_phase_system.txt")
    with open(test_filename, "w"):
        pass
    ez.logger.info(f"Logging to {test_filename}")

    comps = {
        "SRC": EEGSynth(n_time=n_time, fs=fs, n_ch=n_channels, alpha_freq=10.0),
        "BUTTER": ButterworthZeroPhase(
            order=4,
            cuton=30.0,
            cutoff=45.0,
            coef_type="sos",
        ),
        "LOG": MessageLogger(output=test_filename),
        "TERM": TerminateOnTotal(total=n_total),
    }

    conns = (
        (comps["SRC"].OUTPUT_SIGNAL, comps["BUTTER"].INPUT_SIGNAL),
        (comps["BUTTER"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )

    ez.run(components=comps, connections=conns)

    messages = list(message_log(test_filename))
    assert len(messages) >= n_total

    for msg in messages:
        assert isinstance(msg, AxisArray)
        assert msg.data.shape[1] == n_channels
        assert msg.data.shape[0] == n_time
