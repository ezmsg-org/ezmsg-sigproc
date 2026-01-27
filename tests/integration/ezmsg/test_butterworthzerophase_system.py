import tempfile
from pathlib import Path

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal

from ezmsg.sigproc.butterworthzerophase import (
    ButterworthBackwardFilterTransformer,
    ButterworthZeroPhase,
    ButterworthZeroPhaseSettings,
)
from tests.helpers.synth import EEGSynth


def test_butterworth_zero_phase_system():
    fs = 1000.0
    n_time = 50
    n_total = 10
    n_channels = 96
    order = 4
    cuton = 30.0
    cutoff = 45.0
    coef_type = "sos"

    # Compute expected pad_length for this filter configuration
    settings = ButterworthZeroPhaseSettings(order=order, cuton=cuton, cutoff=cutoff, coef_type=coef_type)
    backward = ButterworthBackwardFilterTransformer(settings)
    pad_length = backward._compute_pad_length(fs)

    test_filename = Path(tempfile.gettempdir())
    test_filename = test_filename / Path("test_butterworth_zero_phase_system.txt")
    with open(test_filename, "w"):
        pass
    ez.logger.info(f"Logging to {test_filename}")

    comps = {
        "SRC": EEGSynth(n_time=n_time, fs=fs, n_ch=n_channels, alpha_freq=10.0),
        "BUTTER": ButterworthZeroPhase(
            order=order,
            cuton=cuton,
            cutoff=cutoff,
            coef_type=coef_type,
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

    total_output_samples = 0
    for msg in messages:
        assert isinstance(msg, AxisArray)
        # Non-time dimensions must always be preserved, even during warmup
        assert msg.data.shape[1] == n_channels
        # Time dimension may be 0 during warmup while buffering for backward pass
        assert msg.data.shape[0] >= 0
        # Data should be finite (for non-empty messages)
        if msg.data.size > 0:
            assert np.isfinite(msg.data).all()
        total_output_samples += msg.data.shape[0]

    # Total output should be input samples minus pad_length delay
    total_input_samples = n_total * n_time
    expected_output = total_input_samples - pad_length
    # Allow some tolerance since we may have received slightly more than n_total messages
    assert total_output_samples >= expected_output
