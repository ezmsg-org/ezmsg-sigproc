import os

import numpy as np
from frozendict import frozendict
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotalSettings, TerminateOnTotal
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messagecodec import message_log

from ezmsg.sigproc.scaler import scaler_np
from ezmsg.sigproc.scaler import AdaptiveStandardScalerSettings, AdaptiveStandardScaler
from ezmsg.sigproc.synth import Counter, CounterSettings

from tests.helpers.util import get_test_fn


def test_scaler_system(
    tau: float = 1.0,
    fs: float = 10.0,
    duration: float = 2.0,
    test_name: str | None = None,
):
    """
    For this test, we assume that Counter and scaler_np are functioning properly.
    The purpose of this test is exclusively to test that the AdaptiveStandardScaler and AdaptiveStandardScalerSettings
    generated classes are wrapping scaler_np and exposing its parameters.
    This test passing should only be considered a success if test_scaler_np also passed.
    """
    block_size: int = 4
    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    comps = {
        "COUNTER": Counter(
            CounterSettings(
                n_time=block_size,
                fs=fs,
                n_ch=1,
                dispatch_rate=duration,  # Simulation duration in 1.0 seconds
                mod=None,
            )
        ),
        "SCALER": AdaptiveStandardScaler(
            AdaptiveStandardScalerSettings(time_constant=tau, axis="time")
        ),
        "LOG": MessageLogger(
            MessageLoggerSettings(
                output=test_filename,
            )
        ),
        "TERM": TerminateOnTotal(
            TerminateOnTotalSettings(
                total=int(duration * fs / block_size),
            )
        ),
    }
    conns = (
        (comps["COUNTER"].OUTPUT_SIGNAL, comps["SCALER"].INPUT_SIGNAL),
        (comps["SCALER"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    # Collect result
    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    data = np.concatenate([_.data for _ in messages]).squeeze()

    expected_input = AxisArray(
        np.arange(len(data))[None, :],
        dims=["ch", "time"],
        axes=frozendict({"time": AxisArray.TimeAxis(fs=fs)}),
    )
    _scaler = scaler_np(time_constant=tau, axis="time")
    expected_output = _scaler.send(expected_input)
    assert np.allclose(expected_output.data.squeeze(), data)
