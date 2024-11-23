import copy
import os
import importlib.util

import numpy as np
from ezmsg.util.messages.chunker import array_chunker
from frozendict import frozendict
import pytest
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotalSettings, TerminateOnTotal
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messagecodec import message_log

from ezmsg.sigproc.scaler import scaler, scaler_np, EWMA, ewma_step
from ezmsg.sigproc.scaler import AdaptiveStandardScalerSettings, AdaptiveStandardScaler
from ezmsg.sigproc.synth import Counter, CounterSettings

from util import get_test_fn, assert_messages_equal


def test_ewma():
    alpha = 0.6
    n_times = 100
    n_ch = 32
    n_feat = 4
    data = np.arange(1, n_times * n_ch * n_feat + 1, dtype=float).reshape(
        n_times, n_ch, n_feat
    )

    # Expected
    expected = [data[0]]
    for ix, dat in enumerate(data):
        expected.append(ewma_step(dat, expected[-1], alpha))
    expected = np.stack(expected)[1:]

    ewma = EWMA(alpha=alpha)
    res = ewma.compute(data)
    assert np.allclose(res, expected)


@pytest.fixture
def fixture_arrays():
    # Test data values taken from river:
    # https://github.com/online-ml/river/blob/main/river/preprocessing/scale.py#L511-L536C17
    data = np.array([5.278, 5.050, 6.550, 7.446, 9.472, 10.353, 11.784, 11.173])
    expected_result = np.array([0.0, -0.816, 0.812, 0.695, 0.754, 0.598, 0.651, 0.124])
    return data, expected_result


@pytest.mark.skipif(
    importlib.util.find_spec("river") is None, reason="requires `river` package"
)
def test_adaptive_standard_scaler_river(fixture_arrays):
    data, expected_result = fixture_arrays

    test_input = AxisArray(
        np.tile(data, (2, 1)),
        dims=["ch", "time"],
        axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
    )

    backup = [copy.deepcopy(test_input)]

    # The River example used alpha = 0.6
    # tau = -gain / np.log(1 - alpha) and here we're using gain = 0.01
    tau = 0.010913566679372915
    _scaler = scaler(time_constant=tau, axis="time")
    output = _scaler.send(test_input)
    assert np.allclose(output.data[0], expected_result, atol=1e-3)
    assert_messages_equal([test_input], backup)


def test_scaler_np(fixture_arrays):
    data, expected_result = fixture_arrays
    chunker = array_chunker(data, 4, fs=100.0)
    test_input = list(chunker)
    backup = copy.deepcopy(test_input)

    tau = 0.010913566679372915
    gen = scaler_np(time_constant=tau, axis="time")
    outputs = []
    for chunk in test_input:
        outputs.append(gen.send(chunk))
    output = AxisArray.concatenate(*outputs, dim="time")
    assert np.allclose(output.data, expected_result, atol=1e-3)
    assert_messages_equal(test_input, backup)


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
