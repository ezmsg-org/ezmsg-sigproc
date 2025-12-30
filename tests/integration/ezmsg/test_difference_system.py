"""Integration tests for ezmsg.sigproc.math.difference module."""

import os

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

from ezmsg.sigproc.math.difference import ConstDifference, ConstDifferenceSettings, Difference
from tests.helpers.synth import Counter, CounterSettings, Oscillator, OscillatorSettings
from tests.helpers.util import get_test_fn


def test_difference_two_signals_system(
    fs: float = 100.0,
    n_time: int = 10,
    n_messages: int = 5,
    test_name: str | None = None,
):
    """
    Test that Difference unit correctly subtracts two synchronized signals.

    Uses two Oscillators and verifies the difference matches expected.
    """
    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    # Two oscillators: OSC1 - OSC2
    comps = {
        "OSC1": Oscillator(
            OscillatorSettings(
                n_time=n_time,
                fs=fs,
                n_ch=1,
                dispatch_rate=fs / n_time,
                freq=5.0,
                amp=3.0,  # Larger amplitude
            )
        ),
        "OSC2": Oscillator(
            OscillatorSettings(
                n_time=n_time,
                fs=fs,
                n_ch=1,
                dispatch_rate=fs / n_time,
                freq=5.0,
                amp=1.0,  # Smaller amplitude, same frequency
            )
        ),
        "DIFF": Difference(),
        "LOG": MessageLogger(
            MessageLoggerSettings(
                output=test_filename,
            )
        ),
        "TERM": TerminateOnTotal(
            TerminateOnTotalSettings(
                total=n_messages,
            )
        ),
    }
    conns = (
        (comps["OSC1"].OUTPUT_SIGNAL, comps["DIFF"].INPUT_SIGNAL_A),
        (comps["OSC2"].OUTPUT_SIGNAL, comps["DIFF"].INPUT_SIGNAL_B),
        (comps["DIFF"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    # Collect result
    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    assert len(messages) >= n_messages

    # Verify each message has correct shape
    for msg in messages:
        assert msg.data.shape == (n_time, 1)

    # Reconstruct the full signal and verify it's the difference of two sinusoids
    data = np.concatenate([_.data for _ in messages]).squeeze()
    n_samples = len(data)
    t = np.arange(n_samples) / fs

    # Expected: 3.0 * sin(2*pi*5*t) - 1.0 * sin(2*pi*5*t) = 2.0 * sin(2*pi*5*t)
    expected = 2.0 * np.sin(2 * np.pi * 5.0 * t)
    assert np.allclose(data, expected, atol=1e-10)


def test_const_difference_system(
    fs: float = 100.0,
    n_time: int = 10,
    n_messages: int = 5,
    subtract_value: float = 50.0,
    test_name: str | None = None,
):
    """
    Test that ConstDifference unit correctly subtracts a constant from a signal.
    """
    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    comps = {
        "COUNTER": Counter(
            CounterSettings(
                n_time=n_time,
                fs=fs,
                n_ch=1,
                dispatch_rate=fs / n_time,
            )
        ),
        "DIFF": ConstDifference(ConstDifferenceSettings(value=subtract_value, subtrahend=True)),
        "LOG": MessageLogger(
            MessageLoggerSettings(
                output=test_filename,
            )
        ),
        "TERM": TerminateOnTotal(
            TerminateOnTotalSettings(
                total=n_messages,
            )
        ),
    }
    conns = (
        (comps["COUNTER"].OUTPUT_SIGNAL, comps["DIFF"].INPUT_SIGNAL),
        (comps["DIFF"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    # Collect result
    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    assert len(messages) >= n_messages

    # Verify the constant was subtracted
    data = np.concatenate([_.data for _ in messages]).squeeze()
    n_samples = len(data)

    # Counter produces 0, 1, 2, 3, ... so with subtract_value=50, we expect -50, -49, -48, ...
    expected = np.arange(n_samples) - subtract_value
    assert np.allclose(data, expected)


def test_const_difference_subtrahend_false_system(
    fs: float = 100.0,
    n_time: int = 10,
    n_messages: int = 5,
    value: float = 100.0,
    test_name: str | None = None,
):
    """
    Test ConstDifference with subtrahend=False (value - input).
    """
    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    comps = {
        "COUNTER": Counter(
            CounterSettings(
                n_time=n_time,
                fs=fs,
                n_ch=1,
                dispatch_rate=fs / n_time,
            )
        ),
        "DIFF": ConstDifference(ConstDifferenceSettings(value=value, subtrahend=False)),
        "LOG": MessageLogger(
            MessageLoggerSettings(
                output=test_filename,
            )
        ),
        "TERM": TerminateOnTotal(
            TerminateOnTotalSettings(
                total=n_messages,
            )
        ),
    }
    conns = (
        (comps["COUNTER"].OUTPUT_SIGNAL, comps["DIFF"].INPUT_SIGNAL),
        (comps["DIFF"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    # Collect result
    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    assert len(messages) >= n_messages

    # Verify: value - input
    data = np.concatenate([_.data for _ in messages]).squeeze()
    n_samples = len(data)

    # Counter produces 0, 1, 2, 3, ... so with value=100 and subtrahend=False:
    # result = 100 - counter = 100, 99, 98, ...
    expected = value - np.arange(n_samples)
    assert np.allclose(data, expected)
