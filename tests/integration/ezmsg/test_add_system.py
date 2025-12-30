"""Integration tests for ezmsg.sigproc.math.add module."""

import os

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

from ezmsg.sigproc.math.add import Add, ConstAdd, ConstAddSettings
from tests.helpers.synth import Counter, CounterSettings, Oscillator, OscillatorSettings
from tests.helpers.util import get_test_fn


def test_add_two_signals_system(
    fs: float = 100.0,
    n_time: int = 10,
    n_messages: int = 5,
    test_name: str | None = None,
):
    """
    Test that Add unit correctly adds two synchronized signals.

    Uses two Counter units with different starting values to verify
    element-wise addition.
    """
    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    # We'll use two oscillators with different frequencies
    # and verify the sum matches expected
    comps = {
        "OSC1": Oscillator(
            OscillatorSettings(
                n_time=n_time,
                fs=fs,
                n_ch=1,
                dispatch_rate=fs / n_time,
                freq=5.0,
                amp=1.0,
            )
        ),
        "OSC2": Oscillator(
            OscillatorSettings(
                n_time=n_time,
                fs=fs,
                n_ch=1,
                dispatch_rate=fs / n_time,
                freq=10.0,
                amp=2.0,
            )
        ),
        "ADD": Add(),
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
        (comps["OSC1"].OUTPUT_SIGNAL, comps["ADD"].INPUT_SIGNAL_A),
        (comps["OSC2"].OUTPUT_SIGNAL, comps["ADD"].INPUT_SIGNAL_B),
        (comps["ADD"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
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

    # Reconstruct the full signal and verify it's the sum of two sinusoids
    data = np.concatenate([_.data for _ in messages]).squeeze()
    n_samples = len(data)
    t = np.arange(n_samples) / fs

    # Expected: 1.0 * sin(2*pi*5*t) + 2.0 * sin(2*pi*10*t)
    expected = 1.0 * np.sin(2 * np.pi * 5.0 * t) + 2.0 * np.sin(2 * np.pi * 10.0 * t)
    assert np.allclose(data, expected, atol=1e-10)


def test_const_add_system(
    fs: float = 100.0,
    n_time: int = 10,
    n_messages: int = 5,
    add_value: float = 100.0,
    test_name: str | None = None,
):
    """
    Test that ConstAdd unit correctly adds a constant to a signal.
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
        "ADD": ConstAdd(ConstAddSettings(value=add_value)),
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
        (comps["COUNTER"].OUTPUT_SIGNAL, comps["ADD"].INPUT_SIGNAL),
        (comps["ADD"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    # Collect result
    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    assert len(messages) >= n_messages

    # Verify the constant was added
    data = np.concatenate([_.data for _ in messages]).squeeze()
    n_samples = len(data)

    # Counter produces 0, 1, 2, 3, ... so with add_value=100, we expect 100, 101, 102, ...
    expected = np.arange(n_samples) + add_value
    assert np.allclose(data, expected)
