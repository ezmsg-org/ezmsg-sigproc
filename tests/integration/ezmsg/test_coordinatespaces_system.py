"""Integration tests for ezmsg.sigproc.coordinatespaces module."""

import os
import typing

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

from ezmsg.sigproc.coordinatespaces import (
    CoordinateMode,
    CoordinateSpaces,
    CoordinateSpacesSettings,
    cart2pol,
)
from tests.helpers.util import get_test_fn


class CartesianSignalSettings(ez.Settings):
    """Settings for CartesianSignal generator."""

    n_time: int = 100
    fs: float = 100.0


class CartesianSignal(ez.Unit):
    """
    Generates (x, y) coordinate signals for testing.

    Produces a spiral pattern: x = r*cos(theta), y = r*sin(theta)
    where r increases linearly with time and theta increases with time.
    """

    SETTINGS = CartesianSignalSettings
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self._n_sent = 0

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        while True:
            # Generate time indices
            t = np.arange(self._n_sent, self._n_sent + self.SETTINGS.n_time) / self.SETTINGS.fs

            # Create spiral pattern
            theta = 2 * np.pi * t  # One rotation per second
            r = 0.1 + 0.1 * t  # Radius grows with time

            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # Stack as (time, ch) with ch being [x, y]
            data = np.stack([x, y], axis=-1)

            offset = self._n_sent / self.SETTINGS.fs
            result = AxisArray(
                data=data,
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.TimeAxis(fs=self.SETTINGS.fs, offset=offset),
                    "ch": AxisArray.CoordinateAxis(
                        data=np.array(["x", "y"]),
                        dims=["ch"],
                    ),
                },
            )

            self._n_sent += self.SETTINGS.n_time

            yield self.OUTPUT_SIGNAL, result


@pytest.mark.parametrize("mode", [CoordinateMode.CART2POL, CoordinateMode.POL2CART])
def test_coordinatespaces_system(mode: CoordinateMode, test_name: str | None = None):
    """Test CoordinateSpaces unit in an ezmsg system."""
    fs = 100.0
    n_time = 100
    n_messages = 10

    test_filename = get_test_fn(test_name)
    ez.logger.info(f"Test file: {test_filename}")

    comps = {
        "SIGNAL": CartesianSignal(CartesianSignalSettings(n_time=n_time, fs=fs)),
        "COORD": CoordinateSpaces(CoordinateSpacesSettings(mode=mode, axis="ch")),
        "LOG": MessageLogger(MessageLoggerSettings(output=test_filename)),
        "TERM": TerminateOnTotal(TerminateOnTotalSettings(total=n_messages)),
    }
    conns = (
        (comps["SIGNAL"].OUTPUT_SIGNAL, comps["COORD"].INPUT_SIGNAL),
        (comps["COORD"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    # Read logged messages
    messages: list[AxisArray] = list(message_log(test_filename))
    os.remove(test_filename)

    ez.logger.info(f"Received {len(messages)} messages")
    assert len(messages) >= n_messages

    # Verify output shape and structure
    for msg in messages:
        assert msg.data.shape == (n_time, 2), "Output should have shape (n_time, 2)"
        assert "ch" in msg.axes

        # For cart2pol, check that r values are positive
        if mode == CoordinateMode.CART2POL:
            r_values = msg.data[:, 0]
            assert np.all(r_values >= 0), "Radius should be non-negative"
            assert list(msg.axes["ch"].data) == ["r", "theta"]
        else:
            assert list(msg.axes["ch"].data) == ["x", "y"]

    ez.logger.info("Test complete.")


def test_coordinatespaces_roundtrip_system(test_name: str | None = None):
    """Test that cart->pol->cart round-trip preserves data."""
    fs = 100.0
    n_time = 100
    n_messages = 5

    test_filename = get_test_fn(test_name)
    ez.logger.info(f"Test file: {test_filename}")

    comps = {
        "SIGNAL": CartesianSignal(CartesianSignalSettings(n_time=n_time, fs=fs)),
        "CART2POL": CoordinateSpaces(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch")),
        "POL2CART": CoordinateSpaces(CoordinateSpacesSettings(mode=CoordinateMode.POL2CART, axis="ch")),
        "LOG": MessageLogger(MessageLoggerSettings(output=test_filename)),
        "TERM": TerminateOnTotal(TerminateOnTotalSettings(total=n_messages)),
    }
    conns = (
        (comps["SIGNAL"].OUTPUT_SIGNAL, comps["CART2POL"].INPUT_SIGNAL),
        (comps["CART2POL"].OUTPUT_SIGNAL, comps["POL2CART"].INPUT_SIGNAL),
        (comps["POL2CART"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    # Read logged messages
    messages: list[AxisArray] = list(message_log(test_filename))
    os.remove(test_filename)

    ez.logger.info(f"Received {len(messages)} messages")
    assert len(messages) >= n_messages

    # Verify round-trip accuracy for each message
    for msg in messages:
        offset = msg.axes["time"].offset
        n_samples = msg.data.shape[0]
        t = offset + np.arange(n_samples) / fs

        # Expected spiral pattern
        theta = 2 * np.pi * t
        r = 0.1 + 0.1 * t
        expected_x = r * np.cos(theta)
        expected_y = r * np.sin(theta)

        assert np.allclose(msg.data[:, 0], expected_x, atol=1e-10), "X values should match after round-trip"
        assert np.allclose(msg.data[:, 1], expected_y, atol=1e-10), "Y values should match after round-trip"
        assert list(msg.axes["ch"].data) == ["x", "y"]

    ez.logger.info("Round-trip test complete.")


def test_coordinatespaces_values(test_name: str | None = None):
    """Test that coordinate transformation produces correct values."""
    fs = 100.0
    n_time = 100
    n_messages = 3

    test_filename = get_test_fn(test_name)

    comps = {
        "SIGNAL": CartesianSignal(CartesianSignalSettings(n_time=n_time, fs=fs)),
        "COORD": CoordinateSpaces(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch")),
        "LOG": MessageLogger(MessageLoggerSettings(output=test_filename)),
        "TERM": TerminateOnTotal(TerminateOnTotalSettings(total=n_messages)),
    }
    conns = (
        (comps["SIGNAL"].OUTPUT_SIGNAL, comps["COORD"].INPUT_SIGNAL),
        (comps["COORD"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    messages: list[AxisArray] = list(message_log(test_filename))
    os.remove(test_filename)

    assert len(messages) >= n_messages

    # For each message, verify the transformation is correct
    for msg in messages:
        offset = msg.axes["time"].offset
        n_samples = msg.data.shape[0]
        t = offset + np.arange(n_samples) / fs

        # Calculate expected input (Cartesian)
        theta_input = 2 * np.pi * t
        r_input = 0.1 + 0.1 * t
        x_input = r_input * np.cos(theta_input)
        y_input = r_input * np.sin(theta_input)

        # Calculate expected output (polar)
        expected_r, expected_theta = cart2pol(x_input, y_input)

        # Verify transformation
        assert np.allclose(msg.data[:, 0], expected_r), "Radius values should match"
        assert np.allclose(msg.data[:, 1], expected_theta), "Theta values should match"

    ez.logger.info("Value verification test complete.")


if __name__ == "__main__":
    test_coordinatespaces_system(CoordinateMode.CART2POL, test_name="test_cart2pol")
    test_coordinatespaces_system(CoordinateMode.POL2CART, test_name="test_pol2cart")
    test_coordinatespaces_roundtrip_system(test_name="test_roundtrip")
    test_coordinatespaces_values(test_name="test_values")
