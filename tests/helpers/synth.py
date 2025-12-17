"""
Test signal generators for ezmsg-sigproc integration tests.

These are simplified signal generators intended for testing purposes only.
For production use, see ezmsg-simbiophys package.
"""

import asyncio
import time
import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.math.add import Add  # noqa: F401 - re-exported for test use


# Counter - Produces incrementing integer samples
class CounterSettings(ez.Settings):
    n_time: int = 100
    fs: float = 1000.0
    n_ch: int = 1
    dispatch_rate: float | None = None  # Hz or None for fast as possible


class Counter(ez.Unit):
    """Simple counter generator for testing."""

    SETTINGS = CounterSettings
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self._counter = 0
        self._n_sent = 0
        self._t0 = time.time()

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        while True:
            # Sleep if needed
            if self.SETTINGS.dispatch_rate is not None:
                n_disp = 1 + self._n_sent / self.SETTINGS.n_time
                t_next = self._t0 + n_disp / self.SETTINGS.dispatch_rate
                sleep_time = t_next - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            # Generate counter data
            block = np.arange(self._counter, self._counter + self.SETTINGS.n_time)[:, np.newaxis]
            block = np.tile(block, (1, self.SETTINGS.n_ch))

            offset = self._n_sent / self.SETTINGS.fs
            result = AxisArray(
                data=block,
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.TimeAxis(fs=self.SETTINGS.fs, offset=offset),
                    "ch": AxisArray.CoordinateAxis(
                        data=np.array([f"Ch{_}" for _ in range(self.SETTINGS.n_ch)]),
                        dims=["ch"],
                    ),
                },
            )

            self._counter = block[-1, 0] + 1
            self._n_sent += self.SETTINGS.n_time

            yield self.OUTPUT_SIGNAL, result


# WhiteNoise - Produces random Gaussian noise
class WhiteNoiseSettings(ez.Settings):
    n_time: int = 100
    fs: float = 1000.0
    n_ch: int = 1
    dispatch_rate: float | None = None
    loc: float = 0.0
    scale: float = 1.0


class WhiteNoise(ez.Unit):
    """Simple white noise generator for testing."""

    SETTINGS = WhiteNoiseSettings
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self._n_sent = 0
        self._t0 = time.time()

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        while True:
            # Sleep if needed
            if self.SETTINGS.dispatch_rate is not None:
                n_disp = 1 + self._n_sent / self.SETTINGS.n_time
                t_next = self._t0 + n_disp / self.SETTINGS.dispatch_rate
                sleep_time = t_next - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            # Generate noise data
            data = np.random.normal(
                loc=self.SETTINGS.loc,
                scale=self.SETTINGS.scale,
                size=(self.SETTINGS.n_time, self.SETTINGS.n_ch),
            )

            offset = self._n_sent / self.SETTINGS.fs
            result = AxisArray(
                data=data,
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.TimeAxis(fs=self.SETTINGS.fs, offset=offset),
                    "ch": AxisArray.CoordinateAxis(
                        data=np.array([f"Ch{_}" for _ in range(self.SETTINGS.n_ch)]),
                        dims=["ch"],
                    ),
                },
            )

            self._n_sent += self.SETTINGS.n_time

            yield self.OUTPUT_SIGNAL, result


# Oscillator - Produces sinusoidal signals
class OscillatorSettings(ez.Settings):
    n_time: int = 100
    fs: float = 1000.0
    n_ch: int = 1
    dispatch_rate: float | str | None = None  # Hz, "realtime", or None for fast as possible
    freq: float = 10.0  # Hz
    amp: float = 1.0
    phase: float = 0.0
    sync: bool = False  # Adjust freq to sync with sampling rate


class Oscillator(ez.Unit):
    """Simple oscillator generator for testing."""

    SETTINGS = OscillatorSettings
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self._n_sent = 0
        self._t0 = time.time()

        # Calculate synchronized frequency if requested
        self._freq = self.SETTINGS.freq
        if self.SETTINGS.sync:
            period = 1.0 / self.SETTINGS.freq
            mod = round(period * self.SETTINGS.fs)
            self._freq = 1.0 / (mod / self.SETTINGS.fs)

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        while True:
            # Calculate offset based on timing mode
            if self.SETTINGS.dispatch_rate == "realtime":
                # Realtime mode: sleep until wall-clock time matches sample time
                n_next = self._n_sent + self.SETTINGS.n_time
                t_next = self._t0 + n_next / self.SETTINGS.fs
                sleep_time = t_next - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                offset = t_next - self.SETTINGS.n_time / self.SETTINGS.fs
            elif self.SETTINGS.dispatch_rate is not None:
                # Manual dispatch rate mode
                n_disp = 1 + self._n_sent / self.SETTINGS.n_time
                t_next = self._t0 + n_disp / self.SETTINGS.dispatch_rate
                sleep_time = t_next - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                offset = self._n_sent / self.SETTINGS.fs
            else:
                # Fast as possible mode
                offset = self._n_sent / self.SETTINGS.fs

            # Generate sinusoidal data
            sample_indices = np.arange(self._n_sent, self._n_sent + self.SETTINGS.n_time)
            t = sample_indices / self.SETTINGS.fs
            data = self.SETTINGS.amp * np.sin(2 * np.pi * self._freq * t + self.SETTINGS.phase)
            data = data[:, np.newaxis]
            data = np.tile(data, (1, self.SETTINGS.n_ch))

            result = AxisArray(
                data=data,
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.TimeAxis(fs=self.SETTINGS.fs, offset=offset),
                    "ch": AxisArray.CoordinateAxis(
                        data=np.array([f"Ch{_}" for _ in range(self.SETTINGS.n_ch)]),
                        dims=["ch"],
                    ),
                },
            )

            self._n_sent += self.SETTINGS.n_time

            yield self.OUTPUT_SIGNAL, result


# EEGSynth - Combines oscillator and pink noise (simplified version without actual pink noise filter)
class EEGSynthSettings(ez.Settings):
    fs: float = 500.0
    n_time: int = 100
    alpha_freq: float = 10.5
    n_ch: int = 8


class Clock(ez.Unit):
    """Simple clock generator."""

    OUTPUT_SIGNAL = ez.OutputStream(ez.Flag)

    SETTINGS: ez.Settings

    async def initialize(self) -> None:
        self._t0 = time.time()
        self._n_dispatch = 0

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        while True:
            if hasattr(self.SETTINGS, "dispatch_rate") and self.SETTINGS.dispatch_rate is not None:
                target_time = self._t0 + (self._n_dispatch + 1) / self.SETTINGS.dispatch_rate
                sleep_time = target_time - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self._n_dispatch += 1
            yield self.OUTPUT_SIGNAL, ez.Flag()


class ClockSettings(ez.Settings):
    dispatch_rate: float | None = None


class EEGSynth(ez.Collection):
    """
    Simple EEG-like signal generator for testing.
    Combines oscillator (alpha rhythm) with white noise.
    """

    SETTINGS = EEGSynthSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    OSC = Oscillator()
    NOISE = WhiteNoise()
    ADD = Add()

    def configure(self) -> None:
        self.OSC.apply_settings(
            OscillatorSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate=self.SETTINGS.fs / self.SETTINGS.n_time,
                freq=self.SETTINGS.alpha_freq,
            )
        )

        self.NOISE.apply_settings(
            WhiteNoiseSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate=self.SETTINGS.fs / self.SETTINGS.n_time,
                scale=5.0,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.OSC.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_A),
            (self.NOISE.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_B),
            (self.ADD.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
