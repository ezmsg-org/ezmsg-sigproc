import asyncio
import traceback
from dataclasses import dataclass, field
import time
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .butterworthfilter import ButterworthFilter, ButterworthFilterSettings
from .base import ProcessorState, BaseStatefulProducer, BaseProducerUnit, BaseTransformer, \
    BaseTransformerUnit


class ClockSettings(ez.Settings):
    """Settings for clock generator."""

    dispatch_rate: float | str | None = None
    """Dispatch rate in Hz, 'realtime', or None for external clock"""


class ClockState(ProcessorState):
    """State for clock generator."""

    t_0: float = field(default_factory=time.time)  # Start time
    n_dispatch: int = 0  # Number of dispatches


class ClockProducer(BaseStatefulProducer[ClockSettings, ez.Flag, ClockState]):
    """
    Produces clock ticks at specified rate.
    Can be used to drive periodic operations.
    """

    def _reset_state(self) -> None:
        """Reset internal state."""
        self._state.t_0 = time.time()
        self._state.n_dispatch = 0

    def __call__(self) -> ez.Flag:
        """Synchronous clock production. We override __call__ (which uses run_coroutine_sync) to avoid async overhead."""
        if self.state.hash == -1:
            self._reset_state()
            self.state.hash = 0

        if isinstance(self.settings.dispatch_rate, (int, float)):
            # Manual dispatch_rate. (else it is 'as fast as possible')
            target_time = (
                self.state.t_0
                + (self.state.n_dispatch + 1) / self.settings.dispatch_rate
            )
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)

        self.state.n_dispatch += 1
        return ez.Flag()

    async def _produce(self) -> ez.Flag:
        """Generate next clock tick."""
        if isinstance(self.settings.dispatch_rate, (int, float)):
            # Manual dispatch_rate. (else it is 'as fast as possible')
            target_time = (
                self.state.t_0
                + (self.state.n_dispatch + 1) / self.settings.dispatch_rate
            )
            now = time.time()
            if target_time > now:
                await asyncio.sleep(target_time - now)

        self.state.n_dispatch += 1
        return ez.Flag()


def aclock(dispatch_rate: float | None) -> ClockProducer:
    """
    Construct an async generator that yields events at a specified rate.

    Returns:
        A :obj:`ClockProducer` object.
    """
    return ClockProducer(ClockSettings(dispatch_rate=dispatch_rate))


clock = aclock
"""
Alias for :obj:`aclock` expected by synchronous methods. `ClockProducer` can be used in sync or async.
"""


class Clock(
    BaseProducerUnit[
        ClockSettings,  # SettingsType
        ez.Flag,  # MessageType
        ClockProducer,  # ProducerType
    ]
):
    SETTINGS = ClockSettings

    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        # Override so we can not to yield if out is False-like
        while True:
            out = await self.producer.__acall__()
            if out:
                yield self.OUTPUT_SIGNAL, out


# COUNTER - Generate incrementing integer. fs and dispatch_rate parameters combine to give many options. #
class CounterSettings(ez.Settings):
    # TODO: Adapt this to use ezmsg.util.rate?
    """
    Settings for :obj:`Counter`.
    See :obj:`acounter` for a description of the parameters.
    """

    n_time: int
    """Number of samples to output per block."""

    fs: float
    """Sampling rate of signal output in Hz"""

    n_ch: int = 1
    """Number of channels to synthesize"""

    dispatch_rate: float | str | None = None
    """
    Message dispatch rate (Hz), 'realtime', 'ext_clock', or None (fast as possible)
     Note: if dispatch_rate is a float then time offsets will be synthetic and the
     system will run faster or slower than wall clock time.
    """

    mod: int | None = None
    """If set to an integer, counter will rollover"""


@dataclass(unsafe_hash=True, frozen=False)  # Override the metaclass decorator
class CounterState:
    """State for counter generator."""
    hash: int = -1
    clock_zero: float = field(default_factory=lambda: time.time())
    counter_start: int = 0  # next sample's first value
    n_sent: int = 0  # number of samples sent
    timer_type: str = "unspecified"  # "realtime" | "ext_clock" | "manual" | "unspecified"
    new_generator: asyncio.Event = field(default_factory=lambda: asyncio.Event())


class CounterProducer(BaseStatefulProducer[CounterSettings, AxisArray, CounterState]):
    """Produces incrementing integer blocks as AxisArray."""

    # TODO: Adapt this to use ezmsg.util.rate?

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(
            self.settings.dispatch_rate, str
        ) and self.settings.dispatch_rate not in ["realtime", "ext_clock"]:
            raise ValueError(f"Unknown dispatch_rate: {self.settings.dispatch_rate}")
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset internal state."""
        self._state.counter_start = 0
        self._state.n_sent = 0
        self._state.clock_zero = time.time()
        if self.settings.dispatch_rate is not None:
            if isinstance(self.settings.dispatch_rate, str):
                self._state.timer_type = self.settings.dispatch_rate.lower()
            else:
                self._state.timer_type = "manual"

        self._state.new_generator.set()
        # I _think_ the intention is to switch between ext_clock and others without resetting the generator.

    async def _produce(self) -> AxisArray:
        """Generate next counter block."""
        # 1. Prepare counter data
        block_samp = np.arange(
            self.state.counter_start, self.state.counter_start + self.settings.n_time
        )[:, np.newaxis]
        if self.settings.mod is not None:
            block_samp %= self.settings.mod
        block_samp = np.tile(block_samp, (1, self.settings.n_ch))

        # 2. Sleep if necessary. 3. Calculate time offset.
        if self._state.timer_type == "realtime":
            n_next = self.state.n_sent + self.settings.n_time
            t_next = self.state.clock_zero + n_next / self.settings.fs
            await asyncio.sleep(t_next - time.time())
            offset = t_next - self.settings.n_time / self.settings.fs
        elif self._state.timer_type == "manual":
            # manual dispatch rate
            n_disp_next = 1 + self.state.n_sent / self.settings.n_time
            t_disp_next = (
                self.state.clock_zero + n_disp_next / self.settings.dispatch_rate
            )
            await asyncio.sleep(t_disp_next - time.time())
            offset = self.state.n_sent / self.settings.fs
        elif self._state.timer_type == "ext_clock":
            #  ext_clock -- no sleep. Assume this is called at appropriate intervals.
            offset = time.time()
        else:
            # Was None
            offset = self.state.n_sent / self.settings.fs

        # 4. Create output AxisArray
        # Note: We can make this a bit faster by preparing a template for self._state
        result = AxisArray(
            data=block_samp,
            dims=["time", "ch"],
            axes={
                "time": AxisArray.TimeAxis(fs=self.settings.fs, offset=offset),
                "ch": AxisArray.CoordinateAxis(
                    data=np.array([f"Ch{_}" for _ in range(self.settings.n_ch)]),
                    dims=["ch"],
                ),
            },
            key="acounter",
        )

        # 5. Update state
        self.state.counter_start = block_samp[-1, 0] + 1
        self.state.n_sent += self.settings.n_time

        return result


def acounter(
    n_time: int,
    fs: float | None,
    n_ch: int = 1,
    dispatch_rate: float | str | None = None,
    mod: int | None = None,
) -> CounterProducer:
    """
    Construct an asynchronous generator to generate AxisArray objects at a specified rate
    and with the specified sampling rate.

    NOTE: This module uses asyncio.sleep to delay appropriately in realtime mode.
    This method of sleeping/yielding execution priority has quirky behavior with
    sub-millisecond sleep periods which may result in unexpected behavior (e.g.
    fs = 2000, n_time = 1, realtime = True -- may result in ~1400 msgs/sec)

    Returns:
        An asynchronous generator.
    """
    return CounterProducer(
        CounterSettings(
            n_time=n_time, fs=fs, n_ch=n_ch, dispatch_rate=dispatch_rate, mod=mod
        )
    )


class Counter(
    BaseProducerUnit[
        CounterSettings,  # SettingsType
        AxisArray,  # MessageType
        CounterProducer,  # ProducerType
    ]
):
    """Generates monotonically increasing counter. Unit for :obj:`CounterProducer`."""

    SETTINGS = CounterSettings
    INPUT_CLOCK = ez.InputStream(ez.Flag)

    @ez.subscriber(INPUT_CLOCK)
    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    async def on_clock(self, clock: ez.Flag):
        if self.producer.settings.dispatch_rate == "ext_clock":
            out = await self.producer.__anext__()
            yield self.OUTPUT_SIGNAL, out

    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        try:
            while True:
                # Once-only, enter the generator loop
                await self.producer.state.new_generator.wait()
                self.producer.state.new_generator.clear()

                if self.producer.settings.dispatch_rate == "ext_clock":
                    # We shouldn't even be here. Cycle around and wait on the event again.
                    continue

                # We are not using an external clock. Run the generator.
                while not self.producer.state.new_generator.is_set():
                    out = await self.producer.__acall__()
                    yield self.OUTPUT_SIGNAL, out
        except Exception:
            ez.logger.info(traceback.format_exc())


class SinGeneratorSettings(ez.Settings):
    """
    Settings for :obj:`SinGenerator`.
    See :obj:`sin` for parameter descriptions.
    """

    axis: str | None = "time"
    """
    The name of the axis over which the sinusoid passes.
    Note: The axis must exist in the msg.axes and be of type AxisArray.LinearAxis.
    """

    freq: float = 1.0
    """The frequency of the sinusoid, in Hz."""

    amp: float = 1.0  # Amplitude
    """The amplitude of the sinusoid."""

    phase: float = 0.0  # Phase offset (in radians)
    """The initial phase of the sinusoid, in radians."""


class SinTransformer(BaseTransformer[SinGeneratorSettings, AxisArray]):
    """Transforms counter values into sinusoidal waveforms."""

    def _process(self, message: AxisArray) -> AxisArray:
        """Transform input counter values into sinusoidal waveform."""
        axis = self.settings.axis or message.dims[0]

        ang_freq = 2.0 * np.pi * self.settings.freq
        w = (ang_freq * message.get_axis(axis).gain) * message.data
        out_data = self.settings.amp * np.sin(w + self.settings.phase)

        return replace(message, data=out_data)


class SinGenerator(BaseTransformerUnit[SinGeneratorSettings, AxisArray, SinTransformer]):
    """Unit for generating sinusoidal waveforms."""
    SETTINGS = SinGeneratorSettings


def sin(
    axis: str | None = "time",
    freq: float = 1.0,
    amp: float = 1.0,
    phase: float = 0.0,
) -> SinTransformer:
    """
    Construct a generator of sinusoidal waveforms in AxisArray objects.

    Returns:
        A primed generator that expects .send(axis_array) of sample counts
        and yields an AxisArray of sinusoids.
    """
    return SinTransformer(SinGeneratorSettings(axis=axis, freq=freq, amp=amp, phase=phase))


class OscillatorSettings(ez.Settings):
    """Settings for :obj:`Oscillator`"""

    n_time: int
    """Number of samples to output per block."""

    fs: float
    """Sampling rate of signal output in Hz"""

    n_ch: int = 1
    """Number of channels to output per block"""

    dispatch_rate: float | str | None = None
    """(Hz) | 'realtime' | 'ext_clock'"""

    freq: float = 1.0
    """Oscillation frequency in Hz"""

    amp: float = 1.0
    """Amplitude"""

    phase: float = 0.0
    """Phase offset (in radians)"""

    sync: bool = False
    """Adjust `freq` to sync with sampling rate"""


class Oscillator(ez.Collection):
    """
    :obj:`Collection that chains :obj:`Counter` and :obj:`SinGenerator`.
    """

    SETTINGS = OscillatorSettings

    INPUT_CLOCK = ez.InputStream(ez.Flag)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    COUNTER = Counter()
    SIN = SinGenerator()

    def configure(self) -> None:
        # Calculate synchronous settings if necessary
        freq = self.SETTINGS.freq
        mod = None
        if self.SETTINGS.sync:
            period = 1.0 / self.SETTINGS.freq
            mod = round(period * self.SETTINGS.fs)
            freq = 1.0 / (mod / self.SETTINGS.fs)

        self.COUNTER.apply_settings(
            CounterSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate=self.SETTINGS.dispatch_rate,
                mod=mod,
            )
        )

        self.SIN.apply_settings(
            SinGeneratorSettings(
                freq=freq, amp=self.SETTINGS.amp, phase=self.SETTINGS.phase
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_CLOCK, self.COUNTER.INPUT_CLOCK),
            (self.COUNTER.OUTPUT_SIGNAL, self.SIN.INPUT_SIGNAL),
            (self.SIN.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )


class RandomGeneratorSettings(ez.Settings):
    loc: float = 0.0
    """loc argument for :obj:`numpy.random.normal`"""

    scale: float = 1.0
    """scale argument for :obj:`numpy.random.normal`"""


class RandomGenerator(ez.Unit):
    """
    Replaces input data with random data and yields the result.
    """

    SETTINGS = RandomGeneratorSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_SIGNAL)
    async def generate(self, msg: AxisArray) -> typing.AsyncGenerator:
        random_data = np.random.normal(
            size=msg.shape, loc=self.SETTINGS.loc, scale=self.SETTINGS.scale
        )

        yield self.OUTPUT_SIGNAL, replace(msg, data=random_data)


class NoiseSettings(ez.Settings):
    """
    See :obj:`CounterSettings` and :obj:`RandomGeneratorSettings`.
    """

    n_time: int  # Number of samples to output per block
    fs: float  # Sampling rate of signal output in Hz
    n_ch: int = 1  # Number of channels to output
    dispatch_rate: float | str | None = None
    """(Hz), 'realtime', or 'ext_clock'"""
    loc: float = 0.0  # DC offset
    scale: float = 1.0  # Scale (in standard deviations)


WhiteNoiseSettings = NoiseSettings


class WhiteNoise(ez.Collection):
    """
    A :obj:`Collection` that chains a :obj:`Counter` and :obj:`RandomGenerator`.
    """

    SETTINGS = NoiseSettings

    INPUT_CLOCK = ez.InputStream(ez.Flag)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    COUNTER = Counter()
    RANDOM = RandomGenerator()

    def configure(self) -> None:
        self.RANDOM.apply_settings(
            RandomGeneratorSettings(loc=self.SETTINGS.loc, scale=self.SETTINGS.scale)
        )

        self.COUNTER.apply_settings(
            CounterSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate=self.SETTINGS.dispatch_rate,
                mod=None,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_CLOCK, self.COUNTER.INPUT_CLOCK),
            (self.COUNTER.OUTPUT_SIGNAL, self.RANDOM.INPUT_SIGNAL),
            (self.RANDOM.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )


PinkNoiseSettings = NoiseSettings


class PinkNoise(ez.Collection):
    """
    A :obj:`Collection` that chains :obj:`WhiteNoise` and :obj:`ButterworthFilter`.
    """

    SETTINGS = PinkNoiseSettings

    INPUT_CLOCK = ez.InputStream(ez.Flag)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    WHITE_NOISE = WhiteNoise()
    FILTER = ButterworthFilter()

    def configure(self) -> None:
        self.WHITE_NOISE.apply_settings(self.SETTINGS)
        self.FILTER.apply_settings(
            ButterworthFilterSettings(
                axis="time",
                order=1,
                cutoff=self.SETTINGS.fs * 0.01,  # Hz
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_CLOCK, self.WHITE_NOISE.INPUT_CLOCK),
            (self.WHITE_NOISE.OUTPUT_SIGNAL, self.FILTER.INPUT_SIGNAL),
            (self.FILTER.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )


class AddState(ez.State):
    queue_a: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)
    queue_b: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)


class Add(ez.Unit):
    """Add two signals together.  Assumes compatible/similar axes/dimensions."""

    STATE = AddState

    INPUT_SIGNAL_A = ez.InputStream(AxisArray)
    INPUT_SIGNAL_B = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL_A)
    async def on_a(self, msg: AxisArray) -> None:
        self.STATE.queue_a.put_nowait(msg)

    @ez.subscriber(INPUT_SIGNAL_B)
    async def on_b(self, msg: AxisArray) -> None:
        self.STATE.queue_b.put_nowait(msg)

    @ez.publisher(OUTPUT_SIGNAL)
    async def output(self) -> typing.AsyncGenerator:
        while True:
            a = await self.STATE.queue_a.get()
            b = await self.STATE.queue_b.get()

            yield self.OUTPUT_SIGNAL, replace(a, data=a.data + b.data)


class EEGSynthSettings(ez.Settings):
    """See :obj:`OscillatorSettings`."""

    fs: float = 500.0  # Hz
    n_time: int = 100
    alpha_freq: float = 10.5  # Hz
    n_ch: int = 8


class EEGSynth(ez.Collection):
    """
    A :obj:`Collection` that chains a :obj:`Clock` to both :obj:`PinkNoise`
    and :obj:`Oscillator`, then :obj:`Add` s the result.
    """

    SETTINGS = EEGSynthSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    CLOCK = Clock()
    NOISE = PinkNoise()
    OSC = Oscillator()
    ADD = Add()

    def configure(self) -> None:
        self.CLOCK.apply_settings(
            ClockSettings(dispatch_rate=self.SETTINGS.fs / self.SETTINGS.n_time)
        )

        self.OSC.apply_settings(
            OscillatorSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate="ext_clock",
                freq=self.SETTINGS.alpha_freq,
            )
        )

        self.NOISE.apply_settings(
            PinkNoiseSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate="ext_clock",
                scale=5.0,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_SIGNAL, self.OSC.INPUT_CLOCK),
            (self.CLOCK.OUTPUT_SIGNAL, self.NOISE.INPUT_CLOCK),
            (self.OSC.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_A),
            (self.NOISE.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_B),
            (self.ADD.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
