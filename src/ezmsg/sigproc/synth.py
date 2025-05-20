import asyncio
import traceback
from dataclasses import dataclass, field
import time
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .butterworthfilter import ButterworthFilterSettings, ButterworthFilterTransformer
from .base import (
    BaseStatefulProducer,
    BaseProducerUnit,
    BaseTransformer,
    BaseTransformerUnit,
    CompositeProducer,
    ProducerType,
    SettingsType,
    MessageInType,
    MessageOutType,
    processor_state,
)
from .util.asio import run_coroutine_sync
from .util.profile import profile_subpub


@dataclass
class AddState:
    queue_a: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)
    queue_b: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)


class AddProcessor:
    def __init__(self):
        self._state = AddState()

    @property
    def state(self) -> AddState:
        return self._state

    @state.setter
    def state(self, state: AddState | bytes | None) -> None:
        if state is not None:
            # TODO: Support hydrating state from bytes
            # if isinstance(state, bytes):
            #     self._state = pickle.loads(state)
            # else:
            self._state = state

    def push_a(self, msg: AxisArray) -> None:
        self._state.queue_a.put_nowait(msg)

    def push_b(self, msg: AxisArray) -> None:
        self._state.queue_b.put_nowait(msg)

    async def __acall__(self) -> AxisArray:
        a = await self._state.queue_a.get()
        b = await self._state.queue_b.get()
        return replace(a, data=a.data + b.data)

    def __call__(self) -> AxisArray:
        return run_coroutine_sync(self.__acall__())

    # Aliases for legacy interface
    async def __anext__(self) -> AxisArray:
        return await self.__acall__()

    def __next__(self) -> AxisArray:
        return self.__call__()


class Add(ez.Unit):
    """Add two signals together.  Assumes compatible/similar axes/dimensions."""

    INPUT_SIGNAL_A = ez.InputStream(AxisArray)
    INPUT_SIGNAL_B = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self.processor = AddProcessor()

    @ez.subscriber(INPUT_SIGNAL_A)
    async def on_a(self, msg: AxisArray) -> None:
        self.processor.push_a(msg)

    @ez.subscriber(INPUT_SIGNAL_B)
    async def on_b(self, msg: AxisArray) -> None:
        self.processor.push_b(msg)

    @ez.publisher(OUTPUT_SIGNAL)
    async def output(self) -> typing.AsyncGenerator:
        while True:
            yield self.OUTPUT_SIGNAL, await self.processor.__acall__()


class ClockSettings(ez.Settings):
    """Settings for clock generator."""

    dispatch_rate: float | str | None = None
    """Dispatch rate in Hz, 'realtime', or None for external clock"""


@processor_state
class ClockState:
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
        if self._hash == -1:
            self._reset_state()
            self._hash = 0

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


@processor_state
class CounterState:
    """
    State for counter generator.
    """

    counter_start: int = 0
    """next sample's first value"""

    n_sent: int = 0
    """number of samples sent"""

    clock_zero: float | None = None
    """time of first sample"""

    timer_type: str = "unspecified"
    """
    "realtime" | "ext_clock" | "manual" | "unspecified"
    """

    new_generator: asyncio.Event | None = None
    """
    Event to signal the counter has been reset.
    """


class CounterProducer(BaseStatefulProducer[CounterSettings, AxisArray, CounterState]):
    """Produces incrementing integer blocks as AxisArray."""

    # TODO: Adapt this to use ezmsg.util.rate?

    @classmethod
    def get_message_type(cls, dir: str) -> typing.Optional[type[AxisArray]]:
        if dir == "in":
            return None
        elif dir == "out":
            return AxisArray
        else:
            raise ValueError(f"Invalid direction: {dir}. Use 'in' or 'out'.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(
            self.settings.dispatch_rate, str
        ) and self.settings.dispatch_rate not in ["realtime", "ext_clock"]:
            raise ValueError(f"Unknown dispatch_rate: {self.settings.dispatch_rate}")
        self._reset_state()
        self._hash = 0

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
        if self._state.new_generator is None:
            self._state.new_generator = asyncio.Event()
        # Set the event to indicate that the state has been reset.
        self._state.new_generator.set()

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
            # Was "unspecified"
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
        AxisArray,  # MessageOutType
        CounterProducer,  # ProducerType
    ]
):
    """Generates monotonically increasing counter. Unit for :obj:`CounterProducer`."""

    SETTINGS = CounterSettings
    INPUT_CLOCK = ez.InputStream(ez.Flag)

    @ez.subscriber(INPUT_CLOCK)
    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    async def on_clock(self, _: ez.Flag):
        if self.producer.settings.dispatch_rate == "ext_clock":
            out = await self.producer.__acall__()
            yield self.OUTPUT_SIGNAL, out

    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        """
        Generate counter output.
        This is an infinite loop, but we will likely only enter the loop once if we are self-timed,
        and twice if we are using an external clock.

        When using an internal clock, we enter the loop, and wait for the event which should have
        been reset upon initialization then we immediately clear, then go to the internal loop
        that will async call __acall__ to let the internal timer determine when to produce an output.

        When using an external clock, we enter the loop, and wait for the event which should have been
        reset upon initialization then we immediately clear, then we hit `continue` to loop back around
        and wait for the event to be set again -- potentially forever. In this case, it is expected that
        `on_clock` will be called to produce the output.
        """
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


class SinTransformer(BaseTransformer[SinGeneratorSettings, AxisArray, AxisArray]):
    """Transforms counter values into sinusoidal waveforms."""

    def _process(self, message: AxisArray) -> AxisArray:
        """Transform input counter values into sinusoidal waveform."""
        axis = self.settings.axis or message.dims[0]

        ang_freq = 2.0 * np.pi * self.settings.freq
        w = (ang_freq * message.get_axis(axis).gain) * message.data
        out_data = self.settings.amp * np.sin(w + self.settings.phase)

        return replace(message, data=out_data)


class SinGenerator(
    BaseTransformerUnit[SinGeneratorSettings, AxisArray, AxisArray, SinTransformer]
):
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
    return SinTransformer(
        SinGeneratorSettings(axis=axis, freq=freq, amp=amp, phase=phase)
    )


class RandomGeneratorSettings(ez.Settings):
    loc: float = 0.0
    """loc argument for :obj:`numpy.random.normal`"""

    scale: float = 1.0
    """scale argument for :obj:`numpy.random.normal`"""


class RandomTransformer(BaseTransformer[RandomGeneratorSettings, AxisArray, AxisArray]):
    """
    Replaces input data with random data and returns the result.
    """

    def __init__(
        self, *args, settings: RandomGeneratorSettings | None = None, **kwargs
    ):
        super().__init__(*args, settings=settings, **kwargs)

    def _process(self, message: AxisArray) -> AxisArray:
        random_data = np.random.normal(
            size=message.shape, loc=self.settings.loc, scale=self.settings.scale
        )
        return replace(message, data=random_data)


class RandomGenerator(
    BaseTransformerUnit[
        RandomGeneratorSettings,
        AxisArray,
        AxisArray,
        RandomTransformer,
    ]
):
    SETTINGS = RandomGeneratorSettings


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


class OscillatorProducer(CompositeProducer[OscillatorSettings, AxisArray]):
    @staticmethod
    def _initialize_processors(
        settings: OscillatorSettings,
    ) -> dict[str, CounterProducer | SinTransformer]:
        # Calculate synchronous settings if necessary
        freq = settings.freq
        mod = None
        if settings.sync:
            period = 1.0 / settings.freq
            mod = round(period * settings.fs)
            freq = 1.0 / (mod / settings.fs)

        return {
            "counter": CounterProducer(
                CounterSettings(
                    n_time=settings.n_time,
                    fs=settings.fs,
                    n_ch=settings.n_ch,
                    dispatch_rate=settings.dispatch_rate,
                    mod=mod,
                )
            ),
            "sin": SinTransformer(
                SinGeneratorSettings(freq=freq, amp=settings.amp, phase=settings.phase)
            ),
        }


class BaseCounterFirstProducerUnit(
    BaseProducerUnit[SettingsType, MessageOutType, ProducerType],
    typing.Generic[SettingsType, MessageInType, MessageOutType, ProducerType],
):
    """
    Base class for units whose primary processor is a composite producer with a CounterProducer as the first
    processor (producer) in the chain.
    """

    INPUT_SIGNAL = ez.InputStream(MessageInType)

    def create_producer(self):
        super().create_producer()

        def recurse_get_counter(proc) -> CounterProducer:
            if hasattr(proc, "_procs"):
                return recurse_get_counter(list(proc._procs.values())[0])
            return proc

        self._counter = recurse_get_counter(self.producer)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, _: ez.Flag):
        if self.producer.settings.dispatch_rate == "ext_clock":
            out = await self.producer.__acall__()
            yield self.OUTPUT_SIGNAL, out

    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        try:
            counter_state = self._counter.state
            while True:
                # Once-only, enter the generator loop
                await counter_state.new_generator.wait()
                counter_state.new_generator.clear()

                if self.producer.settings.dispatch_rate == "ext_clock":
                    # We shouldn't even be here. Cycle around and wait on the event again.
                    continue

                # We are not using an external clock. Run the generator.
                while not counter_state.new_generator.is_set():
                    out = await self.producer.__acall__()
                    yield self.OUTPUT_SIGNAL, out
        except Exception:
            ez.logger.info(traceback.format_exc())


class Oscillator(
    BaseCounterFirstProducerUnit[
        OscillatorSettings, AxisArray, AxisArray, OscillatorProducer
    ]
):
    """Generates sinusoidal waveforms using a counter and sine transformer."""

    SETTINGS = OscillatorSettings


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


class WhiteNoiseProducer(CompositeProducer[NoiseSettings, AxisArray]):
    @staticmethod
    def _initialize_processors(
        settings: NoiseSettings,
    ) -> dict[str, CounterProducer | RandomTransformer]:
        return {
            "counter": CounterProducer(
                CounterSettings(
                    n_time=settings.n_time,
                    fs=settings.fs,
                    n_ch=settings.n_ch,
                    dispatch_rate=settings.dispatch_rate,
                    mod=None,
                )
            ),
            "random": RandomTransformer(
                RandomGeneratorSettings(
                    loc=settings.loc,
                    scale=settings.scale,
                )
            ),
        }


class WhiteNoise(
    BaseCounterFirstProducerUnit[
        NoiseSettings, AxisArray, AxisArray, WhiteNoiseProducer
    ]
):
    """chains a :obj:`Counter` and :obj:`RandomGenerator`."""

    SETTINGS = NoiseSettings


PinkNoiseSettings = NoiseSettings


class PinkNoiseProducer(CompositeProducer[PinkNoiseSettings, AxisArray]):
    @staticmethod
    def _initialize_processors(
        settings: PinkNoiseSettings,
    ) -> dict[str, WhiteNoiseProducer | ButterworthFilterTransformer]:
        return {
            "white_noise": WhiteNoiseProducer(settings=settings),
            "filter": ButterworthFilterTransformer(
                settings=ButterworthFilterSettings(
                    axis="time",
                    order=1,
                    cutoff=settings.fs * 0.01,  # Hz
                )
            ),
        }


class PinkNoise(
    BaseCounterFirstProducerUnit[NoiseSettings, AxisArray, AxisArray, PinkNoiseProducer]
):
    """chains :obj:`WhiteNoise` and :obj:`ButterworthFilter`."""

    SETTINGS = NoiseSettings


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

    Unlike the Oscillator, WhiteNoise, and PinkNoise composite processors which have linear
    flows, this class has a diamond flow, with clock branching to both PinkNoise and Oscillator,
    which then are combined in Add.

    Optional: Refactor as a ProducerUnit, similar to Clock, but we manually add all the other
     transformers.
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
            (self.CLOCK.OUTPUT_SIGNAL, self.OSC.INPUT_SIGNAL),
            (self.CLOCK.OUTPUT_SIGNAL, self.NOISE.INPUT_SIGNAL),
            (self.OSC.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_A),
            (self.NOISE.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_B),
            (self.ADD.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
