"""Add 2 signals or add a constant to a signal."""

import asyncio
import typing
from dataclasses import dataclass, field

import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.baseproc.util.asio import run_coroutine_sync
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

# --- Constant Addition (single input) ---


class ConstAddSettings(ez.Settings):
    value: float = 0.0
    """Number to add to the input data."""


class ConstAddTransformer(BaseTransformer[ConstAddSettings, AxisArray, AxisArray]):
    """Add a constant value to input data."""

    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=message.data + self.settings.value)


class ConstAdd(BaseTransformerUnit[ConstAddSettings, AxisArray, AxisArray, ConstAddTransformer]):
    """Unit wrapper for ConstAddTransformer."""

    SETTINGS = ConstAddSettings


# --- Two-input Addition ---


@dataclass
class AddState:
    """State for Add processor with two input queues."""

    queue_a: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)
    queue_b: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)


class AddProcessor:
    """Processor that adds two AxisArray signals together.

    This processor maintains separate queues for two input streams and
    adds corresponding messages element-wise. It assumes both inputs
    have compatible shapes and aligned time spans.
    """

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
        """Push a message to queue A."""
        self._state.queue_a.put_nowait(msg)

    def push_b(self, msg: AxisArray) -> None:
        """Push a message to queue B."""
        self._state.queue_b.put_nowait(msg)

    async def __acall__(self) -> AxisArray:
        """Await and add the next messages from both queues."""
        a = await self._state.queue_a.get()
        b = await self._state.queue_b.get()
        return replace(a, data=a.data + b.data)

    def __call__(self) -> AxisArray:
        """Synchronously get and add the next messages from both queues."""
        return run_coroutine_sync(self.__acall__())

    # Aliases for legacy interface
    async def __anext__(self) -> AxisArray:
        return await self.__acall__()

    def __next__(self) -> AxisArray:
        return self.__call__()


class Add(ez.Unit):
    """Add two signals together.

    Assumes compatible/similar axes/dimensions and aligned time spans.
    Messages are paired by arrival order (oldest from each queue).
    """

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
