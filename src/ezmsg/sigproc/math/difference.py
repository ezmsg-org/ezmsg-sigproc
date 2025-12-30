"""
Take the difference between 2 signals or between a signal and a constant value.

.. note::
    :obj:`ConstDifferenceTransformer` supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
    :obj:`DifferenceProcessor` (two-input difference) currently requires NumPy arrays.
"""

import asyncio
import typing
from dataclasses import dataclass, field

import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.baseproc.util.asio import run_coroutine_sync
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class ConstDifferenceSettings(ez.Settings):
    value: float = 0.0
    """number to subtract or be subtracted from the input data"""

    subtrahend: bool = True
    """If True (default) then value is subtracted from the input data. If False, the input data
    is subtracted from value."""


class ConstDifferenceTransformer(BaseTransformer[ConstDifferenceSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(
            message,
            data=(message.data - self.settings.value)
            if self.settings.subtrahend
            else (self.settings.value - message.data),
        )


class ConstDifference(BaseTransformerUnit[ConstDifferenceSettings, AxisArray, AxisArray, ConstDifferenceTransformer]):
    SETTINGS = ConstDifferenceSettings


def const_difference(value: float = 0.0, subtrahend: bool = True) -> ConstDifferenceTransformer:
    """
    result = (in_data - value) if subtrahend else (value - in_data)
    https://en.wikipedia.org/wiki/Template:Arithmetic_operations

    Args:
        value: number to subtract or be subtracted from the input data
        subtrahend: If True (default) then value is subtracted from the input data.
         If False, the input data is subtracted from value.

    Returns: :obj:`ConstDifferenceTransformer`.
    """
    return ConstDifferenceTransformer(ConstDifferenceSettings(value=value, subtrahend=subtrahend))


# --- Two-input Difference ---


@dataclass
class DifferenceState:
    """State for Difference processor with two input queues."""

    queue_a: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)
    queue_b: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)


class DifferenceProcessor:
    """Processor that subtracts two AxisArray signals (A - B).

    This processor maintains separate queues for two input streams and
    subtracts corresponding messages element-wise. It assumes both inputs
    have compatible shapes and aligned time spans.
    """

    def __init__(self):
        self._state = DifferenceState()

    @property
    def state(self) -> DifferenceState:
        return self._state

    @state.setter
    def state(self, state: DifferenceState | bytes | None) -> None:
        if state is not None:
            self._state = state

    def push_a(self, msg: AxisArray) -> None:
        """Push a message to queue A (minuend)."""
        self._state.queue_a.put_nowait(msg)

    def push_b(self, msg: AxisArray) -> None:
        """Push a message to queue B (subtrahend)."""
        self._state.queue_b.put_nowait(msg)

    async def __acall__(self) -> AxisArray:
        """Await and subtract the next messages (A - B)."""
        a = await self._state.queue_a.get()
        b = await self._state.queue_b.get()
        return replace(a, data=a.data - b.data)

    def __call__(self) -> AxisArray:
        """Synchronously get and subtract the next messages."""
        return run_coroutine_sync(self.__acall__())

    # Aliases for legacy interface
    async def __anext__(self) -> AxisArray:
        return await self.__acall__()

    def __next__(self) -> AxisArray:
        return self.__call__()


class Difference(ez.Unit):
    """Subtract two signals (A - B).

    Assumes compatible/similar axes/dimensions and aligned time spans.
    Messages are paired by arrival order (oldest from each queue).

    OUTPUT = INPUT_SIGNAL_A - INPUT_SIGNAL_B
    """

    INPUT_SIGNAL_A = ez.InputStream(AxisArray)
    INPUT_SIGNAL_B = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self.processor = DifferenceProcessor()

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
