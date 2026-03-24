"""Time-align two AxisArray streams, outputting paired aligned chunks."""

from __future__ import annotations

import math
import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc.protocols import processor_state
from ezmsg.baseproc.stateful import BaseStatefulTransformer
from ezmsg.util.messages.axisarray import AxisArray

from .util.axisarray_buffer import HybridAxisArrayBuffer


class AlignAlongAxisSettings(ez.Settings):
    axis: str = "time"
    """Axis used for alignment (typically the time axis)."""

    buffer_dur: float = 10.0
    """Buffer duration in seconds for each input stream."""


@processor_state
class AlignAlongAxisState:
    gain: float | None = None
    align_axis: str | None = None
    aligned: bool = False
    buf_a: HybridAxisArrayBuffer | None = None
    buf_b: HybridAxisArrayBuffer | None = None
    # Per-input non-alignment shape for reset detection.
    a_shape_sig: tuple[int, ...] | None = None
    b_shape_sig: tuple[int, ...] | None = None


_AlignPair = tuple[AxisArray, AxisArray]


class AlignAlongAxisProcessor(
    BaseStatefulTransformer[
        AlignAlongAxisSettings,
        AxisArray,
        _AlignPair | None,
        AlignAlongAxisState,
    ]
):
    """Processor that time-aligns two AxisArray streams.

    Input A flows through ``__call__`` / ``_process`` with automatic
    hash-based reset.  Input B flows through :meth:`push_b`.

    Returns ``(aligned_a, aligned_b)`` when alignment succeeds, else ``None``.
    """

    # -- Helpers -------------------------------------------------------------

    def _extract_gain(self, message: AxisArray) -> float | None:
        align_name = self.settings.axis or message.dims[0]
        ax = message.axes.get(align_name)
        if ax is not None and hasattr(ax, "gain"):
            return ax.gain
        if ax is not None and hasattr(ax, "data") and len(ax.data) > 1:
            return float(ax.data[-1] - ax.data[0]) / (len(ax.data) - 1)
        return None

    @staticmethod
    def _non_align_shape(message: AxisArray, align_axis: str) -> tuple[int, ...]:
        align_idx = message.dims.index(align_axis)
        return tuple(s for i, s in enumerate(message.data.shape) if i != align_idx)

    # -- Reset helpers -------------------------------------------------------

    def _full_reset(self, align_axis: str) -> None:
        """
        Reset state. Called either on input A (through default __call__ path)
        or on Input B.
        Args:
            align_axis:

        Returns:

        """
        self._state.align_axis = align_axis
        self._state.buf_a = HybridAxisArrayBuffer(duration=self.settings.buffer_dur, axis=align_axis)
        self._state.buf_b = HybridAxisArrayBuffer(duration=self.settings.buffer_dur, axis=align_axis)
        self._state.gain = None
        self._state.aligned = False
        self._state.a_shape_sig = None
        self._state.b_shape_sig = None

    def _reset_a_state(self) -> None:
        self._state.buf_a = HybridAxisArrayBuffer(duration=self.settings.buffer_dur, axis=self._state.align_axis)

    def _reset_b_state(self) -> None:
        self._state.buf_b = HybridAxisArrayBuffer(duration=self.settings.buffer_dur, axis=self._state.align_axis)

    # -- BaseStatefulTransformer interface ------------------------------------

    def _hash_message(self, message: AxisArray) -> int:
        return hash(self._extract_gain(message))

    def _reset_state(self, message: AxisArray) -> None:
        align_axis = self.settings.axis or message.dims[0]
        self._full_reset(align_axis)

    def _process(self, message: AxisArray) -> _AlignPair | None:
        """Process input A: detect shape changes, buffer, try align."""
        shape_sig = self._non_align_shape(message, self._state.align_axis)
        if self._state.a_shape_sig is not None and shape_sig != self._state.a_shape_sig:
            self._reset_a_state()
            self._state.aligned = False
        self._state.a_shape_sig = shape_sig

        self._state.buf_a.write(message)
        if self._state.gain is None:
            self._state.gain = self._state.buf_a.axis_gain
        return self._try_align()

    # -- Input B entry point ------------------------------------------------

    def push_b(self, message: AxisArray) -> _AlignPair | None:
        """Process input B: check gain, detect shape changes, buffer, try align."""
        align_axis = self.settings.axis or message.dims[0]

        # Gain compatibility check.
        b_gain = self._extract_gain(message)
        if self._state.gain is not None and not math.isclose(b_gain, self._state.gain):
            self._full_reset(align_axis)
            self._hash = self._hash_message(message)

        # Lazy-create buf_b if B arrives before A.
        if self._state.buf_b is None:
            if self._state.align_axis is None:
                self._state.align_axis = align_axis
            self._state.buf_b = HybridAxisArrayBuffer(duration=self.settings.buffer_dur, axis=align_axis)

        shape_sig = self._non_align_shape(message, align_axis)
        if self._state.b_shape_sig is not None and shape_sig != self._state.b_shape_sig:
            self._reset_b_state()
            self._state.aligned = False
        self._state.b_shape_sig = shape_sig

        self._state.buf_b.write(message)
        if self._state.gain is None:
            self._state.gain = self._state.buf_b.axis_gain
        return self._try_align()

    # -- Core alignment logic -----------------------------------------------

    def _try_align(self) -> _AlignPair | None:
        """Align and read from both buffers, returning the pair ``(a, b)``."""
        if self._state.buf_a is None or self._state.buf_b is None:
            return None
        if self._state.buf_a.is_empty() or self._state.buf_b.is_empty():
            return None

        gain = self._state.gain

        # --- Initial alignment (runs once) ---
        if not self._state.aligned:
            first_a = self._state.buf_a.axis_first_value
            final_a = self._state.buf_a.axis_final_value
            first_b = self._state.buf_b.axis_first_value
            final_b = self._state.buf_b.axis_final_value

            overlap_start = max(first_a, first_b)
            overlap_end = min(final_a, final_b)

            if overlap_end < overlap_start - gain / 2:
                if final_a < first_b:
                    self._state.buf_a.seek(self._state.buf_a.available())
                elif final_b < first_a:
                    self._state.buf_b.seek(self._state.buf_b.available())
                return None

            if first_a < overlap_start - gain / 2:
                self._state.buf_a.seek(int(round((overlap_start - first_a) / gain)))
            if first_b < overlap_start - gain / 2:
                self._state.buf_b.seek(int(round((overlap_start - first_b) / gain)))

        # --- Read aligned samples ---
        n_read = min(self._state.buf_a.available(), self._state.buf_b.available())
        if n_read <= 0:
            return None

        aa_a = self._state.buf_a.read(n_read)
        aa_b = self._state.buf_b.read(n_read)
        if aa_a is None or aa_b is None:
            return None

        if not self._state.aligned:
            axis_a = aa_a.axes.get(self._state.align_axis)
            axis_b = aa_b.axes.get(self._state.align_axis)
            if axis_a is not None and axis_b is not None:
                off_a = axis_a.value(0) if hasattr(axis_a, "value") else None
                off_b = axis_b.value(0) if hasattr(axis_b, "value") else None
                if off_a is not None and off_b is not None:
                    if not np.isclose(off_a, off_b, atol=abs(gain) * 1e-6):
                        raise RuntimeError(
                            f"Offset mismatch after alignment: " f"off_a={off_a}, off_b={off_b}, gain={gain}"
                        )
            self._state.aligned = True

        return aa_a, aa_b


class AlignAlongAxis(ez.Unit):
    """Time-align two AxisArray streams and output paired aligned chunks.

    Each subscriber can publish to *both* output streams; when alignment
    succeeds, a paired (A, B) result is yielded to the respective outputs.
    """

    SETTINGS = AlignAlongAxisSettings

    INPUT_SIGNAL_A = ez.InputStream(AxisArray)
    INPUT_SIGNAL_B = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL_A = ez.OutputStream(AxisArray)
    OUTPUT_SIGNAL_B = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self.processor = AlignAlongAxisProcessor(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SIGNAL_A, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL_A)
    @ez.publisher(OUTPUT_SIGNAL_B)
    async def on_a(self, msg: AxisArray) -> typing.AsyncGenerator:
        pair = await self.processor.__acall__(msg)
        if pair is not None:
            yield self.OUTPUT_SIGNAL_A, pair[0]
            yield self.OUTPUT_SIGNAL_B, pair[1]

    @ez.subscriber(INPUT_SIGNAL_B, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL_A)
    @ez.publisher(OUTPUT_SIGNAL_B)
    async def on_b(self, msg: AxisArray) -> typing.AsyncGenerator:
        pair = self.processor.push_b(msg)
        if pair is not None:
            yield self.OUTPUT_SIGNAL_A, pair[0]
            yield self.OUTPUT_SIGNAL_B, pair[1]
