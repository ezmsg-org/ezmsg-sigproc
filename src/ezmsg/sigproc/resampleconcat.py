"""Resample one stream onto another's clock and concatenate them in one step.

This fuses :class:`~ezmsg.sigproc.resample.ResampleProcessor` and
:class:`~ezmsg.sigproc.concat.ConcatProcessor` for the common case of combining two
acquisition streams that share a nominal sampling rate but drift slightly:

* ``INPUT_REFERENCE`` -- the *reference* stream (its clock is the master timebase, and
  its data becomes one half of the concatenated output).
* ``INPUT_SIGNAL`` -- the stream that is resampled onto the reference clock.
* ``OUTPUT_SIGNAL`` -- the two streams concatenated along ``concat_axis``.

Because the resampler emits the reference signal gathered onto the *exact* grid it
resampled the signal onto (``output_reference=True``), the two halves share an identical
alignment axis by construction. No second time-alignment (``Align``) is needed, so this
collapses the otherwise-diamond graph

    REF  -> RESAMPLE.INPUT_REFERENCE
    SIG  -> RESAMPLE.INPUT_SIGNAL
    REF  -> MERGE.INPUT_SIGNAL_A
    RESAMPLE.OUTPUT -> MERGE.INPUT_SIGNAL_B

into a single linear two-input unit. All interpolation logic is reused from
``resample.py`` and all axis/attr-merge logic from ``concat.py``; only the orchestration
is new.
"""

from __future__ import annotations

import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from .concat import ConcatProcessor, ConcatSettings
from .resample import ResampleProcessor, ResampleSettings
from .util.buffer import UpdateStrategy
from .util.message import has_samples_along


class ResampleConcatSettings(ez.Settings):
    axis: str = "time"
    """Alignment axis: the reference stream's clock that the signal is resampled onto."""

    concat_axis: str = "ch"
    """Axis along which the reference and resampled signals are concatenated."""

    # --- Resample passthrough ---
    buffer_duration: float = 2.0
    fill_value: str = "extrapolate"
    max_chunk_delay: float = np.inf
    buffer_update_strategy: UpdateStrategy = "immediate"
    reference_reset_after_chunks: float = 3
    """See :attr:`ezmsg.sigproc.resample.ResampleSettings.reference_reset_after_chunks`."""

    # --- Concat passthrough ---
    relabel_axis: bool = True
    label_a: str = "_a"
    """Per-side label for the reference (A) side. See :class:`~ezmsg.sigproc.concat.ConcatSettings`."""
    label_b: str = "_b"
    """Per-side label for the resampled-signal (B) side."""
    assert_identical_shared_axes: bool = False
    new_key: str | None = None
    auto_coerce_backend: bool = False


class ResampleConcatProcessor:
    """Resample ``INPUT_SIGNAL`` onto ``INPUT_REFERENCE``'s clock, then concatenate.

    Composes an unmodified :class:`ResampleProcessor` (with ``output_reference=True``)
    and an unmodified :class:`ConcatProcessor`. The reference signal (A) and the
    resampled signal (B) are concatenated only when the resampler yields both on the same
    grid, so the two are aligned by construction.
    """

    def __init__(self, settings: ResampleConcatSettings):
        self.settings = settings
        self._resampler = ResampleProcessor(
            settings=ResampleSettings(
                axis=settings.axis,
                resample_rate=None,  # reference-driven; required for output_reference
                buffer_duration=settings.buffer_duration,
                fill_value=settings.fill_value,
                max_chunk_delay=settings.max_chunk_delay,
                buffer_update_strategy=settings.buffer_update_strategy,
                output_reference=True,
                reference_reset_after_chunks=settings.reference_reset_after_chunks,
            )
        )
        self._concat = ConcatProcessor(
            ConcatSettings(
                axis=settings.concat_axis,
                align_axis=settings.axis,
                relabel_axis=settings.relabel_axis,
                label_a=settings.label_a,
                label_b=settings.label_b,
                assert_identical_shared_axes=settings.assert_identical_shared_axes,
                new_key=settings.new_key,
                auto_coerce_backend=settings.auto_coerce_backend,
            )
        )

    @property
    def resample_state(self):
        """Expose resampler state for introspection / tests."""
        return self._resampler.state

    @property
    def concat_state(self):
        """Expose concat state for introspection / tests."""
        return self._concat.state

    def push_reference(self, message: AxisArray) -> None:
        """Feed a reference (clock + side-A) message."""
        self._resampler.push_reference(message)

    def push_signal(self, message: AxisArray) -> None:
        """Feed a signal message to be resampled onto the reference clock (side B)."""
        self._resampler(message)

    def __call__(self, message: AxisArray) -> AxisArray | None:
        """Push a signal message and return the next concatenated chunk, if any."""
        self.push_signal(message)
        return next(self)

    async def __acall__(self, message: AxisArray) -> AxisArray | None:
        return self(message)

    def __next__(self) -> AxisArray | None:
        b = next(self._resampler)  # resampled signal (side B)
        if b.data.shape[0] == 0:
            return None
        a = self._resampler.state.reference_output  # reference on the same grid (side A)
        if a is None:
            return None
        return self._concat._concat(a, b)


class ResampleConcat(ez.Unit):
    """One-step resample-onto-reference + concatenate.

    See module docstring. ``INPUT_REFERENCE`` is the master clock and side A;
    ``INPUT_SIGNAL`` is resampled onto it and becomes side B.
    """

    SETTINGS = ResampleConcatSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    INPUT_REFERENCE = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self.processor = ResampleConcatProcessor(self.SETTINGS)

    def _drain(self) -> typing.Iterator[AxisArray]:
        """Yield every chunk the processor can currently produce.

        Event-driven draining after each push is lossless here: the composed
        resampler is always reference-driven (``resample_rate=None``), and in
        that mode output readiness only ever changes on new input -- the
        wall-clock ``max_chunk_delay`` extrapolation applies to prescribed-rate
        mode only. A single ``next()`` consumes all currently-eligible
        reference values, so this loop runs at most twice.
        """
        while True:
            result = next(self.processor)
            # None / a missing or empty resample axis means "nothing ready"; a
            # chunk that is empty only along other axes (e.g. the concatenated
            # feature axis) is still a real output and must be published.
            if result is None or not has_samples_along(result, self.SETTINGS.axis):
                return
            yield result

    @ez.subscriber(INPUT_REFERENCE)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_reference(self, message: AxisArray) -> typing.AsyncGenerator:
        self.processor.push_reference(message)
        for out in self._drain():
            yield self.OUTPUT_SIGNAL, out

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        self.processor.push_signal(message)
        for out in self._drain():
            yield self.OUTPUT_SIGNAL, out
