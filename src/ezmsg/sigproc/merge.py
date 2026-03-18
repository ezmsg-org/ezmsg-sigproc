"""Time-aligned merge of two AxisArray streams along a non-time axis.

``Merge`` is an :class:`ez.Collection` that composes
:class:`~ezmsg.sigproc.align.AlignAlongAxis` (time-alignment) with
:class:`~ezmsg.sigproc.concat.Concat` (axis-aware concatenation).
"""

from __future__ import annotations

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray

from .align import AlignAlongAxis, AlignAlongAxisProcessor, AlignAlongAxisSettings
from .concat import Concat, ConcatProcessor, ConcatSettings


class MergeSettings(ez.Settings):
    axis: str = "ch"
    """Axis along which to concatenate the two signals."""

    align_axis: str | None = "time"
    """Axis used for alignment. If None, defaults to the first dimension."""

    buffer_dur: float = 10.0
    """Buffer duration in seconds for each input stream."""

    relabel_axis: bool = True
    """Whether to relabel coordinate axis labels to ensure uniqueness."""

    label_a: str = "_a"
    """Suffix appended to signal A labels when relabel_axis is True."""

    label_b: str = "_b"
    """Suffix appended to signal B labels when relabel_axis is True."""

    assert_identical_shared_axes: bool = False
    """If True, raise ValueError when shared CoordinateAxis .data arrays differ."""

    new_key: str | None = None
    """Output AxisArray key. If None, uses the key from signal A."""


class MergeProcessor:
    """Convenience processor that composes alignment + concatenation.

    Preserves the same call interface as the previous monolithic processor
    so that existing code using ``proc(msg_a)`` / ``proc.push_b(msg_b)``
    continues to work unchanged.
    """

    def __init__(self, settings: MergeSettings):
        self.settings = settings
        self._align = AlignAlongAxisProcessor(
            settings=AlignAlongAxisSettings(
                axis=settings.align_axis or "time",
                buffer_dur=settings.buffer_dur,
            )
        )
        self._concat = ConcatProcessor(
            settings=ConcatSettings(
                axis=settings.axis,
                relabel_axis=settings.relabel_axis,
                label_a=settings.label_a,
                label_b=settings.label_b,
                assert_identical_shared_axes=settings.assert_identical_shared_axes,
                new_key=settings.new_key,
            )
        )

    @property
    def align_state(self):
        """Expose alignment state for introspection / tests."""
        return self._align.state

    @property
    def concat_state(self):
        """Expose concatenation state for introspection / tests."""
        return self._concat.state

    def __call__(self, msg_a: AxisArray) -> AxisArray | None:
        pair = self._align(msg_a)
        if pair is not None:
            return self._concat._concat(*pair)
        return None

    async def __acall__(self, msg_a: AxisArray) -> AxisArray | None:
        pair = await self._align.__acall__(msg_a)
        if pair is not None:
            return self._concat._concat(*pair)
        return None

    def push_b(self, msg_b: AxisArray) -> AxisArray | None:
        pair = self._align.push_b(msg_b)
        if pair is not None:
            return self._concat._concat(*pair)
        return None


class Merge(ez.Collection):
    """Merge two AxisArray streams by time-aligning and concatenating.

    Composes :class:`AlignAlongAxis` → :class:`Concat`.
    """

    SETTINGS = MergeSettings

    INPUT_SIGNAL_A = ez.InputStream(AxisArray)
    INPUT_SIGNAL_B = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    ALIGN = AlignAlongAxis()
    CONCAT = Concat()

    def configure(self) -> None:
        self.ALIGN.apply_settings(
            AlignAlongAxisSettings(
                axis=self.SETTINGS.align_axis or "time",
                buffer_dur=self.SETTINGS.buffer_dur,
            )
        )
        self.CONCAT.apply_settings(
            ConcatSettings(
                axis=self.SETTINGS.axis,
                relabel_axis=self.SETTINGS.relabel_axis,
                label_a=self.SETTINGS.label_a,
                label_b=self.SETTINGS.label_b,
                assert_identical_shared_axes=self.SETTINGS.assert_identical_shared_axes,
                new_key=self.SETTINGS.new_key,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL_A, self.ALIGN.INPUT_SIGNAL_A),
            (self.INPUT_SIGNAL_B, self.ALIGN.INPUT_SIGNAL_B),
            (self.ALIGN.OUTPUT_SIGNAL_A, self.CONCAT.INPUT_SIGNAL_A),
            (self.ALIGN.OUTPUT_SIGNAL_B, self.CONCAT.INPUT_SIGNAL_B),
            (self.CONCAT.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
