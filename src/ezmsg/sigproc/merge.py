"""Time-aligned merge of two AxisArray streams along a non-time axis."""

from __future__ import annotations

import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc.protocols import processor_state
from ezmsg.baseproc.stateful import BaseStatefulTransformer
from ezmsg.baseproc.units import BaseProcessorUnit
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from ezmsg.util.messages.util import replace

from .util.axisarray_buffer import HybridAxisArrayBuffer


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

    new_key: str | None = None
    """Output AxisArray key. If None, uses the key from signal A."""


@processor_state
class MergeState:
    # Common state
    gain: float | None = None
    align_axis: str | None = None
    aligned: bool = False
    merged_concat_axis: CoordinateAxis | None = None

    # A state
    buf_a: HybridAxisArrayBuffer | None = None
    concat_axis_a: CoordinateAxis | None = None
    a_concat_dim: int | None = None
    a_other_dims: tuple[int, ...] | None = None

    # B state
    buf_b: HybridAxisArrayBuffer | None = None
    concat_axis_b: CoordinateAxis | None = None
    b_concat_dim: int | None = None
    b_other_dims: tuple[int, ...] | None = None


class MergeProcessor(BaseStatefulTransformer[MergeSettings, AxisArray, AxisArray | None, MergeState]):
    """Processor that time-aligns two AxisArray streams and concatenates them.

    Input A flows through the standard ``__call__`` / ``_process`` path,
    getting automatic ``_hash_message`` / ``_reset_state`` handling from
    :class:`BaseStatefulTransformer`.  Input B flows through :meth:`push_b`,
    which independently tracks its own structure.

    Invalidation rules:

    - Gain mismatch (either input vs stored common gain) → full reset.
    - Concat-axis dimensionality change → per-input buffer reset +
      alignment and merged-axis cache invalidation.
    - Non-align/non-concat axis shape change → per-input buffer reset +
      alignment invalidation.
    """

    # -- Structural extraction helpers ---------------------------------------

    def _extract_gain(self, message: AxisArray) -> float | None:
        """Extract the align-axis gain from a message."""
        align_name = self.settings.align_axis or message.dims[0]
        ax = message.axes.get(align_name)
        if ax is not None and hasattr(ax, "gain"):
            return ax.gain
        if ax is not None and hasattr(ax, "data") and len(ax.data) > 1:
            return float(ax.data[-1] - ax.data[0]) / (len(ax.data) - 1)
        return None

    # -- Reset helpers -------------------------------------------------------

    def _full_reset(self, align_axis: str) -> None:
        """Reset all state — both inputs and common merge state."""
        self._state.align_axis = align_axis
        self._state.buf_a = HybridAxisArrayBuffer(duration=self.settings.buffer_dur, axis=align_axis)
        self._state.buf_b = HybridAxisArrayBuffer(duration=self.settings.buffer_dur, axis=align_axis)
        self._state.gain = None
        self._state.aligned = False
        self._state.concat_axis_a = None
        self._state.concat_axis_b = None
        self._state.merged_concat_axis = None
        self._state.a_concat_dim = None
        self._state.a_other_dims = None
        self._state.b_concat_dim = None
        self._state.b_other_dims = None

    def _reset_a_state(self) -> None:
        """Reset input-A buffer and concat-axis cache."""
        self._state.buf_a = HybridAxisArrayBuffer(duration=self.settings.buffer_dur, axis=self._state.align_axis)
        self._state.concat_axis_a = None

    def _reset_b_state(self) -> None:
        """Reset input-B buffer and concat-axis cache."""
        self._state.buf_b = HybridAxisArrayBuffer(duration=self.settings.buffer_dur, axis=self._state.align_axis)
        self._state.concat_axis_b = None

    # -- BaseStatefulTransformer interface ------------------------------------

    def _hash_message(self, message: AxisArray) -> int:
        """Hash the align-axis gain only.

        Gain changes trigger a full reset via ``_reset_state``.  Concat-axis
        and non-merge dimension changes are handled as partial resets inside
        ``_process`` and ``push_b``.
        """
        return hash(self._extract_gain(message))

    def _reset_state(self, message: AxisArray) -> None:
        """Full reset — called by the base class when gain changes."""
        align_axis = self.settings.align_axis or message.dims[0]
        self._full_reset(align_axis)

    def _process(self, message: AxisArray) -> AxisArray | None:
        """Process input A: detect structural changes, buffer, try merge."""
        # Detect per-input structural changes.
        align_idx = message.dims.index(self._state.align_axis)
        concat_idx = message.get_axis_idx(self.settings.axis)
        concat_dim = message.data.shape[concat_idx]
        other_dims = tuple(s for i, s in enumerate(message.data.shape) if i != align_idx and i != concat_idx)

        if self._state.a_concat_dim is not None and concat_dim != self._state.a_concat_dim:
            self._reset_a_state()
            self._state.aligned = False
            self._state.merged_concat_axis = None
        elif self._state.a_other_dims is not None and other_dims != self._state.a_other_dims:
            self._reset_a_state()
            self._state.aligned = False

        self._state.a_concat_dim = concat_dim
        self._state.a_other_dims = other_dims

        self._state.buf_a.write(message)
        if self._state.gain is None:
            self._state.gain = self._state.buf_a.axis_gain
        self._update_concat_axis(message, "a")
        return self._try_merge()

    # -- Input B entry point ------------------------------------------------

    def push_b(self, message: AxisArray) -> AxisArray | None:
        """Process input B: check gain, detect structural changes, buffer, try merge."""
        align_axis = self.settings.align_axis or message.dims[0]

        # Gain compatibility check.
        b_gain = self._extract_gain(message)
        if self._state.gain is not None and b_gain != self._state.gain:
            self._full_reset(align_axis)
            # Set the base-class hash so the next compatible A goes straight
            # to _process instead of triggering another full reset.
            self._hash = self._hash_message(message)

        # Lazy-create buf_b if B arrives before A.
        if self._state.buf_b is None:
            if self._state.align_axis is None:
                self._state.align_axis = align_axis
            self._state.buf_b = HybridAxisArrayBuffer(duration=self.settings.buffer_dur, axis=align_axis)

        # Detect per-input structural changes.
        align_idx = message.dims.index(align_axis)
        concat_idx = message.get_axis_idx(self.settings.axis)
        concat_dim = message.data.shape[concat_idx]
        other_dims = tuple(s for i, s in enumerate(message.data.shape) if i != align_idx and i != concat_idx)

        if self._state.b_concat_dim is not None and concat_dim != self._state.b_concat_dim:
            self._reset_b_state()
            self._state.aligned = False
            self._state.merged_concat_axis = None
        elif self._state.b_other_dims is not None and other_dims != self._state.b_other_dims:
            self._reset_b_state()
            self._state.aligned = False

        self._state.b_concat_dim = concat_dim
        self._state.b_other_dims = other_dims

        self._state.buf_b.write(message)
        if self._state.gain is None:
            self._state.gain = self._state.buf_b.axis_gain
        self._update_concat_axis(message, "b")
        return self._try_merge()

    # -- Concat-axis caching ------------------------------------------------

    def _update_concat_axis(self, message: AxisArray, which: str) -> None:
        """Track each input's concat-axis labels; invalidate cache on change."""
        concat_dim = self.settings.axis
        if concat_dim not in message.axes:
            return
        ax = message.axes[concat_dim]
        if not hasattr(ax, "data"):
            return

        if which == "a":
            if self._state.concat_axis_a is None or not np.array_equal(self._state.concat_axis_a.data, ax.data):
                self._state.concat_axis_a = ax
                self._state.merged_concat_axis = None
        else:
            if self._state.concat_axis_b is None or not np.array_equal(self._state.concat_axis_b.data, ax.data):
                self._state.concat_axis_b = ax
                self._state.merged_concat_axis = None

    def _build_merged_concat_axis(self) -> CoordinateAxis | None:
        """Build the merged CoordinateAxis from the two cached per-input axes."""
        if self._state.concat_axis_a is None or self._state.concat_axis_b is None:
            return None
        if self.settings.relabel_axis:
            labels_a = np.array([str(lbl) + self.settings.label_a for lbl in self._state.concat_axis_a.data])
            labels_b = np.array([str(lbl) + self.settings.label_b for lbl in self._state.concat_axis_b.data])
        else:
            labels_a = self._state.concat_axis_a.data
            labels_b = self._state.concat_axis_b.data
        return CoordinateAxis(
            data=np.concatenate([labels_a, labels_b]),
            dims=self._state.concat_axis_a.dims,
            unit=self._state.concat_axis_a.unit,
        )

    # -- Core merge logic ---------------------------------------------------

    def _try_merge(self) -> AxisArray | None:
        """Align and read from both buffers, returning the merged result.

        Initial alignment is performed once.  After the first successful
        merge the two streams are assumed to share a common clock and
        never drop samples, so we simply read
        ``min(available_a, available_b)`` on every subsequent call.
        """
        if self._state.buf_a is None or self._state.buf_b is None:
            return None
        if self._state.buf_a.is_empty() or self._state.buf_b.is_empty():
            return None

        gain = self._state.gain

        # --- Initial alignment (runs only until the first successful merge) ---
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

        return self._concat(aa_a, aa_b)

    def _concat(self, a: AxisArray, b: AxisArray) -> AxisArray:
        """Concatenate *a* and *b* along the configured merge axis."""
        # Use the cached merged axis (rebuilt lazily when labels change).
        if self._state.merged_concat_axis is None:
            self._state.merged_concat_axis = self._build_merged_concat_axis()

        key = self.settings.new_key if self.settings.new_key is not None else a.key
        result = AxisArray.concatenate(a, b, dim=self.settings.axis, axis=self._state.merged_concat_axis)
        if key != result.key:
            result = replace(result, key=key)
        return result


class Merge(BaseProcessorUnit[MergeSettings]):
    """Merge two AxisArray streams by time-aligning and concatenating along a non-time axis.

    Input A routes through the processor's ``__acall__`` (triggering
    hash-based reset when the stream structure changes).  Input B
    routes through ``push_b`` which independently tracks its own structure.

    Inherits ``INPUT_SETTINGS`` and ``on_settings`` → ``create_processor``
    from :class:`BaseProcessorUnit`.
    """

    SETTINGS = MergeSettings

    INPUT_SIGNAL_A = ez.InputStream(AxisArray)
    INPUT_SIGNAL_B = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def create_processor(self) -> None:
        self.processor = MergeProcessor(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SIGNAL_A, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_a(self, msg: AxisArray) -> typing.AsyncGenerator:
        result = await self.processor.__acall__(msg)
        if result is not None:
            yield self.OUTPUT_SIGNAL, result

    @ez.subscriber(INPUT_SIGNAL_B, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_b(self, msg: AxisArray) -> typing.AsyncGenerator:
        result = self.processor.push_b(msg)
        if result is not None:
            yield self.OUTPUT_SIGNAL, result
