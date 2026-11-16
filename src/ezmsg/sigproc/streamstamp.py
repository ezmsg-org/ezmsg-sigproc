"""Stamp ``segment`` and ``config_id`` on a stream whose source does not.

A source should set :attr:`~ezmsg.util.messages.axisarray.AxisArray.segment` itself,
changing it on the first message that does not continue the previous one, and
:attr:`~ezmsg.util.messages.axisarray.AxisArray.config_id`, changing it when its layout
changes. Place :obj:`StreamStamp` right after a source that does not, so that every
stateful stage downstream resets across a gap instead of filtering through it, and
compares one int per message instead of inspecting the layout.
"""

import ezmsg.core as ez
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    SegmentTracker,
    processor_state,
    resolve_stream_dim,
)
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis, compute_config_id
from ezmsg.util.messages.util import replace


class StreamStampSettings(ez.Settings):
    tolerance: float | None
    """Largest difference, in seconds, between a chunk's offset and where the previous chunk
    ended that still counts as continuous.

    There is no default because it depends on the source. Two chunks' offsets can each be off
    by the source's timestamp jitter, so allow about twice that. Err on the large side: a
    timing slip of a sample or two costs a filter far less than the reset a reported gap
    causes downstream.

    ``None`` turns detection off: every message keeps the ``segment`` it arrived with,
    though it still gets a ``config_id``. Can be changed while running."""


@processor_state
class StreamStampState:
    tracker: SegmentTracker | None = None
    upstream: int | None = None
    """The upstream ``segment`` of the previous message."""
    config_id: int = 0
    """``compute_config_id`` of the current layout, for messages that do not declare one."""


class StreamStampTransformer(BaseStatefulTransformer[StreamStampSettings, AxisArray, AxisArray, StreamStampState]):
    """Start a new ``segment`` wherever the stream's offsets show a gap, and stamp a ``config_id``.

    Each chunk's stream-axis offset is compared with where the previous chunk ended; a jump
    forward or backward of more than ``tolerance`` seconds starts a new segment, and so does a
    change of the upstream ``segment`` or of the stream's layout (e.g. its sample rate). The
    output ``segment`` is this transformer's own counter, starting at 0, not the upstream one.

    Only gaps *between* chunks are seen: a chunk's ``LinearAxis`` cannot describe a gap within
    it. Split such chunks first (see :func:`ezmsg.baseproc.find_gaps`).

    Gap detection needs a ``LinearAxis`` on the stream dimension. A message whose stream axis
    is a ``CoordinateAxis`` keeps the ``segment`` it arrived with.

    A message that declares a ``config_id`` keeps it. Any other gets
    :func:`~ezmsg.util.messages.axisarray.compute_config_id` of its layout, computed only when
    the layout changes.
    """

    # Read live in `_process`: changing it must not count as a break.
    NONRESET_SETTINGS_FIELDS = frozenset({"tolerance"})
    # Receiving messages without a config_id is this stage's job.
    WARN_UNDECLARED_CONFIG = False

    def _hash_message(self, message: AxisArray) -> int:
        # The stamped id must change whenever `compute_config_id` would: add what the default hash leaves out.
        units = tuple(getattr(ax, "unit", None) for ax in message.axes.values())
        return self._message_hash(message, extra=(str(message.data.dtype), message.stream_dim, units))

    def _reset_state(self, message: AxisArray) -> None:
        # Computed even for a declared stream: it may stop declaring without changing layout.
        self._state.config_id = compute_config_id(message)
        if self._state.tracker is not None:
            # A new layout is a break too; keep counting, since restarting at 0 could repeat the last number.
            self._state.tracker.bump()

    def _reset_segment(self, message: AxisArray) -> None:
        # An upstream break is counted in `_process`; the default full reset would lose the count.
        pass

    def _process(self, message: AxisArray) -> AxisArray:
        config_id = message.config_id or self._state.config_id
        if self.settings.tolerance is None:
            # Off. Forget the count too: on resuming, the last offset seen would be stale.
            self._state.tracker = self._state.upstream = None
            return replace(message, config_id=config_id)
        dim = resolve_stream_dim(message)
        axis = message.axes.get(dim)
        if not isinstance(axis, LinearAxis):
            return replace(message, config_id=config_id)
        if self._state.tracker is None:
            self._state.tracker = SegmentTracker(self.settings.tolerance)
        tracker = self._state.tracker
        tracker.tolerance = self.settings.tolerance
        if self._state.upstream is not None and message.segment != self._state.upstream:
            tracker.bump()
        self._state.upstream = message.segment
        segment = tracker.update(axis.offset, message.data.shape[message.get_axis_idx(dim)], axis.gain)
        return replace(message, segment=segment, config_id=config_id)


class StreamStamp(BaseTransformerUnit[StreamStampSettings, AxisArray, AxisArray, StreamStampTransformer]):
    SETTINGS = StreamStampSettings
