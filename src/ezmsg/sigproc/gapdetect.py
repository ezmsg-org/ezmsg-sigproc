"""Number the segments of a stream whose source does not.

A source should set :attr:`~ezmsg.util.messages.axisarray.AxisArray.segment` itself,
changing it on the first message that does not continue the previous one. Place
:obj:`GapDetect` right after a source that does not, so that every stateful stage
downstream resets across a gap instead of filtering through it.
"""

import ezmsg.core as ez
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    SegmentTracker,
    processor_state,
    resolve_stream_dim,
)
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis
from ezmsg.util.messages.util import replace


class GapDetectSettings(ez.Settings):
    tolerance: float | None
    """Largest difference, in seconds, between a chunk's offset and where the previous chunk
    ended that still counts as continuous.

    There is no default because it depends on the source. Two chunks' offsets can each be off
    by the source's timestamp jitter, so allow about twice that. Err on the large side: a
    timing slip of a sample or two costs a filter far less than the reset a reported gap
    causes downstream.

    ``None`` turns detection off: every message passes through untouched, keeping the
    ``segment`` it arrived with. Can be changed while running."""


@processor_state
class GapDetectState:
    tracker: SegmentTracker | None = None
    upstream: int | None = None
    """The upstream ``segment`` of the previous message."""


class GapDetectTransformer(BaseStatefulTransformer[GapDetectSettings, AxisArray, AxisArray, GapDetectState]):
    """Start a new ``segment`` wherever the stream's offsets show a gap.

    Each chunk's stream-axis offset is compared with where the previous chunk ended; a jump
    forward or backward of more than ``tolerance`` seconds starts a new segment, and so does a
    change of the upstream ``segment`` or of the stream's layout (e.g. its sample rate). The
    output ``segment`` is this transformer's own counter, starting at 0, not the upstream one.

    Only gaps *between* chunks are seen: a chunk's ``LinearAxis`` cannot describe a gap within
    it. Split such chunks first (see :func:`ezmsg.baseproc.find_gaps`).

    Needs a ``LinearAxis`` on the stream dimension. A message whose stream axis is a
    ``CoordinateAxis`` is passed through untouched, its ``segment`` included.
    """

    # Read live in `_process`: changing it must not count as a break.
    NONRESET_SETTINGS_FIELDS = frozenset({"tolerance"})

    def _reset_state(self, message: AxisArray) -> None:
        if self._state.tracker is not None:
            # A new layout is a break too; keep counting, since restarting at 0 could repeat the last number.
            self._state.tracker.bump()

    def _reset_segment(self, message: AxisArray) -> None:
        # An upstream break is counted in `_process`; the default full reset would lose the count.
        pass

    def _process(self, message: AxisArray) -> AxisArray:
        dim = resolve_stream_dim(message)
        axis = message.axes.get(dim)
        if self.settings.tolerance is None:
            # Off. Forget the count too: on resuming, the last offset seen would be stale.
            self._state.tracker = self._state.upstream = None
            return message
        if not isinstance(axis, LinearAxis):
            return message
        if self._state.tracker is None:
            self._state.tracker = SegmentTracker(self.settings.tolerance)
        tracker = self._state.tracker
        tracker.tolerance = self.settings.tolerance
        if self._state.upstream is not None and message.segment != self._state.upstream:
            tracker.bump()
        self._state.upstream = message.segment
        segment = tracker.update(axis.offset, message.data.shape[message.get_axis_idx(dim)], axis.gain)
        return replace(message, segment=segment)


class GapDetect(BaseTransformerUnit[GapDetectSettings, AxisArray, AxisArray, GapDetectTransformer]):
    SETTINGS = GapDetectSettings
