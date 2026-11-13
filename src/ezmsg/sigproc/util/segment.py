"""Segment numbering for an output fed by more than one input."""


class SegmentJoin:
    """Numbers the segments of an output that combines several inputs.

    The output starts a new segment whenever any input's
    :attr:`~ezmsg.util.messages.axisarray.AxisArray.segment` changes. Its number is this
    counter's own, starting at 0, not any input's: two inputs' counters cannot be merged
    into one, only their changes can.
    """

    def __init__(self) -> None:
        self.segment = 0
        self._inputs: dict[str, int] = {}

    def update(self, **segments: int) -> bool:
        """Record each named input's current segment; start a new output segment and return True if any changed."""
        changed = any(self._inputs.get(name, seg) != seg for name, seg in segments.items())
        self._inputs.update(segments)
        self.segment += changed
        return changed
