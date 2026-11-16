"""Segment numbering and ``config_id`` for an output fed by more than one input, or reshaped by its stage."""

from ezmsg.baseproc import end_of_stream, resolve_stream_dim
from ezmsg.util.messages.axisarray import AxisArray, chain_config_id
from ezmsg.util.messages.util import replace


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


async def next_pair(state) -> tuple[AxisArray, AxisArray]:
    """The next ``(a, b)`` for a two-input unit to combine, taken from ``state.queue_a`` and ``state.queue_b``.

    Messages pair in arrival order, except one with no samples (an end-of-stream message): that is
    paired with a zero-length copy of the other input's previous message instead, so the unit emits
    one empty output in the new segment. Before the other input has sent anything there is nothing
    to base that output on, and the message only updates ``state.segments``. ``state`` also needs
    ``held_a``, ``last_a`` and ``last_b``, all initially None.
    """
    while True:
        a, state.held_a = state.held_a, None
        if a is None:
            a = await state.queue_a.get()
        if _no_samples(a):
            state.last_a = a
            if state.last_b is not None:
                return a, _zero_length(state.last_b)
            state.segments.update(a=a.segment)
            continue
        b = await state.queue_b.get()
        if _no_samples(b):
            state.held_a, state.last_b = a, b  # a waits for B's next message
            if state.last_a is not None:
                return _zero_length(state.last_a), b
            state.segments.update(b=b.segment)
            continue
        state.last_a, state.last_b = a, b
        return a, b


def _no_samples(msg: AxisArray) -> bool:
    return bool(msg.dims) and msg.data.shape[msg.get_axis_idx(resolve_stream_dim(msg))] == 0


def _zero_length(msg: AxisArray) -> AxisArray:
    """``msg`` with no samples, still in its own segment."""
    return replace(end_of_stream(msg), segment=msg.segment)


class ConfigJoin:
    """Derives the ``config_id`` of an output that combines several inputs, or whose layout its stage changes.

    The id is ``chain_config_id(*config_ids, *own)``, recomputed only when an input's
    :attr:`~ezmsg.util.messages.axisarray.AxisArray.config_id` or ``own`` (whatever the stage
    itself contributes to the output layout) changes. If any input does not declare one (0),
    neither does the output.
    """

    def __init__(self) -> None:
        self.config_id = 0
        self._key: tuple | None = None

    def update(self, *config_ids: int, own: tuple = ()) -> int:
        """Return the output's config_id for these inputs' ids."""
        key = (config_ids, own)
        if key != self._key:
            self._key = key
            self.config_id = chain_config_id(*config_ids, *own) if all(config_ids) else 0
        return self.config_id


class ConfigChain:
    """Derives the ``config_id`` of a single input's output, for a stage whose contribution is fixed.

    The id is ``chain_config_id(config_id, *own)``, recomputed only when the input's
    :attr:`~ezmsg.util.messages.axisarray.AxisArray.config_id` changes, so a steady stream
    costs one comparison per message. An undeclared input (0) gives an undeclared output.
    Build a new one wherever ``own`` changes (for a stateful stage: at reset).
    """

    def __init__(self, *own) -> None:
        self.own = own
        self.config_id = 0
        self._in = 0

    def update(self, config_id: int) -> int:
        """Return the output's config_id for this input id."""
        if config_id != self._in:
            self._in = config_id
            self.config_id = chain_config_id(config_id, *self.own) if config_id else 0
        return self.config_id


class SettingsConfigChain:
    """Mixin for a stateless transformer whose settings decide how it changes the output layout.

    Stamp ``self._chained_config_id(message)`` on the output, and return from
    :meth:`_config_own` what determines that change. The derived id is cached on the input's
    ``config_id`` and dropped on a settings change: ``update_settings`` calls ``_request_reset``,
    otherwise a no-op on a stateless processor.
    """

    _config_in: int = 0
    _config_out: int = 0

    def _config_own(self, message) -> tuple:
        """What this stage contributes to the output's config_id, given the settings and ``message``'s layout."""
        raise NotImplementedError

    def _chained_config_id(self, message) -> int:
        config_id = message.config_id
        if config_id != self._config_in:
            self._config_in = config_id
            self._config_out = chain_config_id(config_id, *self._config_own(message)) if config_id else 0
        return self._config_out

    def _request_reset(self) -> None:
        super()._request_reset()
        self._config_in = 0
        self._config_out = 0
