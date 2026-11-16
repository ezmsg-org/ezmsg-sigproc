"""Segment numbering and ``config_id`` for an output fed by more than one input, or reshaped by its stage."""

from ezmsg.util.messages.axisarray import chain_config_id


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
