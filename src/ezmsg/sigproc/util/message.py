"""Message (AxisArray) utilities.

Also re-exports sample-message symbols from ezmsg.baseproc.util.message for
backwards compatibility; new code should import those directly from
ezmsg.baseproc instead.
"""

import typing

from ezmsg.baseproc.util.message import (
    SampleMessage,
    SampleTriggerMessage,
    is_sample_message,
)
from ezmsg.util.messages.axisarray import AxisArray

__all__ = [
    "SampleMessage",
    "SampleTriggerMessage",
    "has_samples_along",
    "is_empty_along",
    "is_sample_message",
    "with_fingerprint",
]


def with_fingerprint(axis: AxisArray.CoordinateAxis) -> AxisArray.CoordinateAxis:
    """Compute *axis*'s fingerprint now, and return the axis.

    Every stateful consumer reads the fingerprint of the coordinate axes that
    describe a stream's configuration, and the value is cached on the instance
    and pickled with it. Computing it at the point of construction therefore
    pays the checksum once, for everybody:

    * In this process, the axis object is reused for the life of the stream, so
      one call covers every message and every consumer downstream of it.
    * Across a process boundary it is better than that. Unpickling hands out a
      *new* axis object per message, so a cold axis is re-checksummed by the
      first consumer in every receiving process, on every message, forever.
      A primed one arrives with the answer already attached.

    Apply it to axes that describe the stream -- channel labels, frequency
    labels, feature labels -- not to per-message coordinates along the chunk
    dimension, whose fingerprint no consumer reads and whose data is new every
    message anyway.
    """
    axis.fingerprint  # noqa: B018 -- evaluated for the caching side effect
    return axis


def is_empty_along(message: AxisArray, dims: typing.Iterable[str]) -> bool:
    """True iff any of the named dims is present in ``message`` with zero length.

    Publish gates use this instead of ``data.size == 0`` so a message that is
    empty only along *other* axes — e.g. an upstream selection removed every
    channel while time samples remain — still flows downstream, preserving the
    stream's cadence for consumers that align or merge multiple sources.
    Dims not present in the message are ignored.
    """
    return any(d in message.dims and message.data.shape[message.get_axis_idx(d)] == 0 for d in dims)


def has_samples_along(message: AxisArray, dim: str) -> bool:
    """True iff ``dim`` is present in ``message`` with nonzero length.

    Stricter than ``not is_empty_along(...)``: the dim must exist. Drain loops
    use this to decide whether a chunk is real output, so that a placeholder
    lacking the axis entirely (e.g. ResampleProcessor's pre-init null template,
    ``dims=[""]``) counts as "nothing ready" rather than a publishable chunk.
    """
    return dim in message.dims and message.data.shape[message.get_axis_idx(dim)] > 0
