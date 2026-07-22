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
]


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
