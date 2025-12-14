"""
Backwards-compatible re-exports from ezmsg.baseproc.util.message.

New code should import directly from ezmsg.baseproc instead.
"""

from ezmsg.baseproc.util.message import (
    SampleMessage,
    SampleTriggerMessage,
    is_sample_message,
)

__all__ = [
    "SampleMessage",
    "SampleTriggerMessage",
    "is_sample_message",
]
