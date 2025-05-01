import time
import typing
from dataclasses import dataclass, field

from ezmsg.util.messages.axisarray import AxisArray


@dataclass(unsafe_hash=True)
class SampleTriggerMessage:
    timestamp: float = field(default_factory=time.time)
    """Time of the trigger, in seconds. The Clock depends on the input but defaults to time.time"""

    period: tuple[float, float] | None = None
    """The period around the timestamp, in seconds"""

    value: typing.Any = None
    """A value or 'label' associated with the trigger."""


@dataclass
class SampleMessage:
    trigger: SampleTriggerMessage
    """The time, window, and value (if any) associated with the trigger."""

    sample: AxisArray
    """The data sampled around the trigger."""


def is_sample_message(message: typing.Any) -> typing.TypeGuard[SampleMessage]:
    """Check if the message is a SampleMessage."""
    return hasattr(message, "trigger")
