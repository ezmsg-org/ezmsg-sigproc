"""
Backwards-compatible re-exports from ezmsg.baseproc.util.typeresolution.

New code should import directly from ezmsg.baseproc instead.
"""

from ezmsg.baseproc.util.typeresolution import (
    TypeLike,
    check_message_type_compatibility,
    resolve_typevar,
)

__all__ = [
    "TypeLike",
    "check_message_type_compatibility",
    "resolve_typevar",
]
