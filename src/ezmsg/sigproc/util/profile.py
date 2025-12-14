"""
Backwards-compatible re-exports from ezmsg.baseproc.util.profile.

New code should import directly from ezmsg.baseproc instead.
"""

from ezmsg.baseproc.util.profile import (
    HEADER,
    _setup_logger,
    get_logger_path,
    logger,
    profile_method,
    profile_subpub,
)

__all__ = [
    "HEADER",
    "get_logger_path",
    "logger",
    "profile_method",
    "profile_subpub",
    "_setup_logger",
]
