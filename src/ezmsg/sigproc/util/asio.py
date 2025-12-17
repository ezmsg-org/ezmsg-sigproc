"""
Backwards-compatible re-exports from ezmsg.baseproc.util.asio.

New code should import directly from ezmsg.baseproc instead.
"""

import warnings

warnings.warn(
    "Importing from 'ezmsg.sigproc.util.asio' is deprecated. Please import from 'ezmsg.baseproc.util.asio' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ezmsg.baseproc.util.asio import (  # noqa: E402
    CoroutineExecutionError,
    SyncToAsyncGeneratorWrapper,
    run_coroutine_sync,
)

__all__ = [
    "CoroutineExecutionError",
    "SyncToAsyncGeneratorWrapper",
    "run_coroutine_sync",
]
