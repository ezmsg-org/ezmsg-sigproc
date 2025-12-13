"""
Backwards-compatible re-exports from ezmsg.baseproc.util.asio.

New code should import directly from ezmsg.baseproc instead.
"""

from ezmsg.baseproc.util.asio import (
    CoroutineExecutionError,
    SyncToAsyncGeneratorWrapper,
    run_coroutine_sync,
)

__all__ = [
    "CoroutineExecutionError",
    "SyncToAsyncGeneratorWrapper",
    "run_coroutine_sync",
]
