from typing import TypeVar, Coroutine, Any
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import contextlib

T = TypeVar("T")


class CoroutineExecutionError(Exception):
    """Custom exception for coroutine execution failures"""

    pass


def run_coroutine_sync(coroutine: Coroutine[Any, Any, T], timeout: float = 30) -> T:
    """
    Executes an asyncio coroutine synchronously, with enhanced error handling.

    Args:
        coroutine: The asyncio coroutine to execute
        timeout: Maximum time in seconds to wait for coroutine completion (default: 30)

    Returns:
        The result of the coroutine execution

    Raises:
        CoroutineExecutionError: If execution fails due to threading or event loop issues
        TimeoutError: If execution exceeds the timeout period
        Exception: Any exception raised by the coroutine
    """

    def run_in_new_loop() -> T:
        """
        Creates and runs a new event loop in the current thread.
        Ensures proper cleanup of the loop.
        """
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(
                asyncio.wait_for(coroutine, timeout=timeout)
            )
        finally:
            with contextlib.suppress(Exception):
                # Clean up any pending tasks
                pending = asyncio.all_tasks(new_loop)
                for task in pending:
                    task.cancel()
                new_loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            new_loop.close()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(asyncio.wait_for(coroutine, timeout=timeout))
        except Exception as e:
            raise CoroutineExecutionError(
                f"Failed to execute coroutine: {str(e)}"
            ) from e

    if threading.current_thread() is threading.main_thread():
        if not loop.is_running():
            try:
                return loop.run_until_complete(
                    asyncio.wait_for(coroutine, timeout=timeout)
                )
            except Exception as e:
                raise CoroutineExecutionError(
                    f"Failed to execute coroutine in main loop: {str(e)}"
                ) from e
        else:
            with ThreadPoolExecutor() as pool:
                try:
                    future = pool.submit(run_in_new_loop)
                    return future.result(timeout=timeout)
                except Exception as e:
                    raise CoroutineExecutionError(
                        f"Failed to execute coroutine in thread: {str(e)}"
                    ) from e
    else:
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            return future.result(timeout=timeout)
        except Exception as e:
            raise CoroutineExecutionError(
                f"Failed to execute coroutine threadsafe: {str(e)}"
            ) from e
