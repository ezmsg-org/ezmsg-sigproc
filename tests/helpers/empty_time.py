"""Utilities for testing length-0 time dimension message handling.

Many processors in a signal processing graph may receive messages with length-0
in the "time" dimension when insufficient data has been received upstream to
produce output. These messages should propagate through the graph without error,
preserving metadata and allowing downstream nodes to initialize.
"""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

FS = 100.0
N_CH = 3
N_TIME = 30


def make_msg(n_time: int = N_TIME, n_ch: int = N_CH, fs: float = FS) -> AxisArray:
    """Create a test AxisArray with shape (n_time, n_ch)."""
    data = np.random.randn(n_time, n_ch).astype(np.float64)
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=fs),
                "ch": AxisArray.CoordinateAxis(data=np.arange(n_ch).astype(str), dims=["ch"]),
            }
        ),
    )


def make_empty_msg(n_ch: int = N_CH, fs: float = FS) -> AxisArray:
    """Create a test AxisArray with 0 samples in the time dimension."""
    return make_msg(n_time=0, n_ch=n_ch, fs=fs)


def check_empty_result(result: AxisArray, time_dim: str = "time"):
    """Assert that result has 0 in the time dimension and data is empty."""
    if time_dim in result.dims:
        time_idx = result.dims.index(time_dim)
        assert (
            result.data.shape[time_idx] == 0
        ), f"Expected 0 samples in '{time_dim}' dimension, got shape {result.data.shape}"
    else:
        assert result.data.size == 0, f"Expected empty data, got shape {result.data.shape}"


def check_state_not_corrupted(proc, normal_msg: AxisArray, time_dim: str = "time"):
    """Verify that a processor can still handle normal messages after an empty one."""
    result = proc(normal_msg)
    time_idx = result.dims.index(time_dim) if time_dim in result.dims else 0
    assert (
        result.data.shape[time_idx] > 0
    ), f"Expected non-empty output after empty message, got shape {result.data.shape}"
    assert np.all(np.isfinite(result.data)), "Output contains NaN or Inf after empty message"
