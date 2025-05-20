import asyncio  # noqa: F401
import time

import numpy as np
import pytest

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.synth import (
    clock,
    aclock,
    acounter,
    sin,
)


# TEST CLOCK
@pytest.mark.parametrize("dispatch_rate", [None, 1.0, 2.0, 5.0, 10.0, 20.0])
def test_clock_gen(dispatch_rate: float | None):
    run_time = 1.0
    n_target = int(np.ceil(dispatch_rate * run_time)) if dispatch_rate else 100
    gen = clock(dispatch_rate=dispatch_rate)
    result = []
    t_start = time.time()
    while len(result) < n_target:
        result.append(next(gen))
    t_elapsed = time.time() - t_start
    assert all([_ == ez.Flag() for _ in result])
    if dispatch_rate is not None:
        assert (run_time - 1 / dispatch_rate) < t_elapsed < (run_time + 0.2)
    else:
        # 100 usec per iteration is pretty generous
        assert t_elapsed < (n_target * 1e-4)


@pytest.mark.parametrize("dispatch_rate", [None, 2.0, 20.0])
@pytest.mark.asyncio
async def test_aclock_agen(dispatch_rate: float | None):
    run_time = 1.0
    n_target = int(np.ceil(dispatch_rate * run_time)) if dispatch_rate else 100
    agen = aclock(dispatch_rate=dispatch_rate)
    result = []
    t_start = time.time()
    while len(result) < n_target:
        new_result = await agen.__anext__()
        result.append(new_result)
    t_elapsed = time.time() - t_start
    assert all([_ == ez.Flag() for _ in result])
    if dispatch_rate:
        assert (run_time - 1.1 / dispatch_rate) < t_elapsed < (run_time + 0.1)
    else:
        # 100 usec per iteration is pretty generous
        assert t_elapsed < (n_target * 1e-4)


@pytest.mark.parametrize("block_size", [1, 20])
@pytest.mark.parametrize("fs", [10.0, 1000.0])
@pytest.mark.parametrize("n_ch", [3])
@pytest.mark.parametrize(
    "dispatch_rate", [None, "realtime", "ext_clock", 2.0, 20.0]
)  # "ext_clock" needs a separate test
@pytest.mark.parametrize("mod", [2**3, None])
@pytest.mark.asyncio
async def test_acounter(
    block_size: int,
    fs: float,
    n_ch: int,
    dispatch_rate: float | str | None,
    mod: int | None,
):
    target_dur = 2.6  # 2.6 seconds per test
    if dispatch_rate is None:
        # No sleep / wait
        chunk_dur = 0.1
    elif isinstance(dispatch_rate, str):
        if dispatch_rate == "realtime":
            chunk_dur = block_size / fs
        elif dispatch_rate == "ext_clock":
            # No sleep / wait
            chunk_dur = 0.1
    else:
        # Note: float dispatch_rate will yield different number of samples than expected by target_dur and fs
        chunk_dur = 1.0 / dispatch_rate
    target_messages = int(target_dur / chunk_dur)

    # Run generator
    agen = acounter(block_size, fs, n_ch=n_ch, dispatch_rate=dispatch_rate, mod=mod)
    messages = [await agen.__anext__() for _ in range(target_messages)]

    # Test contents of individual messages
    for msg in messages:
        assert type(msg) is AxisArray
        assert msg.data.shape == (block_size, n_ch)
        assert "time" in msg.axes
        assert msg.axes["time"].gain == 1 / fs
        assert "ch" in msg.axes
        assert np.array_equal(
            msg.axes["ch"].data, np.array([f"Ch{_}" for _ in range(n_ch)])
        )

    agg = AxisArray.concatenate(*messages, dim="time")

    target_samples = block_size * target_messages
    expected_data = np.arange(target_samples)
    if mod is not None:
        expected_data = expected_data % mod
    assert np.array_equal(agg.data[:, 0], expected_data)

    offsets = np.array([m.axes["time"].offset for m in messages])
    expected_offsets = np.arange(target_messages) * block_size / fs
    if dispatch_rate == "realtime" or dispatch_rate == "ext_clock":
        expected_offsets += offsets[0]  # offsets are in real-time
        atol = 0.002
    else:
        # Offsets are synthetic.
        atol = 1.0e-8
    assert np.allclose(offsets[2:], expected_offsets[2:], atol=atol)


# TEST SIN #
def test_sin_gen(freq: float = 1.0, amp: float = 1.0, phase: float = 0.0):
    axis: str | None = "time"
    srate = max(4.0 * freq, 1000.0)
    sim_dur = 30.0
    n_samples = int(srate * sim_dur)
    n_msgs = min(n_samples, 10)
    axis_idx = 0

    messages = []
    for split_dat in np.array_split(
        np.arange(n_samples)[:, None], n_msgs, axis=axis_idx
    ):
        _time_axis = AxisArray.TimeAxis(fs=srate, offset=float(split_dat[0, 0]))
        messages.append(
            AxisArray(split_dat, dims=["time", "ch"], axes={"time": _time_axis})
        )

    def f_test(t):
        return amp * np.sin(2 * np.pi * freq * t + phase)

    gen = sin(axis=axis, freq=freq, amp=amp, phase=phase)
    results = []
    for msg in messages:
        res = gen.send(msg)
        assert np.allclose(res.data, f_test(msg.data / srate))
        results.append(res)
    concat_ax_arr = AxisArray.concatenate(*results, dim="time")
    assert np.allclose(
        concat_ax_arr.data, f_test(np.arange(n_samples) / srate)[:, None]
    )
