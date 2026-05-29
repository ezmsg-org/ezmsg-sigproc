import asyncio

import numpy as np
import pytest
import scipy.interpolate
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.chunker import array_chunker

from ezmsg.sigproc.resample import ResampleProcessor
from tests.helpers.util import requires_mlx


@pytest.fixture
def irregular_messages() -> list[AxisArray]:
    """
    10.2 seconds of 128 Hz (jittery intervals) data split unevenly
    into 10 messages + a duplicate after the first message.
    """
    nch = 3
    avg_fs = 128.0
    dur = 10.2
    ntimes = int(avg_fs * dur)
    tvec = np.arange(ntimes) / avg_fs
    np.random.seed(42)  # For reproducibility
    tvec += np.random.normal(0, 0.2 / avg_fs, ntimes)
    tvec = np.sort(tvec)
    n_msgs = 10
    splits = np.sort(np.random.choice(np.arange(ntimes), n_msgs - 1))
    splits = np.hstack(([0], splits, [ntimes]))  # Ensure we have the borders.
    # Ensure we have a duplicate after the first message
    splits = np.insert(splits, 1, [splits[1]])
    msgs = []
    ch_ax = AxisArray.CoordinateAxis(data=np.arange(nch).astype(str), dims=["ch"], unit="label")
    for msg_ix, split in enumerate(splits[:-1]):
        split_tvec = tvec[split : splits[msg_ix + 1]]
        data = np.random.randn(len(split_tvec), nch)
        msgs.append(
            AxisArray(
                data=data,
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.CoordinateAxis(data=split_tvec, dims=["time"], unit="s"),
                    "ch": ch_ax,
                },
                key="irregular_messages",
            )
        )
    return msgs


@pytest.fixture
def reference_messages() -> list[AxisArray]:
    """10 seconds of data at 500 Hz, split evenly into 10 messages."""
    nch = 1
    fs = 500.0
    dur = 10.0
    ntimes = int(fs * dur)
    n_msgs = 10
    data = np.arange(np.prod((ntimes, nch))).reshape(ntimes, nch)
    return list(array_chunker(data, chunk_len=ntimes // n_msgs, axis=0, fs=fs, tzero=0.0))


@pytest.mark.asyncio
@pytest.mark.parametrize("resample_rate", [None, 128.0])
async def test_resample(irregular_messages, reference_messages: list[AxisArray], resample_rate: float | None):
    ref_cat = AxisArray.concatenate(*reference_messages, dim="time")
    # Calculate expected. This only works if interp1d `kind` is "nearest" or "linear"
    x = np.hstack([_.axes["time"].data for _ in irregular_messages])
    y = np.concatenate([_.data for _ in irregular_messages], axis=0)
    if resample_rate is None:
        newx = ref_cat.axes["time"].value(np.arange(ref_cat.data.shape[0]))
    else:
        newx = np.arange(
            irregular_messages[0].axes["time"].data[0],
            irregular_messages[-1].axes["time"].data[-1],
            1 / resample_rate,
        )
    f = scipy.interpolate.interp1d(x, y, axis=0, kind="linear", fill_value="extrapolate")
    expected_data = f(newx)

    resample = ResampleProcessor(resample_rate=resample_rate, buffer_duration=4.0)
    results = []
    n_returned = 0
    for msg_ix, msg in enumerate(irregular_messages):
        if resample_rate is None and len(reference_messages) > msg_ix:
            resample.push_reference(reference_messages[msg_ix])
        resample(msg)
        result = next(resample)
        msg_len = result.data.shape[0]
        b_match = np.allclose(result.data, expected_data[n_returned : n_returned + msg_len])
        assert b_match, f"Message {msg_ix} data mismatch."
        results.append(result)
        n_returned += result.data.shape[0]

    result_cat = AxisArray.concatenate(*results, dim="time")
    if resample_rate is None:
        assert result_cat.axes["time"].offset == ref_cat.axes["time"].offset
        assert result_cat.axes["time"].gain == ref_cat.axes["time"].gain
    else:
        assert np.allclose(
            [result_cat.axes["time"].offset],
            [irregular_messages[0].axes["time"].data[0]],
        )
        assert result_cat.axes["time"].gain == 1 / resample_rate

    assert np.allclose(result_cat.data, expected_data)


@pytest.mark.asyncio
async def test_resample_project(irregular_messages):
    new_rate = 128.0
    resample = ResampleProcessor(resample_rate=new_rate, max_chunk_delay=0.1, fill_value="last")
    results = []
    n_returned = 0
    for msg_ix, msg in enumerate(irregular_messages[:-1]):
        resample(msg)
        results.append(next(resample))
        n_returned += results[-1].data.shape[0]

    # Sleep for a bit then get the next **projected** result.
    await asyncio.sleep(0.2)
    result = next(resample)
    print(result)

    # Now send the last message, which will overlap with the projected result,
    #  so some of it will be ignored and dropped!
    resample(irregular_messages[-1])
    result = next(resample)
    print(result)


@requires_mlx
def test_resample_preserves_mlx_backend():
    """ResampleProcessor must keep the source's array namespace.

    scipy.interpolate.interp1d always returns numpy, so without the
    namespace coercion an MLX input silently downgrades to numpy and a
    downstream same-backend merge then fails with a cryptic error.
    """
    import mlx.core as mx
    from array_api_compat import get_namespace

    np.random.seed(0)
    fs = 128.0
    n = 200
    tvec = np.sort(np.arange(n) / fs + np.random.normal(0, 0.001, n))
    ch_ax = AxisArray.CoordinateAxis(data=np.arange(3).astype(str), dims=["ch"], unit="label")

    def _mk(a: int, b: int) -> AxisArray:
        return AxisArray(
            data=mx.array(np.random.randn(b - a, 3).astype(np.float32)),
            dims=["time", "ch"],
            axes={
                "time": AxisArray.CoordinateAxis(data=tvec[a:b], dims=["time"], unit="s"),
                "ch": ch_ax,
            },
            key="mlx_stream",
        )

    resample = ResampleProcessor(resample_rate=100.0, buffer_duration=4.0)
    mlx_ns = get_namespace(mx.array([1.0]))
    seen_output = False
    for i in range(0, n, 40):
        resample(_mk(i, min(i + 40, n)))
        result = next(resample)
        if result.data.shape[0] == 0:
            continue
        seen_output = True
        assert isinstance(result.data, mx.array), (
            f"Expected mx.array out, got {type(result.data).__name__}"
        )
        assert get_namespace(result.data) is mlx_ns

    assert seen_output, "Resampler never produced a non-empty output to check."


@pytest.mark.asyncio
async def test_resample_no_input():
    """Test calling next() on ResampleProcessor before receiving any input messages."""
    # Create a ResampleProcessor with a specific rate
    resample = ResampleProcessor(resample_rate=128.0)

    null = next(resample)
    assert np.prod(null.data.shape) == 0
