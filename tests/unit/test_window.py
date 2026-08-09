import copy
from dataclasses import replace

import numpy as np
import pytest
import sparse
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.window import WindowSettings, WindowTransformer
from tests.helpers.empty_time import check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg
from tests.helpers.util import assert_messages_equal, calculate_expected_windows, requires_mlx


def test_window_gen_nodur():
    """
    Test window generator method when window_dur is None. Should be a simple pass through.
    """
    nchans = 64
    data_len = 20
    data = np.arange(nchans * data_len, dtype=float).reshape((nchans, data_len))
    test_msg = AxisArray(
        data=data,
        dims=["ch", "time"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=500.0, offset=0.0),
                "ch": AxisArray.CoordinateAxis(data=np.arange(nchans).astype(str), unit="label", dims=["ch"]),
            }
        ),
        key="test_window_gen_nodur",
    )
    backup = [copy.deepcopy(test_msg)]
    proc = WindowTransformer(window_dur=None)
    result = proc(test_msg)
    assert_messages_equal([test_msg], backup)
    assert result is test_msg
    assert np.shares_memory(result.data, test_msg.data)


@pytest.mark.parametrize("msg_block_size", [60, 1, 5, 10, 100])
@pytest.mark.parametrize("newaxis", ["win", None])
@pytest.mark.parametrize("win_dur", [0.3, 1.0])
@pytest.mark.parametrize("win_shift", [0.2, 1.0, None])
@pytest.mark.parametrize("zero_pad", ["input", "shift", "none"])
@pytest.mark.parametrize("fs", [100.0, 500.0])
@pytest.mark.parametrize("anchor", ["beginning", "middle", "end"])
@pytest.mark.parametrize("time_ax", [0, 1])
def test_window_generator(
    msg_block_size: int,
    newaxis: str | None,
    win_dur: float,
    win_shift: float | None,
    zero_pad: str,
    fs: float,
    anchor: str,
    time_ax: int,
):
    nchans = 5

    shift_len = int(win_shift * fs) if win_shift is not None else None
    win_len = int(win_dur * fs)
    data_len = 2 * max(win_len, msg_block_size)
    if win_shift is not None:
        data_len += shift_len - 1
    tvec = np.arange(data_len) / fs
    data = np.arange(nchans * data_len, dtype=float).reshape((nchans, data_len))
    # Below, we transpose the individual messages if time_ax == 0.

    # Instantiate the processor
    proc = WindowTransformer(
        axis="time",
        newaxis=newaxis,
        window_dur=win_dur,
        window_shift=win_shift,
        zero_pad_until=zero_pad,
        anchor=anchor,
    )

    # Create inputs
    template_msg = AxisArray(
        data[..., ()],
        dims=["ch", "time"] if time_ax == 1 else ["time", "ch"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=fs, offset=0.0),
                "ch": AxisArray.CoordinateAxis(data=np.arange(nchans).astype(str), unit="label", dims=["ch"]),
            }
        ),
        key="test_window_generator",
    )
    n_msgs = int(np.ceil(data_len / msg_block_size))
    in_msgs = []
    for msg_ix in range(n_msgs):
        msg_data = data[..., msg_ix * msg_block_size : (msg_ix + 1) * msg_block_size]
        if time_ax == 0:
            msg_data = np.ascontiguousarray(msg_data.T)
        in_msgs.append(
            replace(
                template_msg,
                data=msg_data,
                axes={
                    **template_msg.axes,
                    "time": replace(template_msg.axes["time"], offset=tvec[msg_ix * msg_block_size]),
                },
            )
        )
    backup = copy.deepcopy(in_msgs)

    # Do the actual processing.
    out_msgs = [proc(_) for _ in in_msgs]

    assert_messages_equal(in_msgs, backup)

    # Post-process the results to yield a single data array and a single vector of offsets.
    win_ax = time_ax
    # time_ax = win_ax + 1

    # Check each return value's metadata (offsets checked at end)
    expected_dims = template_msg.dims[:time_ax] + [newaxis or "win"] + template_msg.dims[time_ax:]
    for msg in out_msgs:
        assert msg.axes["time"].gain == 1 / fs
        assert msg.dims == expected_dims
        assert (newaxis or "win") in msg.axes
        assert msg.axes[(newaxis or "win")].gain == (0.0 if win_shift is None else shift_len / fs)

    result = np.concatenate([_.data for _ in out_msgs], win_ax)
    offsets = np.hstack([_.axes[newaxis or "win"].value(np.arange(_.data.shape[win_ax])) for _ in out_msgs])

    # Calculate the expected results for comparison.
    expected, tvec = calculate_expected_windows(
        data,
        fs,
        win_shift,
        zero_pad,
        anchor,
        msg_block_size,
        shift_len,
        win_len,
        nchans,
        data_len,
        n_msgs,
        win_ax,
    )

    # Compare results to expected
    if win_shift is None:
        assert len(out_msgs) == len(in_msgs)
    assert np.allclose(result, expected)
    assert np.allclose(offsets, tvec)


@pytest.mark.parametrize("win_dur", [0.3, 1.0])
@pytest.mark.parametrize("win_shift", [0.2, 1.0, None])
@pytest.mark.parametrize("zero_pad", ["input", "shift", "none"])
def test_sparse_window(
    win_dur: float,
    win_shift: float | None,
    zero_pad: str,
):
    msg_block_size = 60
    fs = 100.0
    nchans = 5

    # Create sparse data
    shift_len = int(win_shift * fs) if win_shift is not None else None
    win_len = int(win_dur * fs)
    data_len = 2 * max(win_len, msg_block_size)
    if win_shift is not None:
        data_len += shift_len - 1
    tvec = np.arange(data_len) / fs
    rng = np.random.default_rng()
    s = sparse.random((data_len, nchans), density=0.1, random_state=rng) > 0

    # Create WindowTransformer
    proc = WindowTransformer(
        axis="time",
        newaxis="win",
        window_dur=win_dur,
        window_shift=win_shift,
        zero_pad_until=zero_pad,
        anchor="beginning",
    )

    template_msg = AxisArray(
        data=s[:0],
        dims=["time", "ch"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=fs, offset=0.0),
                "ch": AxisArray.CoordinateAxis(data=np.arange(nchans).astype(str), unit="label", dims=["ch"]),
            }
        ),
        key="test_sparse_window",
    )
    n_msgs = int(np.ceil(data_len / msg_block_size))
    in_msgs = [
        replace(
            template_msg,
            data=s[msg_ix * msg_block_size : (msg_ix + 1) * msg_block_size],
            axes={
                **template_msg.axes,
                "time": replace(template_msg.axes["time"], offset=tvec[msg_ix * msg_block_size]),
            },
        )
        for msg_ix in range(n_msgs)
    ]

    # Process messages
    out_msgs = [proc(_) for _ in in_msgs]

    # Assert per-message shape and collect total number of windows and window time vector
    nwins = 0
    win_tvec = []
    for om in out_msgs:
        assert om.dims == ["win", "time", "ch"]
        assert om.data.shape[1] == win_len
        assert om.data.shape[2] == nchans
        nwins += om.data.shape[0]
        win_tvec.append(om.axes["win"].value(np.arange(om.data.shape[0])))
    win_tvec = np.hstack(win_tvec)

    # Calculate the expected time vector; note this method expects data time axis to be last.
    _, expected_tvec = calculate_expected_windows(
        np.arange(nchans * data_len).reshape((nchans, data_len)),
        fs,
        win_shift,
        zero_pad,
        "beginning",
        msg_block_size,
        shift_len,
        win_len,
        nchans,
        data_len,
        n_msgs,
        0,
    )

    assert nwins == len(expected_tvec)
    assert np.allclose(win_tvec, expected_tvec)


def test_window_empty_passthrough():
    from ezmsg.sigproc.window import WindowTransformer

    proc = WindowTransformer(window_dur=None)
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_window_empty_first():
    from ezmsg.sigproc.window import WindowSettings, WindowTransformer

    proc = WindowTransformer(
        WindowSettings(axis="time", newaxis="win", window_dur=0.1, window_shift=0.05, zero_pad_until="shift")
    )
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    assert result.data.size >= 0  # Just check no crash
    check_state_not_corrupted(proc, normal, time_dim="time")


def test_window_empty_with_shift():
    from ezmsg.sigproc.window import WindowSettings, WindowTransformer

    proc = WindowTransformer(
        WindowSettings(axis="time", newaxis="win", window_dur=0.1, window_shift=0.05, zero_pad_until="shift")
    )
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    assert result.data.size >= 0  # Just check no crash
    check_state_not_corrupted(proc, normal, time_dim="time")


@requires_mlx
@pytest.mark.benchmark(group="window")
@pytest.mark.parametrize("n_channels", [32, 256, 1024])
@pytest.mark.parametrize("backend", ["mlx", "numpy"])
def test_window_benchmark(backend, n_channels, benchmark):
    """Benchmark WindowTransformer: numpy vs MLX input."""
    fs = 1000.0
    chunk_samples = 256
    n_chunks = 20
    window_dur = 0.5
    window_shift = 0.1

    rng = np.random.default_rng(42)
    chunks = []
    for i in range(n_chunks):
        d = rng.standard_normal((chunk_samples, n_channels)).astype(np.float32)
        _time_axis = AxisArray.TimeAxis(fs=fs, offset=i * chunk_samples / fs)
        axes = frozendict(
            {
                "time": _time_axis,
                "ch": AxisArray.CoordinateAxis(data=np.arange(n_channels).astype(str), dims=["ch"]),
            }
        )
        if backend == "mlx":
            import mlx.core as mx

            chunks.append(AxisArray(mx.array(d), dims=["time", "ch"], axes=axes, key="bench"))
        else:
            chunks.append(AxisArray(d, dims=["time", "ch"], axes=axes, key="bench"))

    xformer = WindowTransformer(
        axis="time",
        newaxis="win",
        window_dur=window_dur,
        window_shift=window_shift,
        zero_pad_until="none",
    )

    # Warmup
    warmup = xformer(chunks[0])
    if backend == "mlx":
        import mlx.core as mx

        mx.eval(warmup.data)

    def process_all_chunks():
        outputs = [xformer(chunk) for chunk in chunks[1:]]
        if backend == "mlx":
            mx.eval(*[o.data for o in outputs])
        return outputs

    benchmark(process_all_chunks)


# --- Unit output shapes ---


def _batch_msgs(chunk_lens, fs=1000.0, n_ch=4, seed=0):
    """Input messages of the given sample counts, carrying contiguous data."""
    rng = np.random.default_rng(seed)
    total = sum(chunk_lens)
    data = rng.standard_normal((total, n_ch))
    msgs, start = [], 0
    for n in chunk_lens:
        msgs.append(
            AxisArray(
                data=data[start : start + n],
                dims=["time", "ch"],
                key="batch",
                axes={"time": AxisArray.TimeAxis(fs=fs, offset=start / fs)},
            )
        )
        start += n
    return msgs, data


def _drive_unit(settings, msgs):
    import asyncio

    from ezmsg.sigproc.window import Window

    async def _run():
        unit = Window(settings)
        unit.create_processor()
        out = []
        for msg in msgs:
            async for _, ret in unit.on_signal(msg):
                out.append(ret)
        return out

    return asyncio.run(_run())


def test_unit_split_defaults_axis_to_first_dim():
    """Regression: the unit resolved the target axis from SETTINGS.axis, so the
    documented `axis=None` default raised KeyError(None) inside a bare `except`
    -- every message swallowed, the stream silently dead for the whole run."""
    fs, win = 1000.0, 20
    msgs, data = _batch_msgs([30] * 8, fs=fs)
    settings = WindowSettings(
        axis=None, newaxis=None, window_dur=win / fs, window_shift=win / fs, zero_pad_until="none"
    )
    outs = _drive_unit(settings, msgs)

    assert len(outs) == 240 // win
    assert all(o.data.shape[0] == win for o in outs)
    assert np.array_equal(np.concatenate([o.data for o in outs], axis=0), data)
    assert np.allclose([o.axes["time"].offset for o in outs], np.arange(240 // win) * win / fs)


def test_unit_yields_exact_duration_messages_by_default():
    """The default for `newaxis=None`: one message per window, each exactly
    `window_dur` long -- what asking for a window_dur most obviously means."""
    fs, win = 1000.0, 20
    msgs, data = _batch_msgs([30] * 8, fs=fs)
    settings = WindowSettings(
        axis="time", newaxis=None, window_dur=win / fs, window_shift=win / fs, zero_pad_until="none"
    )
    outs = _drive_unit(settings, msgs)

    assert len(outs) == 240 // win
    assert all(o.dims == ["time", "ch"] for o in outs)
    assert all("win" not in o.axes for o in outs)
    assert all(o.data.shape[0] == win for o in outs)  # exactly window_dur, never a multiple
    assert np.array_equal(np.concatenate([o.data for o in outs], axis=0), data)
    assert np.allclose([o.axes["time"].offset for o in outs], np.arange(240 // win) * win / fs)


def test_unit_exact_duration_survives_an_oversized_input():
    """Even when one input completes several windows, the unit still publishes
    them one at a time at exactly window_dur."""
    fs, win = 1000.0, 20
    msgs, data = _batch_msgs([19, 81], fs=fs)  # second chunk completes 5 windows at once
    settings = WindowSettings(
        axis="time", newaxis=None, window_dur=win / fs, window_shift=win / fs, zero_pad_until="none"
    )
    outs = _drive_unit(settings, msgs)
    assert [o.data.shape[0] for o in outs] == [win] * 5
    assert np.array_equal(np.concatenate([o.data for o in outs], axis=0), data[: 5 * win])
    assert np.allclose([o.axes["time"].offset for o in outs], np.arange(5) * win / fs)
