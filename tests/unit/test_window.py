import copy
from dataclasses import replace

import numpy as np
import pytest
import sparse
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.window import WindowTransformer
from tests.helpers.util import assert_messages_equal, calculate_expected_windows


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

    # Check each return value's metadata (offsets checked at end)
    expected_dims = template_msg.dims[:time_ax] + [newaxis or "win"] + template_msg.dims[time_ax:]
    for msg in out_msgs:
        assert msg.axes["time"].gain == 1 / fs
        assert msg.dims == expected_dims
        assert (newaxis or "win") in msg.axes
        assert msg.axes[(newaxis or "win")].gain == (0.0 if win_shift is None else shift_len / fs)

    # Post-process the results to yield a single data array and a single vector of offsets.
    win_ax = time_ax
    # time_ax = win_ax + 1
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
