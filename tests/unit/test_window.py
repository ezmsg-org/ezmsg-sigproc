import copy
from dataclasses import replace

import pytest
import numpy as np
from frozendict import frozendict
import sparse
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.window import windowing

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
                "ch": AxisArray.CoordinateAxis(
                    data=np.arange(nchans).astype(str), unit="label", dims=["ch"]
                ),
            }
        ),
        key="test_window_gen_nodur",
    )
    backup = [copy.deepcopy(test_msg)]
    proc = windowing(window_dur=None)
    result = proc(test_msg)
    assert_messages_equal([test_msg], backup)
    assert result is test_msg
    assert np.shares_memory(result.data, test_msg.data)


@pytest.mark.parametrize("msg_block_size", [1, 5, 10, 20, 60])
@pytest.mark.parametrize("newaxis", [None, "win"])
@pytest.mark.parametrize("win_dur", [0.3, 1.0])
@pytest.mark.parametrize("win_shift", [None, 0.2, 1.0])
@pytest.mark.parametrize("zero_pad", ["input", "shift", "none"])
@pytest.mark.parametrize("fs", [10.0, 500.0])
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
    nchans = 3

    shift_len = int(win_shift * fs) if win_shift is not None else None
    win_len = int(win_dur * fs)
    data_len = 2 * win_len
    if win_shift is not None:
        data_len += shift_len - 1
    data = np.arange(nchans * data_len, dtype=float).reshape((nchans, data_len))
    # Below, we transpose the individual messages if time_ax == 0.
    tvec = np.arange(data_len) / fs

    n_msgs = int(np.ceil(data_len / msg_block_size))

    # Instantiate the processor
    proc = windowing(
        axis="time",
        newaxis=newaxis,
        window_dur=win_dur,
        window_shift=win_shift,
        zero_pad_until=zero_pad,
        anchor=anchor,
    )

    # Create inputs and send them to the process, collecting the results along the way.
    test_msg = AxisArray(
        data[..., ()],
        dims=["ch", "time"] if time_ax == 1 else ["time", "ch"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=fs, offset=0.0),
                "ch": AxisArray.CoordinateAxis(
                    data=np.arange(nchans).astype(str), unit="label", dims=["ch"]
                ),
            }
        ),
        key="test_window_generator",
    )
    messages = []
    backup = []
    results = []
    for msg_ix in range(n_msgs):
        msg_data = data[..., msg_ix * msg_block_size : (msg_ix + 1) * msg_block_size]
        if time_ax == 0:
            msg_data = np.ascontiguousarray(msg_data.T)
        test_msg = replace(
            test_msg,
            data=msg_data,
            axes={
                **test_msg.axes,
                "time": replace(
                    test_msg.axes["time"], offset=tvec[msg_ix * msg_block_size]
                ),
            },
            key=test_msg.key,
        )
        messages.append(test_msg)
        backup.append(copy.deepcopy(test_msg))
        win_msg = proc(test_msg)
        results.append(win_msg)

    assert_messages_equal(messages, backup)

    # Check each return value's metadata (offsets checked at end)
    expected_dims = (
        test_msg.dims[:time_ax] + [newaxis or "win"] + test_msg.dims[time_ax:]
    )
    for msg in results:
        assert msg.axes["time"].gain == 1 / fs
        assert msg.dims == expected_dims
        assert (newaxis or "win") in msg.axes
        assert msg.axes[(newaxis or "win")].gain == (
            0.0 if win_shift is None else shift_len / fs
        )

    # Post-process the results to yield a single data array and a single vector of offsets.
    win_ax = time_ax
    # time_ax = win_ax + 1
    result = np.concatenate([_.data for _ in results], win_ax)
    offsets = np.hstack(
        [
            _.axes[newaxis or "win"].value(np.arange(_.data.shape[win_ax]))
            for _ in results
        ]
    )

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
    assert np.array_equal(result, expected)
    assert np.allclose(offsets, tvec)


@pytest.mark.parametrize("win_dur", [0.3, 1.0])
@pytest.mark.parametrize("win_shift", [0.2, 1.0, None])
@pytest.mark.parametrize("zero_pad", ["input", "shift", "none"])
def test_sparse_window(
    win_dur: float,
    win_shift: float | None,
    zero_pad: str,
):
    fs = 100.0
    n_ch = 5
    n_samps = 1_000
    msg_len = 100
    win_len = int(win_dur * fs)
    rng = np.random.default_rng()
    s = sparse.random((n_samps, n_ch), density=0.1, random_state=rng) > 0
    in_msgs = [
        AxisArray(
            data=s[msg_ix * msg_len : (msg_ix + 1) * msg_len],
            dims=["time", "ch"],
            axes={
                "time": AxisArray.Axis.TimeAxis(fs=fs, offset=msg_ix / fs),
            },
            key="test_sparse_window",
        )
        for msg_ix in range(10)
    ]

    proc = windowing(
        axis="time",
        newaxis="win",
        window_dur=win_dur,
        window_shift=win_shift,
        zero_pad_until=zero_pad,
    )
    out_msgs = [proc.send(_) for _ in in_msgs]
    nwins = 0
    for om in out_msgs:
        assert om.dims == ["win", "time", "ch"]
        assert om.data.shape[1] == win_len
        assert om.data.shape[2] == n_ch
        nwins += om.data.shape[0]
    if win_shift is None:
        # 1:1 mode
        assert nwins == len(out_msgs)
    else:
        shift_len = int(win_shift * fs)
        prepended = 0
        if zero_pad == "input":
            prepended = max(0, win_len - msg_len)
        elif zero_pad == "shift":
            prepended = max(0, win_len - shift_len)
        win_offsets = np.arange(n_samps + prepended)[::shift_len]
        expected_nwins = np.sum(win_offsets <= (n_samps + prepended - win_len))
        assert nwins == expected_nwins
