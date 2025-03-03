import os
import tempfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from frozendict import frozendict
from ezmsg.util.messages.axisarray import AxisArray


def get_test_fn(test_name: str | None = None, extension: str = "txt") -> Path:
    """PYTEST compatible temporary test file creator"""

    # Get current test name if we can..
    if test_name is None:
        test_name = os.environ.get("PYTEST_CURRENT_TEST")
        if test_name is not None:
            test_name = test_name.split(":")[-1].split(" ")[0]
        else:
            test_name = __name__

    file_path = Path(tempfile.gettempdir())
    file_path = file_path / Path(f"{test_name}.{extension}")

    # Create the file
    with open(file_path, "w"):
        pass

    return file_path


def create_messages_with_periodic_signal(
    sin_params: list[dict[str, float]] = [
        {"f": 10.0, "dur": 5.0, "offset": 0.0, "p": 0},
        {"f": 20.0, "dur": 5.0, "offset": 0.0, "p": 0},
        {"f": 70.0, "dur": 5.0, "offset": 0.0, "p": 0},
        {"f": 14.0, "dur": 5.0, "offset": 5.0, "p": 0},
        {"f": 35.0, "dur": 5.0, "offset": 5.0, "p": 0},
        {"f": 300.0, "dur": 5.0, "offset": 5.0, "p": 0},
    ],
    fs: float = 1000.0,
    msg_dur: float = 1.0,
    win_step_dur: float | None = None,
    n_ch: int = 1,
) -> list[AxisArray]:
    """
    Create a continuous signal with periodic components. The signal will be divided into n segments,
    where n is the number of lists in f_sets. Each segment will have sinusoids (of equal amplitude)
    at each of the frequencies in the f_set. Each segment will be seg_dur seconds long.
    """
    t_end = max([_.get("offset", 0.0) + _["dur"] for _ in sin_params])
    t_vec = np.arange(int(t_end * fs)) / fs
    data = np.zeros((len(t_vec),))
    # TODO: each freq should be evaluated independently and the dict should have a "dur" and "offset" value, both in sec
    # TODO: Get rid of `win_dur` and replace with `msg_dur`
    for s_p in sin_params:
        offs = s_p.get("offset", 0.0)
        b_t = np.logical_and(t_vec >= offs, t_vec <= offs + s_p["dur"])
        data[b_t] += s_p.get("a", 1.0) * np.sin(
            2 * np.pi * s_p["f"] * t_vec[b_t] + s_p.get("p", 0)
        )

    # How will we split the data into messages? With a rolling window or non-overlapping?
    if win_step_dur is not None:
        win_step = int(win_step_dur * fs)
        data_splits = sliding_window_view(data, (int(msg_dur * fs),), axis=0)[
            ::win_step
        ]
    else:
        n_msgs = int(t_end / msg_dur)
        data_splits = np.array_split(data, n_msgs, axis=0)

    # Create the output messages
    offset = 0.0
    messages = []
    _ch_axis = AxisArray.CoordinateAxis(
        data=np.array([f"Ch{_}" for _ in range(n_ch)]), unit="label", dims=["ch"]
    )
    for split_dat in data_splits:
        _time_axis = AxisArray.TimeAxis(fs=fs, offset=offset)
        messages.append(
            AxisArray(
                split_dat[..., None] + np.zeros((1, n_ch)),
                dims=["time", "ch"],
                axes=frozendict(
                    {
                        "time": _time_axis,
                        "ch": _ch_axis,
                    }
                ),
                key="ezmsg.sigproc generated periodic signal",
            )
        )
        offset += split_dat.shape[0] / fs
    return messages


def assert_messages_equal(messages1, messages2):
    # Verify the inputs have not changed as a result of processing.
    for msg_ix in range(len(messages1)):
        msg1 = messages1[msg_ix]
        msg2 = messages2[msg_ix]
        assert type(msg1) is type(msg2)
        if isinstance(msg1, AxisArray):
            assert np.array_equal(msg1.data, msg2.data)
            assert msg1.dims == msg2.dims
            assert list(msg1.axes.keys()) == list(msg2.axes.keys())
            for k, v in msg1.axes.items():
                assert k in msg2.axes
                assert v == msg2.axes[k]
        else:
            assert msg1.__dict__ == msg2.__dict__


def calculate_expected_windows(
    orig,
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
):
    """
    Used by unit/test_window and integration/ezmsg/test_window to calculate the expected output of a windowing operation.
    """
    # For the calculation, we assume time_ax is last then transpose if necessary at the end.
    expected = orig.copy()
    tvec = np.arange(orig.shape[1]) / fs
    # Prepend the data with zero-padding, if necessary.
    if win_shift is None or zero_pad == "input":
        n_cut = msg_block_size
    elif zero_pad == "shift":
        n_cut = shift_len
    else:  # "none" -- no buffer needed
        n_cut = win_len
    n_keep = win_len - n_cut
    if n_keep > 0:
        expected = np.concatenate(
            (np.zeros((nchans, win_len))[..., -n_keep:], expected), axis=-1
        )
        tvec = np.hstack(((np.arange(-win_len, 0) / fs)[-n_keep:], tvec))
    # Moving window -- assumes step size of 1
    expected = sliding_window_view(expected, win_len, axis=-1)
    tvec = sliding_window_view(tvec, win_len)
    # Mimic win_shift
    if win_shift is None:
        # 1:1 mode. Each input (block) yields a new output.
        # If the window length is smaller than the block size then we only the tail of each block.
        first = max(min(msg_block_size, data_len) - win_len, 0)
        if tvec[::msg_block_size].shape[0] < n_msgs:
            expected = np.concatenate(
                (expected[:, first::msg_block_size], expected[:, -1:]), axis=1
            )
            tvec = np.hstack((tvec[first::msg_block_size, 0], tvec[-1:, 0]))
        else:
            expected = expected[:, first::msg_block_size]
            tvec = tvec[first::msg_block_size, 0]
    else:
        expected = expected[:, ::shift_len]
        tvec = tvec[::shift_len, 0]

    if anchor == "middle":
        tvec = tvec + win_len / (2 * fs)
    elif anchor == "end":
        tvec = tvec + win_len / fs

    # Transpose to put time_ax and win_ax in the correct locations.
    if win_ax == 0:
        expected = np.moveaxis(expected, 0, -1)

    return expected, tvec


def gaussian(x, x0, sigma):
    return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)


def make_chirp(t, t0, a):
    frequency = (a * (t + t0)) ** 2
    chirp = np.sin(2 * np.pi * frequency * t)
    return chirp, frequency
