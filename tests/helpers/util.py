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
        {"f": 10.0, "dur": 5.0, "offset": 0.0},
        {"f": 20.0, "dur": 5.0, "offset": 0.0},
        {"f": 70.0, "dur": 5.0, "offset": 0.0},
        {"f": 14.0, "dur": 5.0, "offset": 5.0},
        {"f": 35.0, "dur": 5.0, "offset": 5.0},
        {"f": 300.0, "dur": 5.0, "offset": 5.0},
    ],
    fs: float = 1000.0,
    msg_dur: float = 1.0,
    win_step_dur: float | None = None,
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
        data=np.array(["Ch1"]), unit="label", dims=["ch"]
    )
    for split_dat in data_splits:
        _time_axis = AxisArray.TimeAxis(fs=fs, offset=offset)
        messages.append(
            AxisArray(
                split_dat[..., None],
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
