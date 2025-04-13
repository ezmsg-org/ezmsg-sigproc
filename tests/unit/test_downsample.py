import copy

import pytest
import numpy as np
from frozendict import frozendict

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.downsample import DownsampleTransformer

from tests.helpers.util import assert_messages_equal


@pytest.mark.parametrize("block_size", [1, 5, 10, 20])
@pytest.mark.parametrize("target_rate", [19.0, 9.5, 6.3])
@pytest.mark.parametrize("factor", [None, 1, 2])
def test_downsample_core(block_size: int, target_rate: float, factor: int | None):
    in_fs = 19.0
    test_dur = 4.0
    n_channels = 2
    n_features = 3
    num_samps = int(np.ceil(test_dur * in_fs))
    num_msgs = int(np.ceil(num_samps / block_size))
    sig = np.arange(num_samps * n_channels * n_features).reshape(
        num_samps, n_channels, n_features
    )
    # tvec = np.arange(num_samps) / in_fs

    def msg_generator():
        for msg_ix in range(num_msgs):
            msg_sig = sig[msg_ix * block_size : (msg_ix + 1) * block_size]
            msg_idx: float = msg_sig[0, 0, 0] / (n_channels * n_features)
            msg_offs = msg_idx / in_fs
            msg = AxisArray(
                data=msg_sig,
                dims=["time", "ch", "feat"],
                axes=frozendict(
                    {
                        "time": AxisArray.TimeAxis(fs=in_fs, offset=msg_offs),
                        "ch": AxisArray.CoordinateAxis(
                            data=np.arange(n_channels).astype(str), dims=["ch"]
                        ),
                        "feat": AxisArray.CoordinateAxis(
                            data=np.array([f"Feat{_ + 1}" for _ in range(n_features)]),
                            dims=["feat"],
                        ),
                    }
                ),
                key="test_downsample_core",
            )
            yield msg

    in_msgs = list(msg_generator())
    backup = [copy.deepcopy(msg) for msg in in_msgs]

    proc = DownsampleTransformer(axis="time", target_rate=target_rate, factor=factor)
    out_msgs = []
    for msg in in_msgs:
        res = proc(msg)
        if res.data.size:
            out_msgs.append(res)

    assert_messages_equal(in_msgs, backup)

    # Assert correctness of gain
    expected_factor: int = int(in_fs // target_rate) if factor is None else factor
    assert all(msg.axes["time"].gain == expected_factor / in_fs for msg in out_msgs)

    # Assert messages have the correct timestamps
    expected_offsets = (
        np.cumsum([0] + [_.data.shape[0] for _ in out_msgs]) * expected_factor / in_fs
    )
    actual_offsets = np.array([_.axes["time"].offset for _ in out_msgs])
    assert np.allclose(actual_offsets, expected_offsets[:-1])

    # Compare returned values to expected values.
    allres_msg = AxisArray.concatenate(*out_msgs, dim="time")
    assert np.array_equal(allres_msg.data, sig[::expected_factor])
