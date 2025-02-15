import copy

import pytest
import numpy as np
from ezmsg.util.messages.chunker import array_chunker

from ezmsg.sigproc.transpose import TransposeTransformer

from tests.helpers.util import assert_messages_equal


@pytest.mark.parametrize(
    "axes",
    [
        ["time", "d1", "d2"],  # No-op
        None,  # No-op
        ["time", ...],  # No-op
        [..., "time"],  # Move time to the end
        ["time", "d2", "d1"],  # Keep time but Swap d1 and d2
    ],
)
@pytest.mark.parametrize("order", ["C", "F", None])
def test_transpose(axes: tuple[int, ...] | None, order: str | None):
    fs = 100.0
    dur = 30.0
    n_ch = 3
    n_feat = 2
    n_time = int(fs * dur)
    data = np.arange(n_time * n_ch * n_feat, dtype=float).reshape(n_time, n_ch, n_feat)
    chunker = array_chunker(data, 4, axis=0, fs=fs)
    test_input = list(chunker)
    backup = copy.deepcopy(test_input)

    proc = TransposeTransformer(axes=axes, order=order)
    results = [proc(_) for _ in test_input]

    # Assert input is unchanged
    assert_messages_equal(test_input, backup)

    # Assert output is transposed
    if (axes is None or axes in [["time", "d1", "d2"], ["time", ...]]) and (
        order is None or order == "C"
    ):
        # No-op path.
        assert_messages_equal(results, test_input)
        if order is None:
            # Further assert that it was a no-op passthrough!
            for msg_ix, msg_in in enumerate(test_input):
                msg_out = results[msg_ix]
                assert msg_out is msg_in
    else:
        ax_ints = [0, 1, 2]
        if axes == [..., "time"]:
            ax_ints = [1, 2, 0]
        elif axes == ["time", "d2", "d1"]:
            ax_ints = [0, 2, 1]
        for msg_ix, msg_in in enumerate(test_input):
            msg_out = results[msg_ix]
            assert msg_out.dims == [msg_in.dims[ix] for ix in ax_ints]
            assert np.allclose(msg_out.data, np.transpose(msg_in.data, ax_ints))
            if order == "C":
                assert msg_out.data.flags.c_contiguous
            elif order == "F":
                assert msg_out.data.flags.f_contiguous
