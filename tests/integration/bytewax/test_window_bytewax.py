import pytest
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.chunker import array_chunker

from ezmsg.sigproc.window import WindowTransformer

try:
    import bytewax.operators as op
    from bytewax.dataflow import Dataflow
    from bytewax.testing import TestingSource, TestingSink, run_main

    b_bytewax = True
except ImportError:
    b_bytewax = False


@pytest.fixture
def axarr_list() -> list[AxisArray]:
    data = np.arange(3000 * 4 * 5).reshape(3000, 4, 5)
    gen = array_chunker(data, chunk_len=100, axis=0, fs=1000.0, tzero=0.0)
    return list(gen)


@pytest.mark.skipif(not b_bytewax, reason="Bytewax not installed")
def test_window_bytewax(axarr_list):
    result = []
    flow = Dataflow("test_window_bytewax")
    input_stream = op.input("input", flow, TestingSource(axarr_list))
    keyed_stream = op.key_on("key_stream", input_stream, lambda _: "ALL")
    windowed_stream = op.stateful_map(
        "windower",
        keyed_stream,
        WindowTransformer(
            axis="time",
            newaxis="win",
            window_dur=1.0,
            window_shift=0.1,
            zero_pad_until="none",
        ).stateful_op,
    )
    op.output("out", windowed_stream, TestingSink(result))
    run_main(flow)

    # Calculated expected result
    in_data = AxisArray.concatenate(*axarr_list, dim="time").data
    expected = sliding_window_view(in_data, window_shape=1000, axis=0)[::100]
    expected = expected.transpose([0, 3, 1, 2])

    # Collect the output
    out_data = AxisArray.concatenate(*[_[1] for _ in result], dim="win").data

    # Compare to collected
    assert np.array_equal(out_data, expected)


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
