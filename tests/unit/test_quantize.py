import copy

import pytest
import numpy as np
from frozendict import frozendict

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.quantize import QuantizeTransformer

from tests.helpers.util import assert_messages_equal


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16, 32, 64])
def test_quantize(bits: int):
    data_range = [-8_192.0, 8_192.0]
    min_step = (data_range[1] - data_range[0]) / 2**bits
    min_step = max(min_step, 2e-12)  # Practically, this is the minimum step size
    data = np.array(
        [
            [
                data_range[0],
                data_range[0] + min_step,
                -min_step,
                0,
                min_step,
                data_range[1] - min_step,
                data_range[1],
            ]
        ]
    )

    # Create an AxisArray message
    input_msg = AxisArray(
        data=data,
        dims=["time", "channel"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
                "channel": AxisArray.CoordinateAxis(
                    data=np.array([f"Ch{i}" for i in range(7)]), dims=["channel"]
                ),
            }
        ),
        key="test_quantize",
    )

    # Create a backup for comparison
    backup = copy.deepcopy(input_msg)

    # Create and apply the quantizer
    quantizer = QuantizeTransformer(
        min_val=data_range[0], max_val=data_range[1], bits=bits
    )
    output_msg = quantizer(input_msg)

    # Verify original message wasn't modified
    assert_messages_equal([input_msg], [backup])

    # Verify output data type is integer
    if bits <= 1:
        assert output_msg.data.dtype == bool
    else:
        assert np.issubdtype(output_msg.data.dtype, np.integer)
        if bits <= 8:
            assert output_msg.data.dtype == np.uint8
        elif bits <= 16:
            assert output_msg.data.dtype == np.uint16
        elif bits <= 32:
            assert output_msg.data.dtype == np.uint32
        else:
            assert output_msg.data.dtype == np.uint64

    # Verify the quantization mapping
    if bits <= 1:
        assert not np.min(output_msg.data)
        assert np.max(output_msg.data)
    else:
        assert output_msg.data[0, 0] == 0
        assert output_msg.data[0, 3] == 2 ** (bits - 1)
        if bits == 64:
            # 64-bit quantization behaves strangely because float64
            #  does not map to uint64 correctly. The QuantizeTransformer
            #  does some clipping to prevent this, so here we just verify
            #  that the number is very large.
            assert output_msg.data[0, 6] > 2**63
        else:
            assert output_msg.data[0, 1] == 1
            assert output_msg.data[0, 5] == 2**bits - 2
            assert output_msg.data[0, 6] == 2**bits - 1
