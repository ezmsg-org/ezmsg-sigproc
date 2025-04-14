import copy

import pytest
import numpy as np
from frozendict import frozendict

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.quantize import QuantizeTransformer

from tests.helpers.util import assert_messages_equal


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16, 32, 64])
def test_quantize(bits: int):
    data_range = (-1e9, 1e9)
    # Create sample data with values ranging from -2000 to +2000
    data = np.linspace(data_range[0], data_range[1], 100).reshape(20, 5)

    # Create an AxisArray message
    input_msg = AxisArray(
        data=data,
        dims=["time", "channel"],
        axes=frozendict({
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
            "channel": AxisArray.CoordinateAxis(
                data=np.array([f"Ch{i}" for i in range(5)]),
                dims=["channel"]
            )
        }),
        key="test_quantize"
    )

    # Create a backup for comparison
    backup = copy.deepcopy(input_msg)

    # Create and apply the quantizer
    quantizer = QuantizeTransformer(min_val=data_range[0], max_val=data_range[1], bits=bits)
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
    # The first element should be close to 0 (minimum)
    assert output_msg.data[0, 0] == 0
    if bits <= 1:
        assert np.min(output_msg.data) == False
        assert np.max(output_msg.data) == True
    else:
        # Verify output range is [0, 255]
        assert np.min(output_msg.data) >= 0
        assert np.max(output_msg.data) == 2**bits - 1
        assert output_msg.data[-1, -1] == 2**bits - 1

    # # Middle element should be close to 127/128
    # mid_index = len(data) // 2
    # mid_value = output_msg.data.flatten()[mid_index]
    # assert 120 <= mid_value <= 135
