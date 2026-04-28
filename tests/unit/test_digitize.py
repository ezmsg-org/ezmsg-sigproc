import copy

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.digitize import DigitizeDType, DigitizeTransformer, digitize
from tests.helpers.empty_time import check_empty_result, make_empty_msg
from tests.helpers.util import assert_messages_equal


def _make_msg(data: np.ndarray) -> AxisArray:
    return AxisArray(
        data=data,
        dims=["time", "channel"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
                "channel": AxisArray.CoordinateAxis(
                    data=np.array([f"Ch{i}" for i in range(data.shape[1])]),
                    dims=["channel"],
                ),
            }
        ),
        key="test_digitize",
    )


def test_digitize_int16_maps_input_range_to_signed_dtype_range():
    input_msg = _make_msg(np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]))
    backup = copy.deepcopy(input_msg)

    output_msg = DigitizeTransformer(min_val=-1.0, max_val=1.0, dtype="int16")(input_msg)

    assert_messages_equal([input_msg], [backup])
    assert output_msg.data.dtype == np.int16
    np.testing.assert_array_equal(
        output_msg.data,
        np.array(
            [
                [
                    np.iinfo(np.int16).min,
                    np.iinfo(np.int16).min,
                    0,
                    np.iinfo(np.int16).max,
                    np.iinfo(np.int16).max,
                ]
            ],
            dtype=np.int16,
        ),
    )
    assert output_msg.dims == input_msg.dims
    assert output_msg.axes == input_msg.axes
    assert output_msg.key == input_msg.key


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("int16", np.int16),
        (DigitizeDType.INT32, np.int32),
        ("int64", np.int64),
    ],
)
def test_digitize_supported_dtypes(dtype: str | DigitizeDType, expected_dtype: type[np.signedinteger]):
    input_msg = _make_msg(np.array([[-1.0, 0.0, 1.0]]))
    output_msg = DigitizeTransformer(min_val=-1.0, max_val=1.0, dtype=dtype)(input_msg)

    dtype_info = np.iinfo(expected_dtype)
    assert output_msg.data.dtype == expected_dtype
    np.testing.assert_array_equal(
        output_msg.data,
        np.array([[dtype_info.min, 0, dtype_info.max]], dtype=expected_dtype),
    )


def test_digitize_helper():
    proc = digitize(min_val=0.0, max_val=10.0, dtype="int32")
    input_msg = _make_msg(np.array([[0.0, 5.0, 10.0]]))

    output_msg = proc(input_msg)

    dtype_info = np.iinfo(np.int32)
    np.testing.assert_array_equal(
        output_msg.data,
        np.array([[dtype_info.min, 0, dtype_info.max]], dtype=np.int32),
    )


def test_digitize_invalid_dtype_raises():
    proc = DigitizeTransformer(min_val=-1.0, max_val=1.0, dtype="int8")

    with pytest.raises(ValueError, match="Unrecognized digitize dtype"):
        proc(_make_msg(np.array([[0.0]])))


def test_digitize_invalid_range_raises():
    proc = DigitizeTransformer(min_val=1.0, max_val=1.0, dtype="int16")

    with pytest.raises(ValueError, match="max_val must be greater than min_val"):
        proc(_make_msg(np.array([[0.0]])))


def test_digitize_empty_time():
    proc = DigitizeTransformer(min_val=-1.0, max_val=1.0, dtype="int16")
    result = proc(make_empty_msg())

    check_empty_result(result)
    assert result.data.dtype == np.int16
