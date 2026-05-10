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


def test_digitize_emits_inverse_mapping_attrs():
    """Output ``attrs`` carry the inverse-mapping coefficients needed to
    recover (an approximation of) the original float values via
    ``data * conversion + offset``.

    For a symmetric int16 range the offset is essentially zero (within
    ½ LSB), and the conversion is ``(max-min)/(2**16-1)``.

    ``min_val`` / ``max_val`` are deliberately *not* emitted — a consumer
    with the data dtype can recover them from ``conversion`` + ``offset``
    alone, so carrying them on the wire would be redundant.
    """
    proc = DigitizeTransformer(min_val=-8.0, max_val=8.0, dtype="int16")
    floats = np.linspace(-8.0, 8.0, 9, dtype=np.float64)[None, :]
    out = proc(_make_msg(floats))

    expected_conv = 16.0 / 65535.0
    assert set(out.attrs) >= {"conversion", "offset"}
    assert "min_val" not in out.attrs
    assert "max_val" not in out.attrs
    assert out.attrs["conversion"] == pytest.approx(expected_conv, rel=1e-9)
    assert abs(out.attrs["offset"]) < 1e-3, f"symmetric int16 range should give offset ~ 0; got {out.attrs['offset']}"

    # Round-trip: data * conversion + offset recovers the original
    # values within ½ LSB (≈ 16 / 65535).
    recovered = out.data.astype(np.float64) * out.attrs["conversion"] + out.attrs["offset"]
    np.testing.assert_allclose(recovered, floats, atol=expected_conv)


def test_digitize_preserves_existing_attrs():
    """Pre-existing ``attrs`` flow through alongside the new ones."""
    proc = DigitizeTransformer(min_val=-1.0, max_val=1.0, dtype="int16")
    msg = _make_msg(np.array([[0.0]]))
    msg.attrs["upstream_key"] = "preserve_me"

    out = proc(msg)
    assert out.attrs["upstream_key"] == "preserve_me"
    assert "conversion" in out.attrs
