import pytest
import numpy as np

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.extract_axis import ExtractAxisData


@pytest.mark.parametrize(
    "message",
    [
        AxisArray(
            data=np.arange(500 * 32 * 3).reshape(500, 32, 3),
            dims=["time", "ch", "bank"],
            axes={
                "time": AxisArray.TimeAxis(offset=0.0, fs=100.0),
                "ch": AxisArray.CoordinateAxis(
                    data=np.arange(32).astype(str), dims=["ch"]
                ),
                "bank": AxisArray.LinearAxis(gain=1.0, offset=0.0),
                "freq": AxisArray.CoordinateAxis(
                    data=60 + 0.1 * np.random.randn(500, 32),
                    dims=["time", "ch"],
                ),
            },
            key="test_extract_axis",
        )
    ],
)
class TestExtractAxisData:
    def test_extract_linear_axis(self, message):
        processor = ExtractAxisData(axis="bank", reference="time")
        result = processor(message)

        # Verify the output
        assert result.dims == ["time"]
        assert "time" in result.axes
        assert "ch" not in result.axes
        assert "bank" not in result.axes
        assert "freq" not in result.axes
        assert result.data.shape == (500,)
        assert np.allclose(result.data, np.arange(500))

    def test_extract_reference_axis(self, message):
        processor = ExtractAxisData(axis="time", reference="time")
        result = processor(message)

        # Verify the output
        assert result.dims == ["time"]
        assert "time" in result.axes
        assert "ch" not in result.axes
        assert "bank" not in result.axes
        assert "freq" not in result.axes
        assert np.allclose(result.data, np.arange(500.0) / 100.0)

    def test_extract_coordinate_axis(self, message):
        processor = ExtractAxisData(axis="freq", reference="time")
        result = processor(message)

        # Verify the output
        assert result.dims == ["time", "ch"]
        assert "time" in result.axes
        assert "ch" in result.axes
        assert result.data.shape == (500, 32)
        assert np.array_equal(result.data, message.axes["freq"].data)
        assert np.shares_memory(result.data, message.axes["freq"].data)
