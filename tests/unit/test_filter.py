import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.filter import FilterCoefficients, FilterSettings, FilterTransformer


def test_filter_transformer_accepts_dataclass_coefficients():
    data = np.arange(10.0)
    msg = AxisArray(
        data=data,
        dims=["time"],
        axes={"time": AxisArray.TimeAxis(fs=1.0, offset=0.0)},
        key="test",
    )
    coefs = FilterCoefficients(b=np.array([1.0]), a=np.array([1.0, 0.0]))
    transformer = FilterTransformer(settings=FilterSettings(axis="time", coefs=coefs))
    out = transformer(msg)
    assert np.allclose(out.data, data)
