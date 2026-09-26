"""Factor-only Decimate configuration must filter before subsampling."""

import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.decimate import ChebyForDecimateTransformer, Decimate, DecimateSettings
from ezmsg.sigproc.downsample import DownsampleTransformer


@pytest.mark.parametrize("factor", [1, 2, 5])
@pytest.mark.parametrize("fs", [500.0, 1000.0])
def test_factor_only(factor, fs):
    settings = DecimateSettings(target_rate=None, factor=factor, axis=None)
    component = Decimate(settings=settings)
    component.configure()
    filt = ChebyForDecimateTransformer(settings=component.FILTER.SETTINGS)
    down = DownsampleTransformer(settings=component.DOWNSAMPLE.SETTINGS)
    data = np.random.default_rng(0).normal(size=1000)
    message = AxisArray(data, dims=["time"], axes={"time": AxisArray.TimeAxis(fs=fs)})
    output = down(filt(message))
    if factor == 1:
        expected = data
    else:
        b, a = scipy.signal.cheby1(8, 0.05, 0.8 / factor)
        expected, _ = scipy.signal.lfilter(b, a, data, zi=scipy.signal.lfilter_zi(b, a) * data[0])
    np.testing.assert_allclose(output.data, expected[::factor])
    assert output.axes["time"].gain == pytest.approx(factor / fs)


@pytest.mark.parametrize(
    "settings",
    [{}, {"factor": 0}, {"factor": 1.5}, {"target_rate": -1}, {"target_rate": float("nan")}],
)
def test_invalid_configuration(settings):
    with pytest.raises(ValueError, match="Decimate"):
        Decimate(**settings).configure()


def test_factor_overrides_target_rate():
    component = Decimate(factor=2, target_rate=123)
    component.configure()
    assert component.FILTER.SETTINGS.Wn == 0.4
    assert not component.FILTER.SETTINGS.wn_hz
