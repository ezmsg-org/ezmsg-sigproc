import copy
import importlib.util

import numpy as np
from ezmsg.util.messages.chunker import array_chunker
from frozendict import frozendict
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.scaler import scaler, scaler_np, EWMA, ewma_step

from tests.helpers.util import assert_messages_equal


def test_ewma():
    alpha = 0.6
    n_times = 100
    n_ch = 32
    n_feat = 4
    data = np.arange(1, n_times * n_ch * n_feat + 1, dtype=float).reshape(
        n_times, n_ch, n_feat
    )

    # Expected
    expected = [data[0]]
    for ix, dat in enumerate(data):
        expected.append(ewma_step(dat, expected[-1], alpha))
    expected = np.stack(expected)[1:]

    ewma = EWMA(alpha=alpha)
    res = ewma.compute(data)
    assert np.allclose(res, expected)


@pytest.fixture
def fixture_arrays():
    # Test data values taken from river:
    # https://github.com/online-ml/river/blob/main/river/preprocessing/scale.py#L511-L536C17
    data = np.array([5.278, 5.050, 6.550, 7.446, 9.472, 10.353, 11.784, 11.173])
    expected_result = np.array([0.0, -0.816, 0.812, 0.695, 0.754, 0.598, 0.651, 0.124])
    return data, expected_result


@pytest.mark.skipif(
    importlib.util.find_spec("river") is None, reason="requires `river` package"
)
def test_adaptive_standard_scaler_river(fixture_arrays):
    data, expected_result = fixture_arrays

    test_input = AxisArray(
        np.tile(data, (2, 1)),
        dims=["ch", "time"],
        axes=frozendict({"time": AxisArray.TimeAxis(fs=100.0)}),
    )

    backup = [copy.deepcopy(test_input)]

    # The River example used alpha = 0.6
    # tau = -gain / np.log(1 - alpha) and here we're using gain = 0.01
    tau = 0.010913566679372915
    _scaler = scaler(time_constant=tau, axis="time")
    output = _scaler.send(test_input)
    assert np.allclose(output.data[0], expected_result, atol=1e-3)
    assert_messages_equal([test_input], backup)


def test_scaler_np(fixture_arrays):
    data, expected_result = fixture_arrays
    chunker = array_chunker(data, 4, fs=100.0)
    test_input = list(chunker)
    backup = copy.deepcopy(test_input)

    tau = 0.010913566679372915
    gen = scaler_np(time_constant=tau, axis="time")
    outputs = []
    for chunk in test_input:
        outputs.append(gen.send(chunk))
    output = AxisArray.concatenate(*outputs, dim="time")
    assert np.allclose(output.data, expected_result, atol=1e-3)
    assert_messages_equal(test_input, backup)
