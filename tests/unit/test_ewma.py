import numpy as np

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.ewma import (
    EWMATransformer,
    ewma_step,
    _alpha_from_tau,
    _tau_from_alpha,
)


def test_tc_from_alpha():
    # np.log(1-0.6) = -dt / tau
    alpha = 0.6
    dt = 0.01
    tau = 0.010913566679372915
    assert np.isclose(_tau_from_alpha(alpha, dt), tau)
    assert np.isclose(_alpha_from_tau(tau, dt), alpha)


def test_ewma():
    time_constant = 0.010913566679372915
    fs = 100.0
    alpha = _alpha_from_tau(time_constant, 1 / fs)
    n_times = 100
    n_ch = 32
    n_feat = 4
    data = np.arange(1, n_times * n_ch * n_feat + 1, dtype=float).reshape(
        n_times, n_ch, n_feat
    )
    msg = AxisArray(
        data=data,
        dims=["time", "ch", "feat"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": AxisArray.CoordinateAxis(
                data=np.arange(n_ch).astype(str), dims=["ch"]
            ),
            "feat": AxisArray.CoordinateAxis(data=np.arange(n_feat), dims=["feat"]),
        },
    )

    # Expected
    expected = [data[0]]
    for ix, dat in enumerate(data):
        expected.append(ewma_step(dat, expected[-1], alpha))
    expected = np.stack(expected)[1:]

    ewma = EWMATransformer(time_constant=time_constant, axis="time")
    res = ewma(msg)
    assert np.allclose(res.data, expected)
