import numpy as np
import pytest
import scipy.special
from frozendict import frozendict

from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.activation import activation, ActivationFunction, ACTIVATIONS


@pytest.mark.parametrize(
    "function", [_ for _ in ActivationFunction] + ActivationFunction.options()
)
def test_activation(function: str):
    in_fs = 19.0
    sig = np.arange(24, dtype=float).reshape(4, 3, 2)
    if function in [ActivationFunction.LOGIT, "logit"]:
        sig += 1e-9
        sig /= np.max(sig) + 1e-3

    def msg_generator():
        for msg_ix in range(sig.shape[0]):
            msg_sig = sig[msg_ix : msg_ix + 1]
            msg = AxisArray(
                data=msg_sig,
                dims=["time", "ch", "feat"],
                axes=frozendict(
                    {"time": AxisArray.TimeAxis(fs=in_fs, offset=msg_ix / in_fs)}
                ),
            )
            yield msg

    proc = activation(function=function)
    out_msgs = [proc.send(_) for _ in msg_generator()]
    out_dat = AxisArray.concatenate(*out_msgs, dim="time").data

    if function in ACTIVATIONS:
        expected_func = ACTIVATIONS[function]
    else:
        expected_func = {
            "sigmoid": scipy.special.expit,
            "expit": scipy.special.expit,
            "logit": scipy.special.logit,
            "log_expit": scipy.special.log_expit,
        }.get(function.lower(), lambda x: x)
    expected_dat = expected_func(sig)
    assert np.allclose(out_dat, expected_dat)
