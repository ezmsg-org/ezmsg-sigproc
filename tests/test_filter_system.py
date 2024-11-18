import typing

import pytest
import numpy as np
import scipy.signal
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.terminate import TerminateOnTotal

from ezmsg.sigproc.butterworthfilter import ButterworthFilter
from ezmsg.sigproc.cheby import ChebyshevFilter
from ezmsg.sigproc.synth import EEGSynth

from util import get_test_fn


@pytest.mark.parametrize("filter_type", ["butter", "cheby1", "cheby2"])
@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_filter_system(filter_type: str, coef_type: str):
    test_filename = get_test_fn()
    test_filename_raw = test_filename.parent / (
        test_filename.stem + "raw" + test_filename.suffix
    )

    order = 4
    cuton = 5.0
    cutoff = 20.0
    Wn = (cuton, cutoff)
    btype = "bandpass"
    fs = 500.0
    n_ch = 8

    if filter_type == "butter":
        filter_comp = ButterworthFilter(
            order=order, cuton=cuton, cutoff=cutoff, axis="time", coef_type=coef_type
        )
    else:
        filter_comp = ChebyshevFilter(
            order=order,
            ripple_tol=0.1,
            Wn=Wn,
            btype=btype,
            cheby_type=filter_type,
            axis="time",
            coef_type=coef_type,
        )

    comps = {
        "SRC": EEGSynth(n_time=100, fs=fs, n_ch=n_ch, alpha_freq=10.5),  # 2 seconds
        "FILTER": filter_comp,
        "LOGRAW": MessageLogger(output=test_filename_raw),
        "LOGFILT": MessageLogger(output=test_filename),
        "TERM": TerminateOnTotal(10),
    }
    conns = (
        (comps["SRC"].OUTPUT_SIGNAL, comps["FILTER"].INPUT_SIGNAL),
        (comps["FILTER"].OUTPUT_SIGNAL, comps["LOGFILT"].INPUT_MESSAGE),
        (comps["SRC"].OUTPUT_SIGNAL, comps["LOGRAW"].INPUT_MESSAGE),
        (comps["LOGFILT"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    messages: typing.List[AxisArray] = [_ for _ in message_log(test_filename)]
    assert len(messages) >= 10
    inputs = AxisArray.concatenate(
        *[_ for _ in message_log(test_filename_raw)], dim="time"
    )
    outputs = AxisArray.concatenate(*messages, dim="time")

    # Calculate expected
    if filter_type == "butter":
        coefs = scipy.signal.butter(
            order,
            Wn=Wn,
            btype=btype,
            fs=fs,
            output=coef_type,
        )
    elif filter_type == "cheby1":
        coefs = scipy.signal.cheby1(
            order,
            rp=0.1,
            Wn=Wn,
            btype=btype,
            fs=fs,
            output=coef_type,
        )
    else:  # filter_type == "cheby2":
        coefs = scipy.signal.cheby2(
            order,
            rs=0.1,
            Wn=(5.0, 20.0),
            btype=btype,
            fs=fs,
            output=coef_type,
        )
    if coef_type == "ba":
        zi = scipy.signal.lfilter_zi(*coefs)[:, None]
        expected, _ = scipy.signal.lfilter(
            coefs[0], coefs[1], inputs.data, axis=inputs.get_axis_idx("time"), zi=zi
        )
    else:  # coef_type == "sos":
        zi = scipy.signal.sosfilt_zi(coefs)[:, :, None] + np.zeros((1, 1, n_ch))
        expected, _ = scipy.signal.sosfilt(
            coefs, inputs.data, axis=inputs.get_axis_idx("time"), zi=zi
        )
    assert np.allclose(outputs.data, expected)
