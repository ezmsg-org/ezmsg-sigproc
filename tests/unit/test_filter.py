import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.filter import FilterCoefficients, FilterSettings, FilterTransformer
from tests.helpers.empty_time import check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg


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


def test_filter_empty_after_init():
    coefs = FilterCoefficients(b=np.array([1.0]), a=np.array([1.0, 0.0]))
    proc = FilterTransformer(settings=FilterSettings(axis="time", coefs=coefs))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_filter_empty_none_passthrough():
    proc = FilterTransformer(settings=FilterSettings(axis="time", coefs=None))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_filter_empty_first():
    coefs = FilterCoefficients(b=np.array([1.0]), a=np.array([1.0, 0.0]))
    proc = FilterTransformer(settings=FilterSettings(axis="time", coefs=coefs))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_chebyshev_empty_after_init():
    from ezmsg.sigproc.cheby import ChebyshevFilterSettings, ChebyshevFilterTransformer

    proc = ChebyshevFilterTransformer(
        ChebyshevFilterSettings(order=4, ripple_tol=0.5, Wn=10.0, btype="lowpass", wn_hz=True, axis="time")
    )
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_chebyshev_empty_first():
    from ezmsg.sigproc.cheby import ChebyshevFilterSettings, ChebyshevFilterTransformer

    proc = ChebyshevFilterTransformer(
        ChebyshevFilterSettings(order=4, ripple_tol=0.5, Wn=10.0, btype="lowpass", wn_hz=True, axis="time")
    )
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)
