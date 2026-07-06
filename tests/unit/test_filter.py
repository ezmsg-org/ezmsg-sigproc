import numpy as np
import pytest
import scipy.signal
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


@pytest.mark.parametrize("coef_type", ["ba", "sos"])
def test_filter_edge_scale_zi_suppresses_dc_startup(coef_type):
    """The plain causal FilterTransformer honors edge_scale_zi: with it on, a
    constant-DC input passes a unity-DC-gain low-pass without a start-up ramp;
    with it off (the base-settings default) the output ramps up from ~0.

    This exercises the knob directly on FilterBaseSettings, independent of the
    zero-phase composite that owns the True default.
    """
    fs = 1000.0
    dc = 5000.0
    data = np.full((500, 2), dc)

    if coef_type == "ba":
        b, a = scipy.signal.butter(4, 50.0, btype="low", fs=fs)
        coefs = FilterCoefficients(b=b, a=a)
    else:
        coefs = scipy.signal.butter(4, 50.0, btype="low", fs=fs, output="sos")

    def run(edge_scale_zi: bool) -> np.ndarray:
        proc = FilterTransformer(
            settings=FilterSettings(
                axis="time", coef_type=coef_type, coefs=coefs, edge_scale_zi=edge_scale_zi
            )
        )
        msg = AxisArray(
            data=data.copy(),
            dims=["time", "ch"],
            axes={"time": AxisArray.TimeAxis(fs=fs)},
            key="dc",
        )
        return proc(msg).data

    y_on = run(True)
    y_off = run(False)

    # DC gain of a low-pass is 1, so the settled output is `dc` in both cases.
    assert np.allclose(y_on[-1], dc, rtol=1e-3, atol=1.0)
    assert np.allclose(y_off[-1], dc, rtol=1e-3, atol=1.0)

    # With edge-scaling the output starts already at steady state (flat)...
    assert np.allclose(y_on, dc, rtol=0, atol=1e-6)
    assert np.ptp(y_on) < 1e-6
    # ...without it, the first sample is far below dc and there is a big ramp.
    assert abs(y_off[0, 0]) < 0.5 * dc
    assert np.ptp(y_off[:, 0]) > 0.5 * dc


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
