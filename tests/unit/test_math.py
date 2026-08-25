import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.math.abs import AbsTransformer
from ezmsg.sigproc.math.anscombe import (
    _D_MIN,
    AnscombeTransformer,
    InverseAnscombeSettings,
    InverseAnscombeTransformer,
    InverseMethod,
)
from ezmsg.sigproc.math.clip import ClipSettings, ClipTransformer
from ezmsg.sigproc.math.difference import ConstDifferenceSettings, ConstDifferenceTransformer
from ezmsg.sigproc.math.invert import InvertTransformer
from ezmsg.sigproc.math.log import LogSettings, LogTransformer
from ezmsg.sigproc.math.pow import PowSettings, PowTransformer
from ezmsg.sigproc.math.scale import ScaleSettings, ScaleTransformer
from tests.helpers.empty_time import check_empty_result, make_empty_msg
from tests.helpers.util import requires_mlx


def test_abs():
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    xformer = AbsTransformer()
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, np.abs(in_dat))


@pytest.mark.parametrize("dtype", [int, float])
def test_anscombe(dtype):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans).astype(dtype)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = AnscombeTransformer()
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == (n_times, n_chans)
    assert np.allclose(msg_out.data, 2.0 * np.sqrt(in_dat + 3 / 8))


def test_anscombe_stabilizes_poisson_variance():
    """The point of the transform: variance ~1 regardless of the Poisson rate."""
    rng = np.random.default_rng(42)
    rates = np.array([5.0, 50.0, 500.0])
    in_dat = rng.poisson(rates, size=(20000, rates.size)).astype(float)

    # Before: variance tracks the rate. After: variance is ~1 for every channel.
    assert not np.allclose(np.var(in_dat, axis=0), 1.0, atol=0.2)
    out = AnscombeTransformer()(AxisArray(in_dat, dims=["time", "ch"])).data
    assert np.allclose(np.var(out, axis=0), 1.0, atol=0.05)


def test_inverse_anscombe_algebraic_round_trips():
    in_dat = np.arange(130 * 255).reshape(130, 255).astype(float)
    fwd = AnscombeTransformer()(AxisArray(in_dat, dims=["time", "ch"]))
    inv = InverseAnscombeTransformer(InverseAnscombeSettings(method=InverseMethod.ALGEBRAIC))(fwd)
    assert np.allclose(inv.data, in_dat)


@pytest.mark.parametrize("method", ["exact", "asymptotic", "algebraic"])
def test_inverse_anscombe_accepts_str(method: str):
    """Settings take the enum or its string value, as elsewhere in the package."""
    in_dat = np.linspace(1.0, 50.0, 130 * 255).reshape(130, 255)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    by_str = InverseAnscombeTransformer(InverseAnscombeSettings(method=method))(msg_in)
    by_enum = InverseAnscombeTransformer(InverseAnscombeSettings(method=InverseMethod(method)))(msg_in)
    assert np.array_equal(by_str.data, by_enum.data)


def test_inverse_anscombe_rejects_bad_method():
    with pytest.raises(ValueError):
        InverseAnscombeTransformer(InverseAnscombeSettings(method="nope"))(
            AxisArray(np.ones((4, 2)), dims=["time", "ch"])
        )


def test_inverse_anscombe_exact_matches_reference():
    """Spot-check the closed form against hand-evaluated reference values."""
    d = np.array([_D_MIN, 2.0, 5.0, 20.0])
    expected = (d / 2) ** 2 + 0.25 * np.sqrt(1.5) / d - 1.375 / d**2 + 0.625 * np.sqrt(1.5) / d**3 - 0.125
    out = InverseAnscombeTransformer(InverseAnscombeSettings())(AxisArray(d[None, :], dims=["time", "ch"]))
    assert np.allclose(out.data[0], expected)
    # At the forward transform's value for zero counts the closed form is exactly 0.
    assert np.isclose(out.data[0, 0], 0.0, atol=1e-12)


@pytest.mark.parametrize("value", [-5.0, 0.0, 1.0])
def test_inverse_anscombe_exact_floors_at_zero(value: float):
    """Values at or below the zero-count image must not blow up through the D**-3 term."""
    in_dat = np.full((4, 2), value)
    out = InverseAnscombeTransformer(InverseAnscombeSettings())(AxisArray(in_dat, dims=["time", "ch"]))
    assert np.all(np.isfinite(out.data))
    assert np.allclose(out.data, 0.0, atol=1e-12)


@pytest.mark.parametrize("rate", [0.5, 2.0, 20.0])
def test_inverse_anscombe_exact_is_less_biased(rate: float):
    """The whole reason EXACT is the default: recover the Poisson rate at low counts.

    Each inverse maps a *denoised* stabilized value back to a rate, so the property
    under test is ``inverse(E[forward(z)]) == rate``, not ``E[inverse(forward(z))]``.
    """
    rng = np.random.default_rng(7)
    counts = rng.poisson(rate, size=(400000, 1)).astype(float)
    stabilized = AnscombeTransformer()(AxisArray(counts, dims=["time", "ch"]))
    denoised = AxisArray(np.mean(stabilized.data, axis=0, keepdims=True), dims=["time", "ch"])

    def bias(method):
        out = InverseAnscombeTransformer(InverseAnscombeSettings(method=method))(denoised)
        return abs(float(out.data[0, 0]) - rate)

    assert bias(InverseMethod.EXACT) < 0.02
    assert bias(InverseMethod.EXACT) < bias(InverseMethod.ALGEBRAIC)
    if rate < 5.0:
        # The asymptotic inverse only earns its name at higher rates.
        assert bias(InverseMethod.EXACT) < bias(InverseMethod.ASYMPTOTIC)


@pytest.mark.parametrize("min_val", [1, 2])
@pytest.mark.parametrize("max_val", [133, 134])
def test_clip(min_val: float, max_val: float):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = ClipTransformer(ClipSettings(min=min_val, max=max_val))
    msg_out = xformer(msg_in)

    assert all(msg_out.data[np.where(in_dat < min_val)] == min_val)
    assert all(msg_out.data[np.where(in_dat > max_val)] == max_val)


@pytest.mark.parametrize("value", [-100, 0, 100])
@pytest.mark.parametrize("subtrahend", [False, True])
def test_const_difference(value: float, subtrahend: bool):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = ConstDifferenceTransformer(ConstDifferenceSettings(value=value, subtrahend=subtrahend))
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, (in_dat - value) if subtrahend else (value - in_dat))


def test_invert():
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    xformer = InvertTransformer()
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, 1 / in_dat)


@pytest.mark.parametrize("base", [np.e, 2, 10])
@pytest.mark.parametrize("dtype", [int, float])
@pytest.mark.parametrize("clip_zero", [False, True])
def test_log(base: float, dtype, clip_zero: bool):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans).astype(dtype)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    xformer = LogTransformer(LogSettings(base=base, clip_zero=clip_zero))
    msg_out = xformer(msg_in)
    if clip_zero and dtype is float:
        in_dat = np.clip(in_dat, a_min=np.finfo(msg_in.data.dtype).tiny, a_max=None)
    assert np.array_equal(msg_out.data, np.log(in_dat) / np.log(base))


@pytest.mark.parametrize("scale_factor", [0.1, 0.5, 2.0, 10.0])
def test_scale(scale_factor: float):
    n_times = 130
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = ScaleTransformer(ScaleSettings(scale=scale_factor))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == (n_times, n_chans)
    assert np.array_equal(msg_out.data, in_dat * scale_factor)


@pytest.mark.parametrize("exponent", [0.5, 2.0, 3.0])
def test_pow(exponent: float):
    n_times = 130
    n_chans = 255
    in_dat = np.abs(np.arange(n_times * n_chans).reshape(n_times, n_chans)).astype(float) + 1.0
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = PowTransformer(PowSettings(exponent=exponent))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == (n_times, n_chans)
    assert np.allclose(msg_out.data, in_dat**exponent)


def test_abs_empty_time():
    from ezmsg.sigproc.math.abs import AbsTransformer

    proc = AbsTransformer()
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_anscombe_empty_time():
    from ezmsg.sigproc.math.anscombe import AnscombeTransformer

    proc = AnscombeTransformer()
    result = proc(make_empty_msg())
    check_empty_result(result)


@pytest.mark.parametrize("method", list(InverseMethod))
def test_inverse_anscombe_empty_time(method: InverseMethod):
    from ezmsg.sigproc.math.anscombe import InverseAnscombeSettings, InverseAnscombeTransformer

    proc = InverseAnscombeTransformer(InverseAnscombeSettings(method=method))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_clip_empty_time():
    from ezmsg.sigproc.math.clip import ClipSettings, ClipTransformer

    proc = ClipTransformer(ClipSettings(min=0.0, max=1.0))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_const_difference_empty_time():
    from ezmsg.sigproc.math.difference import ConstDifferenceSettings, ConstDifferenceTransformer

    proc = ConstDifferenceTransformer(ConstDifferenceSettings(value=5.0))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_invert_empty_time():
    from ezmsg.sigproc.math.invert import InvertTransformer

    proc = InvertTransformer()
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_log_empty_time():
    from ezmsg.sigproc.math.log import LogSettings, LogTransformer

    proc = LogTransformer(LogSettings(base=np.e, clip_zero=True))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_pow_empty_time():
    from ezmsg.sigproc.math.pow import PowSettings, PowTransformer

    proc = PowTransformer(PowSettings(exponent=2.0))
    result = proc(make_empty_msg())
    check_empty_result(result)


def test_scale_empty_time():
    from ezmsg.sigproc.math.scale import ScaleSettings, ScaleTransformer

    proc = ScaleTransformer(ScaleSettings(scale=2.0))
    result = proc(make_empty_msg())
    check_empty_result(result)


@requires_mlx
@pytest.mark.parametrize("clip_zero", [False, True])
def test_log_mlx_matches_numpy(clip_zero: bool):
    """Log must work on MLX arrays and agree with the NumPy path.

    ``clip_zero=True`` used to reach ``xp.isdtype`` and ``finfo.smallest_normal``,
    neither of which MLX has, so it raised there; and it forced a host sync per
    message via ``bool(xp.any(data <= 0))``, which is now gone.
    """
    import mlx.core as mx

    in_dat = np.linspace(-1.0, 100.0, 130 * 255, dtype=np.float32).reshape(130, 255)
    settings = LogSettings(base=10.0, clip_zero=clip_zero)

    out_np = LogTransformer(settings)(AxisArray(in_dat, dims=["time", "ch"])).data
    out_mx = np.asarray(LogTransformer(settings)(AxisArray(mx.array(in_dat), dims=["time", "ch"])).data)

    assert np.array_equal(np.isnan(out_np), np.isnan(out_mx))
    finite = np.isfinite(out_np) & np.isfinite(out_mx)
    assert np.allclose(out_np[finite], out_mx[finite], rtol=1e-6, atol=1e-6)
    if clip_zero:
        # Nothing may be NaN: every non-positive input was raised to smallest_normal.
        assert not np.any(np.isnan(out_mx))


@requires_mlx
def test_anscombe_mlx_matches_numpy():
    import mlx.core as mx

    in_dat = np.linspace(0.0, 100.0, 130 * 255, dtype=np.float32).reshape(130, 255)

    out_np = AnscombeTransformer()(AxisArray(in_dat, dims=["time", "ch"])).data
    out_mx = np.asarray(AnscombeTransformer()(AxisArray(mx.array(in_dat), dims=["time", "ch"])).data)

    assert np.allclose(out_np, out_mx, rtol=1e-6, atol=1e-6)


@requires_mlx
@pytest.mark.parametrize("method", list(InverseMethod))
def test_inverse_anscombe_mlx_matches_numpy(method: InverseMethod):
    """EXACT reaches ``xp.clip``, so it needs the same array-API care as Log."""
    import mlx.core as mx

    in_dat = np.linspace(0.0, 20.0, 130 * 255, dtype=np.float32).reshape(130, 255)
    xformer = InverseAnscombeTransformer(InverseAnscombeSettings(method=method))

    out_np = xformer(AxisArray(in_dat, dims=["time", "ch"])).data
    out_mx = np.asarray(xformer(AxisArray(mx.array(in_dat), dims=["time", "ch"])).data)

    assert np.allclose(out_np, out_mx, rtol=1e-5, atol=1e-5)
