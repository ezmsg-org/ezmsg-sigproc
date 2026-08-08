"""Direct-kernel SOS filtering: bit-identity, fallback, and cache invalidation.

This path exists purely to remove scipy's fixed per-call overhead, so the only
acceptable output is one bit-identical to ``scipy.signal.sosfilt``. It also
leans on a private scipy entry point, so the fallback behaviour is as much the
subject of these tests as the fast path itself.
"""

import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.filter import FilterSettings, FilterTransformer
from ezmsg.sigproc.util import sosfilt_direct

FS = 1000.0


def _sos(order=4):
    return scipy.signal.butter(order, [30.0, 200.0], btype="bandpass", fs=FS, output="sos")


def _msg(data, dims=("ch", "time")):
    return AxisArray(data, dims=list(dims), axes={"time": AxisArray.TimeAxis(fs=FS)}, key="t")


# ---------------------------------------------------------------------------
# Bit-identity with scipy
# ---------------------------------------------------------------------------


def test_private_kernel_is_available_and_verified():
    """If this fails the fast path silently disables; worth knowing explicitly."""
    assert sosfilt_direct.available() is True


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "shape, axis",
    [((4, 500), 1), ((500, 4), 0), ((3, 200, 5), 1), ((7,), 0), ((256, 33), 1)],
)
def test_direct_matches_scipy_bit_for_bit(order, dtype, shape, axis):
    sos = _sos(order)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(shape).astype(dtype)
    zi_shape = (sos.shape[0],) + shape[:axis] + (2,) + shape[axis + 1 :]
    zi = rng.standard_normal(zi_shape).astype(dtype)

    expected, expected_zf = scipy.signal.sosfilt(sos, x, axis=axis, zi=zi)
    dtype_p = np.result_type(sos, x, zi)
    actual, actual_zf = sosfilt_direct.DirectSosfilt(sos, dtype_p).apply(x, axis, zi)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_zf, expected_zf)


def test_direct_does_not_mutate_its_input():
    """The kernel works in place; we must be filtering a copy, not the caller's array."""
    sos = _sos()
    x = np.random.default_rng(2).standard_normal((4, 200))
    before = x.copy()
    zi = np.zeros((sos.shape[0], 4, 2))
    sosfilt_direct.DirectSosfilt(sos, np.float64).apply(x, 1, zi)
    np.testing.assert_array_equal(x, before)


def test_negative_axis_is_handled():
    sos = _sos()
    x = np.random.default_rng(3).standard_normal((4, 200))
    zi = np.zeros((sos.shape[0], 4, 2))
    expected = scipy.signal.sosfilt(sos, x, axis=-1, zi=zi)[0]
    actual = sosfilt_direct.DirectSosfilt(sos, np.float64).apply(x, -1, zi)[0]
    np.testing.assert_array_equal(actual, expected)


def test_rejects_malformed_sos():
    with pytest.raises(ValueError, match="n_sections, 6"):
        sosfilt_direct.DirectSosfilt(np.zeros((4, 5)), np.float64)


# ---------------------------------------------------------------------------
# Gating / fallback
# ---------------------------------------------------------------------------


def test_can_apply_rejects_unsupported_dtypes():
    sos = _sos()
    zi = np.zeros((sos.shape[0], 4, 2))
    assert sosfilt_direct.can_apply(sos, np.zeros((4, 10)), zi)
    assert sosfilt_direct.can_apply(sos, np.zeros((4, 10), dtype=np.float32), zi)
    assert not sosfilt_direct.can_apply(sos, np.zeros((4, 10), dtype=np.complex128), zi)
    assert not sosfilt_direct.can_apply(sos, np.zeros((4, 10), dtype=object), zi)


@pytest.mark.skipif(
    np.dtype(np.longdouble) == np.dtype(np.float64),
    reason="longdouble is an alias for float64 on this platform (e.g. Apple Silicon)",
)
def test_can_apply_rejects_longdouble():
    sos = _sos()
    zi = np.zeros((sos.shape[0], 4, 2))
    assert not sosfilt_direct.can_apply(sos, np.zeros((4, 10), dtype=np.longdouble), zi)


def test_can_apply_rejects_non_numpy():
    sos = _sos()
    assert not sosfilt_direct.can_apply(sos, [[1.0, 2.0]], np.zeros((sos.shape[0], 1, 2)))


def test_transformer_falls_back_when_kernel_unavailable(monkeypatch):
    """With the private kernel gone, results must be unchanged."""
    data = np.random.default_rng(4).standard_normal((8, 400))
    fast = FilterTransformer(FilterSettings(axis="time", coefs=_sos(), coef_type="sos"))(_msg(data)).data

    monkeypatch.setattr(sosfilt_direct, "_verified", False)
    slow = FilterTransformer(FilterSettings(axis="time", coefs=_sos(), coef_type="sos"))(_msg(data)).data

    np.testing.assert_array_equal(fast, slow)


def test_setting_disables_the_fast_path():
    data = np.random.default_rng(5).standard_normal((8, 400))
    tf_off = FilterTransformer(FilterSettings(axis="time", coefs=_sos(), coef_type="sos", use_fast_sosfilt=False))
    tf_on = FilterTransformer(FilterSettings(axis="time", coefs=_sos(), coef_type="sos"))
    off, on = tf_off(_msg(data)).data, tf_on(_msg(data)).data

    assert tf_off.state.sos_direct is False, "disabled path should be marked checked-and-unusable"
    assert tf_on.state.sos_direct not in (None, False)
    np.testing.assert_array_equal(off, on)


def test_ba_coefficients_do_not_use_the_sos_path():
    b, a = scipy.signal.butter(4, 0.2, output="ba")
    from ezmsg.sigproc.filter import FilterCoefficients

    tf = FilterTransformer(FilterSettings(axis="time", coefs=FilterCoefficients(b=b, a=a), coef_type="ba"))
    tf(_msg(np.random.default_rng(6).standard_normal((8, 400))))
    assert tf.state.sos_direct is False


# ---------------------------------------------------------------------------
# Cache invalidation -- the regression this nearly shipped with
# ---------------------------------------------------------------------------


def test_update_coefficients_invalidates_the_cached_kernel():
    """DirectSosfilt holds a converted copy of the coefficients.

    A same-length coefficient swap does not reset zi, so without explicit
    invalidation the filter would keep running the *old* coefficients. Caught by
    test_butterworth_update_settings; pinned directly here.
    """
    lo = scipy.signal.butter(4, [30.0, 200.0], btype="bandpass", fs=FS, output="sos")
    hi = scipy.signal.butter(4, [100.0, 400.0], btype="bandpass", fs=FS, output="sos")
    assert lo.shape == hi.shape, "same-length swap is the case that skips the reset"

    data = np.random.default_rng(7).standard_normal((8, 600))

    tf = FilterTransformer(FilterSettings(axis="time", coefs=lo, coef_type="sos"))
    tf(_msg(data))  # prime the cache with `lo`
    assert tf.state.sos_direct not in (None, False)
    tf.update_coefficients(hi)
    assert tf.state.sos_direct is None, "cache must be invalidated on coefficient change"
    after_update = tf(_msg(data)).data

    # Reference: a fresh transformer whose first chunk already ran `lo`, then `hi`.
    ref = FilterTransformer(FilterSettings(axis="time", coefs=lo, coef_type="sos"))
    ref(_msg(data))
    ref.settings = ref.settings.__class__(**{**ref.settings.__dict__, "coefs": hi})
    ref.state.sos_direct = False  # force the public scipy path
    expected = ref(_msg(data)).data

    np.testing.assert_array_equal(after_update, expected)


def test_streaming_matches_one_shot():
    sos = _sos()
    data = np.random.default_rng(8).standard_normal((16, 3_000))
    one = FilterTransformer(FilterSettings(axis="time", coefs=sos, coef_type="sos"))(_msg(data)).data

    tf = FilterTransformer(FilterSettings(axis="time", coefs=sos, coef_type="sos"))
    chunked = np.concatenate([tf(_msg(data[:, s : s + 250])).data for s in range(0, 3_000, 250)], axis=1)

    np.testing.assert_array_equal(chunked, one)
