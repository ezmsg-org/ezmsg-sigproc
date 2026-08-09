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

    assert tf_off.state.sos_direct is None, "disabled path must never build a kernel"
    assert tf_on.state.sos_direct is not None
    np.testing.assert_array_equal(off, on)


def test_ba_coefficients_do_not_use_the_sos_path():
    b, a = scipy.signal.butter(4, 0.2, output="ba")
    from ezmsg.sigproc.filter import FilterCoefficients

    tf = FilterTransformer(FilterSettings(axis="time", coefs=FilterCoefficients(b=b, a=a), coef_type="ba"))
    tf(_msg(np.random.default_rng(6).standard_normal((8, 400))))
    assert tf.state.sos_direct is None


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
    assert tf.state.sos_direct is not None
    tf.update_coefficients(hi)
    assert tf.state.sos_direct is None, "cache must be invalidated on coefficient change"
    assert tf.state.sos_direct_dtype is None, "the dtype key must be cleared with it"
    after_update = tf(_msg(data)).data

    # Reference: the same sequence down the public scipy path, which is
    # bit-identical to the direct one and cannot go stale.
    ref = FilterTransformer(FilterSettings(axis="time", coefs=lo, coef_type="sos", use_fast_sosfilt=False))
    ref(_msg(data))
    ref.update_coefficients(hi)
    expected = ref(_msg(data)).data

    np.testing.assert_array_equal(after_update, expected)


def test_streaming_matches_one_shot():
    sos = _sos()
    data = np.random.default_rng(8).standard_normal((16, 3_000))
    one = FilterTransformer(FilterSettings(axis="time", coefs=sos, coef_type="sos"))(_msg(data)).data

    tf = FilterTransformer(FilterSettings(axis="time", coefs=sos, coef_type="sos"))
    chunked = np.concatenate([tf(_msg(data[:, s : s + 250])).data for s in range(0, 3_000, 250)], axis=1)

    np.testing.assert_array_equal(chunked, one)


# ---------------------------------------------------------------------------
# The cached kernel is keyed on the promoted dtype, not just the coefficients
# ---------------------------------------------------------------------------


def _public(sos, chunks, axis=1):
    """The same chunk sequence down public ``sosfilt``, which is the reference."""
    tf = FilterTransformer(FilterSettings(axis="time", coefs=sos, coef_type="sos", use_fast_sosfilt=False))
    return [np.asarray(tf(_msg(c)).data) for c in chunks]


def test_dtype_widening_mid_stream_matches_public_scipy():
    """A float32 stream that later receives float64 must promote.

    The converted coefficients bake in the dtype they were built for, so a cache
    keyed only on the coefficients kept filtering in float32 and returned float32
    where public ``sosfilt`` returns float64.
    """
    sos = _sos().astype(np.float32)
    rng = np.random.default_rng(20)
    chunks = [rng.standard_normal((4, 200)).astype(np.float32), rng.standard_normal((4, 200))]

    tf = FilterTransformer(FilterSettings(axis="time", coefs=sos, coef_type="sos"))
    got = [np.asarray(tf(_msg(c)).data) for c in chunks]
    want = _public(sos, chunks)

    assert [g.dtype for g in got] == [np.dtype(np.float32), np.dtype(np.float64)]
    for g, w in zip(got, want):
        assert g.dtype == w.dtype
        np.testing.assert_array_equal(g, w)


def test_dtype_narrowing_mid_stream_matches_public_scipy():
    """The reverse direction: float64 then float32 (zi keeps it float64)."""
    sos = _sos().astype(np.float32)
    rng = np.random.default_rng(21)
    chunks = [rng.standard_normal((4, 200)), rng.standard_normal((4, 200)).astype(np.float32)]

    tf = FilterTransformer(FilterSettings(axis="time", coefs=sos, coef_type="sos"))
    got = [np.asarray(tf(_msg(c)).data) for c in chunks]
    for g, w in zip(got, _public(sos, chunks)):
        assert g.dtype == w.dtype
        np.testing.assert_array_equal(g, w)


def test_complex_mid_stream_falls_back_to_public_scipy():
    """Complex input must not be cast down to the dtype the stream started in.

    The kernel handles only real float32/float64, so a cached float32 filter
    silently discarded the imaginary part.
    """
    sos = _sos().astype(np.float32)
    rng = np.random.default_rng(22)
    chunks = [
        rng.standard_normal((4, 200)).astype(np.float32),
        rng.standard_normal((4, 200)) + 1j * rng.standard_normal((4, 200)),
    ]

    tf = FilterTransformer(FilterSettings(axis="time", coefs=sos, coef_type="sos"))
    got = [np.asarray(tf(_msg(c)).data) for c in chunks]

    assert got[1].dtype == np.complex128
    assert np.any(got[1].imag != 0), "the imaginary part must survive"
    assert tf.state.sos_direct is None, "an unsupported dtype must not hold a kernel"
    for g, w in zip(got, _public(sos, chunks)):
        assert g.dtype == w.dtype
        np.testing.assert_array_equal(g, w)


def test_stable_dtype_stream_keeps_one_cached_kernel():
    """The dtype check must not rebuild the kernel on every chunk."""
    sos = _sos()
    rng = np.random.default_rng(23)
    tf = FilterTransformer(FilterSettings(axis="time", coefs=sos, coef_type="sos"))
    tf(_msg(rng.standard_normal((4, 200))))
    first = tf.state.sos_direct
    assert first is not None
    for _ in range(4):
        tf(_msg(rng.standard_normal((4, 200))))
    assert tf.state.sos_direct is first, "cache rebuilt despite an unchanged dtype"


# ---------------------------------------------------------------------------
# Coefficient validation the private kernel does not perform
# ---------------------------------------------------------------------------


def test_construction_rejects_non_unit_a0():
    """The kernel assumes a0 == 1 and filters with whatever it is handed."""
    sos = _sos()
    sos[0, 3] = 2.0
    with pytest.raises(ValueError, match=r"sos\[:, 3\] should be all ones"):
        sosfilt_direct.DirectSosfilt(sos, np.float64)


def test_construction_rejects_empty_and_misshapen_sos():
    with pytest.raises(ValueError, match="at least one section"):
        sosfilt_direct.DirectSosfilt(np.zeros((0, 6)), np.float64)
    with pytest.raises(ValueError, match=r"shape \(n_sections, 6\)"):
        sosfilt_direct.DirectSosfilt(np.zeros((3, 5)), np.float64)
    with pytest.raises(ValueError, match="must be 2D"):
        sosfilt_direct.DirectSosfilt(np.zeros((2, 3, 6)), np.float64)


def test_construction_accepts_1d_sos_like_public_scipy():
    """Public sosfilt treats a bare six-element sos as a single section.

    Rejecting it on ``ndim != 2`` would be stricter than the function this
    replaces. (The transformer itself never gets this far -- ``sosfilt_zi``
    requires 2-D -- but the class must not be the one drawing that line.)
    """
    sos_1d = _sos(order=2)[0]
    assert sos_1d.shape == (6,)
    rng = np.random.default_rng(24)
    x = rng.standard_normal((4, 300))
    zi = rng.standard_normal((1, 4, 2))

    want, want_zf = scipy.signal.sosfilt(sos_1d, x, axis=1, zi=zi)
    got, got_zf = sosfilt_direct.DirectSosfilt(sos_1d, np.float64).apply(x, 1, zi)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got_zf, want_zf)


def test_invalid_coefficients_raise_the_public_scipy_error():
    """A bad same-length update must surface scipy's own error, not be filtered.

    Before validation the direct path accepted sos[0, 3] = 2 and produced an
    answer public scipy refuses to produce at all.
    """
    sos = _sos()
    data = np.random.default_rng(25).standard_normal((4, 300))

    tf = FilterTransformer(FilterSettings(axis="time", coefs=sos, coef_type="sos"))
    tf(_msg(data))  # prime the cache with valid coefficients

    bad = sos.copy()
    bad[0, 3] = 2.0
    tf.update_coefficients(bad)
    with pytest.raises(ValueError, match=r"sos\[:, 3\] should be all ones"):
        tf(_msg(data))
