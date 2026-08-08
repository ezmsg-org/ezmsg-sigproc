"""numpy FIR dispatch: correctness, method selection, and chunk-invariance.

Before this dispatch existed, a numpy FIR went to ``scipy.signal.lfilter``,
which for ``len(a) == 1`` degrades to ``np.apply_along_axis(np.convolve, ...)``
-- one Python-level call per channel. Both replacement paths must reproduce that
result to float roundoff; the tests below pin that, the method selection, and
the chunk-invariance difference between the two paths (which is the property
that matters when an offline run has to reproduce an online one).
"""

import numpy as np
import pytest
import scipy.signal
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.filter import FilterCoefficients, FilterSettings, FilterTransformer

FS = 1000.0


def _taps(n, dtype=np.float64):
    return scipy.signal.firwin(n, 0.2).astype(dtype)


def _msg(data, dims, offset=0.0):
    return AxisArray(data, dims=dims, axes={"time": AxisArray.TimeAxis(fs=FS, offset=offset)}, key="t")


def _make(b, **kw):
    a = np.array([1.0], dtype=b.dtype)
    return FilterTransformer(FilterSettings(axis="time", coefs=FilterCoefficients(b=b, a=a), coef_type="ba", **kw))


def _run(b, data, dims, chunk=None, **kw):
    tf = _make(b, **kw)
    ax = dims.index("time")
    if chunk is None:
        return np.asarray(tf(_msg(data, dims)).data)
    outs = []
    for s in range(0, data.shape[ax], chunk):
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(s, s + chunk)
        outs.append(np.asarray(tf(_msg(data[tuple(sl)], dims, offset=s / FS)).data))
    return np.concatenate(outs, axis=ax)


def _lfilter_reference(b, data, axis):
    """What the old numpy FIR path produced: lfilter with edge-scaled zi."""
    first = tuple(slice(0, 1) if i == axis else slice(None) for i in range(data.ndim))
    zi_base = scipy.signal.lfilter_zi(b, np.array([1.0], dtype=b.dtype))
    n_tail = data.ndim - axis - 1
    expand = (None,) * axis + (slice(None),) + (None,) * n_tail
    tile = data.shape[:axis] + (1,) + data.shape[axis + 1 :]
    zi = np.tile(zi_base[expand], tile) * data[first]
    return scipy.signal.lfilter(b, np.array([1.0], dtype=b.dtype), data, axis=axis, zi=zi)[0]


# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_taps, expected", [(2, "conv1d"), (33, "conv1d"), (63, "conv1d"), (64, "fft"), (129, "fft")])
def test_numpy_fir_selects_method_by_tap_count(n_taps, expected):
    tf = _make(_taps(n_taps))
    tf(_msg(np.random.default_rng(0).standard_normal((8, 500)), ["ch", "time"]))
    assert tf.state.fir_method == expected


def test_threshold_is_configurable():
    """Raising the threshold forces the chunk-invariant time-domain path."""
    data = np.random.default_rng(0).standard_normal((8, 500))
    tf = _make(_taps(129), fir_fft_min_taps=10_000)
    tf(_msg(data, ["ch", "time"]))
    assert tf.state.fir_method == "conv1d"


def test_iir_is_untouched_by_the_fir_dispatch():
    """An actual IIR must not be diverted into either FIR path."""
    b, a = scipy.signal.butter(4, 0.2, output="ba")
    tf = FilterTransformer(FilterSettings(axis="time", coefs=FilterCoefficients(b=b, a=a), coef_type="ba"))
    tf(_msg(np.random.default_rng(0).standard_normal((8, 500)), ["ch", "time"]))
    assert tf.state.fir_method is None


# ---------------------------------------------------------------------------
# Correctness vs the path this replaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_taps", [2, 9, 33, 63, 64, 65, 129, 257])
@pytest.mark.parametrize("dims", [["ch", "time"], ["time", "ch"]])
def test_matches_lfilter_reference(n_taps, dims):
    b = _taps(n_taps)
    data = np.random.default_rng(0).standard_normal((256, 2_000))
    if dims[0] == "time":
        data = np.ascontiguousarray(data.T)
    axis = dims.index("time")

    expected = _lfilter_reference(b, data, axis)
    actual = _run(b, data, dims)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12 * np.abs(expected).max())


def test_matches_lfilter_reference_3d():
    """Time in the middle of a 3-D array."""
    b = _taps(129)
    data = np.random.default_rng(1).standard_normal((4, 1_000, 8))
    expected = _lfilter_reference(b, data, 1)
    actual = _run(b, data, ["ch", "time", "feat"])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize("n_taps", [33, 129])
def test_chunked_matches_one_shot(n_taps):
    """State carry-over across chunks must reconstruct the one-shot result."""
    b = _taps(n_taps)
    data = np.random.default_rng(2).standard_normal((32, 3_000))
    one = _run(b, data, ["ch", "time"])
    chunked = _run(b, data, ["ch", "time"], chunk=250)
    np.testing.assert_allclose(chunked, one, rtol=0, atol=1e-12 * np.abs(one).max())


# ---------------------------------------------------------------------------
# Chunk-invariance: the property that differs between the two paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_time_domain_path_is_chunk_size_invariant(dtype):
    """The time-domain path gives bit-identical output at any chunking.

    This is what makes an offline re-run reproduce an online run exactly, and
    it is the reason fir_fft_min_taps is exposed rather than hard-coded.
    """
    b = _taps(129, dtype)
    data = np.random.default_rng(3).standard_normal((32, 3_000)).astype(dtype)
    kw = {"fir_fft_min_taps": 10_000}  # force time-domain
    one = _run(b, data, ["ch", "time"], **kw)
    c250 = _run(b, data, ["ch", "time"], chunk=250, **kw)
    c333 = _run(b, data, ["ch", "time"], chunk=333, **kw)
    np.testing.assert_array_equal(c250, one)
    np.testing.assert_array_equal(c333, one)


def test_fft_path_is_not_chunk_size_invariant():
    """Documents the tradeoff: the FFT path's rounding depends on chunk length.

    Not a defect -- the transform length follows the chunk length, so the
    roundoff differs. Asserted so the docstring's claim stays honest and so a
    future change that silently made FFT the only path would fail here.
    """
    b = _taps(129, np.float32)
    data = np.random.default_rng(4).standard_normal((32, 3_000)).astype(np.float32)
    one = _run(b, data, ["ch", "time"])
    c250 = _run(b, data, ["ch", "time"], chunk=250)
    assert not np.array_equal(c250, one)
    # ...but still correct to float32 roundoff.
    np.testing.assert_allclose(c250, one, rtol=0, atol=1e-5 * np.abs(one).max())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_tap_fir():
    """M == 0: pure scaling, no state."""
    b = np.array([0.5])
    data = np.random.default_rng(5).standard_normal((4, 100))
    np.testing.assert_allclose(_run(b, data, ["ch", "time"]), data * 0.5)


def test_chunk_shorter_than_filter():
    """Chunks smaller than the filter order still reconstruct the one-shot result."""
    b = _taps(129)
    data = np.random.default_rng(6).standard_normal((8, 1_200))
    one = _run(b, data, ["ch", "time"])
    chunked = _run(b, data, ["ch", "time"], chunk=30)  # 30 < 128 taps of state
    np.testing.assert_allclose(chunked, one, rtol=0, atol=1e-12 * np.abs(one).max())


def test_fft_length_is_a_fast_size():
    """next_fast_len must be applied; the natural N + 2M is often a bad length."""
    from ezmsg.sigproc.filter import _next_fast_len

    # 17 taps on an 8192-sample chunk: N + 2M = 8224 = 2**5 * 257, a large prime
    # factor. This was measured at 56.8 ms vs 24.9 ms at the next fast length.
    assert _next_fast_len(8224) != 8224
    assert _next_fast_len(8224) >= 8224
