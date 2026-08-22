import copy
from pathlib import Path

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.affinetransform import (
    AffineTransformSettings,
    AffineTransformTransformer,
    CommonRereferenceSettings,
    CommonRereferenceTransformer,
)
from tests.helpers.empty_time import N_CH, check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg
from tests.helpers.util import assert_messages_equal, requires_mlx


def test_affine_transform():
    n_times = 13
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(
        data=in_dat,
        dims=["time", "ch"],
        axes={"ch": AxisArray.CoordinateAxis(data=np.array([f"ch_{i}" for i in range(n_chans)]), dims=["ch"])},
    )

    backup = [copy.deepcopy(msg_in)]

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=np.eye(n_chans), axis="ch"))
    msg_out = xformer(msg_in)
    assert msg_out.data.shape == in_dat.shape
    assert np.allclose(msg_out.data, in_dat)
    assert not np.may_share_memory(msg_out.data, in_dat)

    assert_messages_equal([msg_in], backup)

    # Call again just to make sure the transformer doesn't crash
    _ = xformer(msg_in)

    # Test with weights from a CSV file.
    csv_path = Path(__file__).parents[1] / "resources" / "xform.csv"
    weights = np.loadtxt(csv_path, delimiter=",")
    expected_out = in_dat @ weights.T
    # Same result: expected_out = np.vstack([(step[None, :] * weights).sum(axis=1) for step in in_dat])

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=csv_path, axis="ch", right_multiply=False))
    msg_out = xformer(msg_in)
    assert np.allclose(msg_out.data, expected_out)
    assert len(msg_out.axes["ch"].data) == weights.shape[0]
    assert (msg_out.axes["ch"].data[:-1] == msg_in.axes["ch"].data).all()

    # Try again as str, not Path
    xformer = AffineTransformTransformer(
        AffineTransformSettings(weights=str(csv_path), axis="ch", right_multiply=False)
    )
    msg_out = xformer(msg_in)
    assert np.allclose(msg_out.data, expected_out)
    assert len(msg_out.axes["ch"].data) == weights.shape[0]

    # Try again as direct ndarray
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", right_multiply=False))
    msg_out = xformer(msg_in)
    assert np.allclose(msg_out.data, expected_out)
    assert len(msg_out.axes["ch"].data) == weights.shape[0]

    # One more time, but we pre-transpose the weights and do not override right_multiply
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights.T, axis="ch", right_multiply=True))
    msg_out = xformer(msg_in)
    assert np.allclose(msg_out.data, expected_out)
    assert len(msg_out.axes["ch"].data) == weights.shape[0]


def test_affine_passthrough():
    n_times = 13
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    backup = [copy.deepcopy(msg_in)]

    xformer = AffineTransformTransformer(AffineTransformSettings(weights="passthrough", axis="does not matter"))
    msg_out = xformer(msg_in)
    # We wouldn't want out_data is in_dat ezmsg pipeline but it's fine for the transformer
    assert msg_out.data is in_dat
    assert_messages_equal([msg_out], backup)


def test_affine_invalid_kernel():
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=np.eye(4), axis="ch", kernel="fastest"))
    with pytest.raises(ValueError, match="kernel must be one of"):
        xformer(AxisArray(np.zeros((3, 4)), dims=["time", "ch"]))


def test_common_rereference():
    n_times = 300
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    backup = [copy.deepcopy(msg_in)]

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", include_current=True))
    msg_out = xformer(msg_in)
    assert np.allclose(
        msg_out.data,
        msg_in.data - np.mean(msg_in.data, axis=1, keepdims=True),
    )

    assert_messages_equal([msg_in], backup)

    # Use a slow deliberate way of calculating the CAR uniquely for each channel, excluding itself.
    #  common_rereference uses a faster way of doing this, but we test against something intuitive.
    expected_out = []
    for ch_ix in range(n_chans):
        idx = np.arange(n_chans)
        idx = np.hstack((idx[:ch_ix], idx[ch_ix + 1 :]))
        expected_out.append(msg_in.data[..., ch_ix] - np.mean(msg_in.data[..., idx], axis=1))
    expected_out = np.stack(expected_out).T

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", include_current=False))
    msg_out = xformer(msg_in)
    assert np.allclose(msg_out.data, expected_out)


def test_common_rereference_preserves_float_dtype():
    """float32 in, float32 out. Promoting to float64 doubles the bandwidth of
    every downstream stage for no benefit; integers still promote."""
    rng = np.random.default_rng(0)
    for groups in (None, [[0, 1, 2, 3], [4, 5, 6, 7]]):
        xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", channel_groups=groups))
        out = xformer(AxisArray(rng.standard_normal((20, 8)).astype(np.float32), dims=["time", "ch"]))
        assert out.data.dtype == np.float32
        xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", channel_groups=groups))
        out = xformer(AxisArray(np.arange(20 * 8, dtype=np.int16).reshape(20, 8), dims=["time", "ch"]))
        assert out.data.dtype == np.float64


def test_common_rereference_groups():
    n_times = 300
    n_chans = 8
    rng = np.random.default_rng(42)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    group_a = [0, 1, 2, 3]
    group_b = [4, 5, 6, 7]
    groups = [group_a, group_b]

    # --- include_current=True ---
    xformer = CommonRereferenceTransformer(
        CommonRereferenceSettings(mode="mean", axis="ch", include_current=True, channel_groups=groups)
    )
    msg_out = xformer(msg_in)

    # Expected: per-group CAR
    expected = np.zeros_like(in_dat)
    for group in groups:
        group_data = in_dat[:, group]
        ref = np.mean(group_data, axis=1, keepdims=True)
        expected[:, group] = group_data - ref

    assert np.allclose(msg_out.data, expected)
    assert not np.may_share_memory(msg_out.data, in_dat)

    # --- include_current=False ---
    xformer = CommonRereferenceTransformer(
        CommonRereferenceSettings(mode="mean", axis="ch", include_current=False, channel_groups=groups)
    )
    msg_out = xformer(msg_in)

    # Expected: per-group CAR excluding current channel (slow deliberate way)
    expected = np.zeros_like(in_dat)
    for group in groups:
        group_data = in_dat[:, group]
        N = len(group)
        for i, ch in enumerate(group):
            others = [j for j in range(N) if j != i]
            ref = np.mean(group_data[:, others], axis=1)
            expected[:, ch] = group_data[:, i] - ref

    assert np.allclose(msg_out.data, expected)


def test_common_rereference_unequal_groups_leave_one_out():
    """Groups of different sizes need a per-channel N/(N-1) gain, not a scalar."""
    rng = np.random.default_rng(9)
    in_dat = rng.standard_normal((100, 8))
    groups = [[0, 1, 2], [3, 4, 5, 6, 7]]
    xformer = CommonRereferenceTransformer(
        CommonRereferenceSettings(mode="mean", axis="ch", include_current=False, channel_groups=groups)
    )
    msg_out = xformer(AxisArray(in_dat, dims=["time", "ch"]))

    expected = np.zeros_like(in_dat)
    for group in groups:
        block = in_dat[:, group]
        loo = (block.sum(axis=1, keepdims=True) - block) / (len(group) - 1)
        expected[:, group] = block - loo
    assert np.allclose(msg_out.data, expected)


def test_common_rereference_ungrouped_channels_pass_through():
    """Channels in no group are left alone, matching car_matrix, which leaves
    them identity. (They used to be silently zeroed.)"""
    rng = np.random.default_rng(4)
    in_dat = rng.standard_normal((50, 6))
    groups = [[0, 1], [2, 3]]
    expected = in_dat.copy()
    for group in groups:
        expected[:, group] = in_dat[:, group] - in_dat[:, group].mean(axis=1, keepdims=True)

    for mode in ("mean", "median"):
        xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode=mode, axis="ch", channel_groups=groups))
        msg_out = xformer(AxisArray(in_dat, dims=["time", "ch"]))
        assert np.allclose(msg_out.data[:, 4:], in_dat[:, 4:])
        if mode == "mean":
            assert np.allclose(msg_out.data, expected)


def test_common_rereference_median():
    rng = np.random.default_rng(6)
    in_dat = rng.standard_normal((80, 6))
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="median", axis="ch"))
    assert np.allclose(xformer(msg_in).data, in_dat - np.median(in_dat, axis=1, keepdims=True))

    groups = [[0, 2, 4], [1, 3, 5]]
    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="median", axis="ch", channel_groups=groups))
    expected = np.zeros_like(in_dat)
    for group in groups:
        block = in_dat[:, group]
        expected[:, group] = block - np.median(block, axis=1, keepdims=True)
    assert np.allclose(xformer(msg_in).data, expected)


def test_common_rereference_non_last_axis():
    """Channel-major (ch, time) chunks, as offline pipelines produce."""
    rng = np.random.default_rng(8)
    in_dat = rng.standard_normal((8, 200))
    groups = [[0, 2, 4, 6], [1, 3, 5, 7]]
    expected = np.zeros_like(in_dat)
    for group in groups:
        block = in_dat[group]
        expected[group] = block - block.mean(axis=0, keepdims=True)

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", channel_groups=groups))
    msg_out = xformer(AxisArray(in_dat, dims=["ch", "time"]))
    assert msg_out.data.shape == in_dat.shape
    assert np.allclose(msg_out.data, expected)


def _banked_ch_axis(banks: list[str], arrays: list[int] | None = None):
    """Structured ch CoordinateAxis like ezmsg-blackrock ChannelMap emits."""
    dt = np.dtype([("label", "U16"), ("bank", "U1"), ("elec", "i4"), ("array", "i4")])
    ch = np.zeros(len(banks), dtype=dt)
    ch["bank"] = banks
    ch["elec"] = list(range(1, len(banks) + 1))
    ch["array"] = arrays if arrays is not None else [0] * len(banks)
    ch["label"] = [f"ch{i}" for i in range(len(banks))]
    return AxisArray.CoordinateAxis(data=ch, dims=["ch"])


def test_common_rereference_group_by_field():
    """channel_groups='bank' derives per-bank groups from a structured ch axis."""
    n_times = 300
    banks = ["A", "A", "A", "A", "B", "B", "B", "B"]
    rng = np.random.default_rng(42)
    in_dat = rng.standard_normal((n_times, len(banks)))
    msg_in = AxisArray(in_dat, dims=["time", "ch"], axes={"ch": _banked_ch_axis(banks)})

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", channel_groups="bank"))
    msg_out = xformer(msg_in)

    # Expected: per-bank CAR (bank A = ch 0-3, bank B = ch 4-7)
    expected = np.zeros_like(in_dat)
    for group in ([0, 1, 2, 3], [4, 5, 6, 7]):
        cd = in_dat[:, group]
        expected[:, group] = cd - np.mean(cd, axis=1, keepdims=True)
    assert np.allclose(msg_out.data, expected)


def test_common_rereference_group_by_multiple_fields():
    """A bank label that repeats across arrays must not merge the two arrays."""
    rng = np.random.default_rng(13)
    banks = ["A", "A", "A", "A"]
    arrays = [0, 0, 1, 1]
    in_dat = rng.standard_normal((60, 4))
    msg_in = AxisArray(in_dat, dims=["time", "ch"], axes={"ch": _banked_ch_axis(banks, arrays)})

    xformer = CommonRereferenceTransformer(
        CommonRereferenceSettings(mode="mean", axis="ch", channel_groups=["array", "bank"])
    )
    expected = np.zeros_like(in_dat)
    for group in ([0, 1], [2, 3]):
        cd = in_dat[:, group]
        expected[:, group] = cd - np.mean(cd, axis=1, keepdims=True)
    assert np.allclose(xformer(msg_in).data, expected)


def test_common_rereference_group_by_field_fallback():
    """A field spec with no such field falls back to global CAR."""
    n_times = 50
    n_chans = 8
    rng = np.random.default_rng(0)
    in_dat = rng.standard_normal((n_times, n_chans))
    # Unstructured (label-only) ch axis -> no 'bank' field available.
    ch_ax = AxisArray.CoordinateAxis(data=np.array([str(i) for i in range(n_chans)]), dims=["ch"])
    msg_in = AxisArray(in_dat, dims=["time", "ch"], axes={"ch": ch_ax})

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", channel_groups="bank"))
    msg_out = xformer(msg_in)
    assert np.allclose(msg_out.data, in_dat - in_dat.mean(axis=1, keepdims=True))


def test_common_rereference_group_by_field_interleaved():
    """Per-bank CAR works when channels of a bank are non-contiguous."""
    n_times = 200
    banks = ["A", "B", "A", "B", "A", "B"]  # interleaved -> non-contiguous groups
    rng = np.random.default_rng(7)
    in_dat = rng.standard_normal((n_times, len(banks)))
    msg_in = AxisArray(in_dat, dims=["time", "ch"], axes={"ch": _banked_ch_axis(banks)})

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", channel_groups="bank"))
    msg_out = xformer(msg_in)

    expected = np.zeros_like(in_dat)
    for group in ([0, 2, 4], [1, 3, 5]):
        cd = in_dat[:, group]
        expected[:, group] = cd - np.mean(cd, axis=1, keepdims=True)
    assert np.allclose(msg_out.data, expected)


def test_common_rereference_group_by_field_exclude_current():
    """A field spec composes with include_current=False (per-bank, leave-one-out)."""
    n_times = 100
    banks = ["A", "A", "A", "B", "B", "B"]
    rng = np.random.default_rng(3)
    in_dat = rng.standard_normal((n_times, len(banks)))
    msg_in = AxisArray(in_dat, dims=["time", "ch"], axes={"ch": _banked_ch_axis(banks)})

    xformer = CommonRereferenceTransformer(
        CommonRereferenceSettings(mode="mean", axis="ch", channel_groups="bank", include_current=False)
    )
    msg_out = xformer(msg_in)

    expected = np.zeros_like(in_dat)
    for group in ([0, 1, 2], [3, 4, 5]):
        cd = in_dat[:, group]
        n = cd.shape[1]
        # leave-one-out mean within the bank: (sum - self) / (n - 1)
        loo_ref = (cd.sum(axis=1, keepdims=True) - cd) / (n - 1)
        expected[:, group] = cd - loo_ref
    assert np.allclose(msg_out.data, expected)


def test_common_rereference_singleton_group_exclude_current():
    """A lone channel in a derived bank + include_current=False must not divide by
    N-1 == 0. The singleton passes through unchanged; larger banks still do LOO."""
    n_times = 100
    banks = ["A", "B", "B", "B"]  # bank A is a single channel
    rng = np.random.default_rng(5)
    in_dat = rng.standard_normal((n_times, len(banks)))
    msg_in = AxisArray(in_dat, dims=["time", "ch"], axes={"ch": _banked_ch_axis(banks)})

    xformer = CommonRereferenceTransformer(
        CommonRereferenceSettings(mode="mean", axis="ch", channel_groups="bank", include_current=False)
    )
    msg_out = xformer(msg_in)  # must not raise ZeroDivisionError

    expected = np.zeros_like(in_dat)
    # Lone channel: no other channels to reference -> unchanged.
    expected[:, 0] = in_dat[:, 0]
    # Bank B: leave-one-out mean within the bank.
    cd = in_dat[:, [1, 2, 3]]
    loo_ref = (cd.sum(axis=1, keepdims=True) - cd) / (cd.shape[1] - 1)
    expected[:, [1, 2, 3]] = cd - loo_ref
    assert np.allclose(msg_out.data, expected)


def test_common_rereference_all_singleton_groups_exclude_current():
    """Every group a singleton -> nothing to reference against -> passthrough."""
    rng = np.random.default_rng(15)
    in_dat = rng.standard_normal((30, 3))
    xformer = CommonRereferenceTransformer(
        CommonRereferenceSettings(mode="mean", axis="ch", channel_groups=[[0], [1], [2]], include_current=False)
    )
    assert np.array_equal(xformer(AxisArray(in_dat, dims=["time", "ch"])).data, in_dat)


def test_common_rereference_field_values_change_is_not_detected():
    """Intentional concession: a live bank remap at fixed key + channel count is
    NOT re-derived. _hash_message folds only an O(1) "field present" boolean, not
    the field's bytes, to keep the per-message hash from scaling with channel
    count. A genuine remap on real hardware arrives with a new key or channel
    count (see the escape-hatch assertion below)."""
    n_times = 80
    rng = np.random.default_rng(11)
    in_dat = rng.standard_normal((n_times, 4))

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", channel_groups="bank"))

    # First layout: two banks of two.
    msg1 = AxisArray(in_dat, dims=["time", "ch"], axes={"ch": _banked_ch_axis(["A", "A", "B", "B"])}, key="dev")
    xformer(msg1)
    assert [list(g) for g in xformer._state.groups] == [[0, 1], [2, 3]]

    # Same key and channel count, different bank assignment -> hash unchanged,
    # so the cached groups are (deliberately) NOT re-derived.
    msg2 = AxisArray(in_dat, dims=["time", "ch"], axes={"ch": _banked_ch_axis(["A", "B", "A", "B"])}, key="dev")
    xformer(msg2)
    assert [list(g) for g in xformer._state.groups] == [[0, 1], [2, 3]]

    # Escape hatch: a new key (as a real remap would carry) forces re-derivation.
    msg3 = AxisArray(in_dat, dims=["time", "ch"], axes={"ch": _banked_ch_axis(["A", "B", "A", "B"])}, key="dev2")
    xformer(msg3)
    assert [list(g) for g in xformer._state.groups] == [[0, 2], [1, 3]]


def test_common_rereference_explicit_groups_beat_field():
    """One explicit all-channel group overrides what a field spec would derive."""
    n_times = 50
    banks = ["A", "A", "A", "A", "B", "B", "B", "B"]
    rng = np.random.default_rng(1)
    in_dat = rng.standard_normal((n_times, len(banks)))
    msg_in = AxisArray(in_dat, dims=["time", "ch"], axes={"ch": _banked_ch_axis(banks)})

    xformer = CommonRereferenceTransformer(
        CommonRereferenceSettings(mode="mean", axis="ch", channel_groups=[list(range(len(banks)))])
    )
    msg_out = xformer(msg_in)
    assert np.allclose(msg_out.data, in_dat - in_dat.mean(axis=1, keepdims=True))


def test_common_rereference_invalid_groups():
    msg_in = AxisArray(np.zeros((10, 4)), dims=["time", "ch"])
    with pytest.raises(ValueError, match="out-of-range"):
        CommonRereferenceTransformer(CommonRereferenceSettings(axis="ch", channel_groups=[[0, 4]]))(msg_in)
    with pytest.raises(ValueError, match="overlap"):
        CommonRereferenceTransformer(CommonRereferenceSettings(axis="ch", channel_groups=[[0, 1], [1]]))(msg_in)


def test_car_passthrough():
    n_times = 300
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="passthrough"))
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, in_dat)
    assert np.may_share_memory(msg_out.data, in_dat)


# --- Block-diagonal matmul tests ---
#
# Which kernel `auto` picks is a performance decision (issue #210) that depends
# on channel count and chunk length, so correctness tests force `kernel="blocks"`
# to pin the block path. Tests that assert a *choice* say so explicitly.


def _make_block_diagonal_weights(block_sizes: list[int], rng=None) -> np.ndarray:
    """Helper: create a block-diagonal weight matrix from random blocks."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = sum(block_sizes)
    weights = np.zeros((n, n))
    offset = 0
    for size in block_sizes:
        weights[offset : offset + size, offset : offset + size] = rng.standard_normal((size, size))
        offset += size
    return weights


def _blocked(weights, msg, **kwargs):
    """Run through the forced block kernel and assert it really was used."""
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel="blocks", **kwargs))
    out = xformer(msg)
    assert xformer._state.blocks is not None and len(xformer._state.blocks) > 1
    assert xformer._state.weights is None
    return out


def test_block_diagonal_matches_dense():
    """The block kernel and the dense kernel agree, and `auto` agrees with both."""
    n_times = 30
    n_chans = 128
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights([64, 64], rng=rng)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])
    expected = in_dat @ weights

    assert np.allclose(_blocked(weights, msg_in).data, expected)
    for kernel in ("auto", "dense"):
        xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel=kernel))
        assert np.allclose(xformer(msg_in).data, expected)


def test_block_diagonal_selected_when_it_pays():
    """`auto` routes wide block-diagonal weights into the block kernel and small
    ones into a dense matmul (issue #210)."""
    rng = np.random.default_rng(42)

    wide = _make_block_diagonal_weights([64] * 16, rng=rng)
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=wide, axis="ch"))
    xformer(AxisArray(rng.standard_normal((30, 1024)), dims=["time", "ch"]))
    assert xformer._state.blocks is not None

    narrow = _make_block_diagonal_weights([16] * 8, rng=rng)
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=narrow, axis="ch"))
    xformer(AxisArray(rng.standard_normal((30, 128)), dims=["time", "ch"]))
    assert xformer._state.blocks is None


def test_block_diagonal_non_contiguous_blocks():
    """Regression for issue #198.

    W is genuinely block-diagonal over two *non-contiguous* channel groups. The
    block structure is derived from W, so no caller-supplied grouping can make
    the result disagree with a dense matmul -- which is exactly what used to
    happen, silently, when a hint split a true block in two.
    """
    rng = np.random.default_rng(0)
    n = 128
    groups = [list(range(0, 32)) + list(range(96, 128)), list(range(32, 96))]
    weights = np.zeros((n, n))
    for group in groups:
        weights[np.ix_(group, group)] = rng.standard_normal((len(group), len(group)))

    in_dat = rng.standard_normal((10, n))
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    expected = in_dat @ weights

    # A grouping that splits the true blocks (and one that duplicates them) can
    # no longer be supplied for array weights, but every kernel must still agree.
    for kernel in ("auto", "dense", "blocks"):
        xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel=kernel))
        assert np.allclose(xformer(msg_in).data, expected), kernel

    # Forced blocks must reach it via a channel permutation.
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel="blocks"))
    xformer(msg_in)
    assert len(xformer._state.blocks) == 2
    assert xformer._state.in_perm is not None


def test_block_diagonal_channel_groups_ignored_for_array_weights():
    """A grouping argument cannot change the answer for explicit weights."""
    rng = np.random.default_rng(3)
    weights = _make_block_diagonal_weights([64] * 4, rng=rng)
    in_dat = rng.standard_normal((30, 256))
    msg_in = AxisArray(in_dat, dims=["time", "ch"])
    expected = in_dat @ weights

    for groups in (None, [list(range(0, 32)), list(range(32, 256))], [list(range(256))]):
        xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", channel_groups=groups))
        assert np.allclose(xformer(msg_in).data, expected)


def test_block_diagonal_unsorted_channels():
    """Test with channels interleaved across blocks (not sorted by block)."""
    n_times = 30
    n_chans = 128
    rng = np.random.default_rng(42)

    sorted_weights = _make_block_diagonal_weights([64, 64], rng=rng)
    perm = np.arange(n_chans)
    rng.shuffle(perm)
    weights = sorted_weights[np.ix_(perm, perm)]

    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])
    expected = in_dat @ weights

    out = _blocked(weights, msg_in)
    assert out.data.shape == expected.shape
    assert np.allclose(out.data, expected)


def test_block_diagonal_many_blocks():
    """Test with many small blocks (8 blocks of 32 channels)."""
    n_times = 30
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights([32] * 8, rng=rng)
    in_dat = rng.standard_normal((n_times, 256))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    assert np.allclose(_blocked(weights, msg_in).data, in_dat @ weights)


def test_block_diagonal_unequal_block_sizes():
    """Test with blocks of different sizes."""
    n_times = 30
    block_sizes = [32, 64, 96]
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights(block_sizes, rng=rng)
    in_dat = rng.standard_normal((n_times, sum(block_sizes)))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    assert np.allclose(_blocked(weights, msg_in).data, in_dat @ weights)


def test_block_diagonal_not_triggered_for_dense():
    """Test that a fully-connected weight matrix falls back to standard matmul."""
    n_chans = 64
    rng = np.random.default_rng(42)
    weights = rng.standard_normal((n_chans, n_chans))

    in_dat = rng.standard_normal((10, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    for kernel in ("auto", "blocks"):
        xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel=kernel))
        msg_out = xformer(msg_in)
        assert np.allclose(msg_out.data, in_dat @ weights)
        # There is no structure to find, so even a forced request gets dense.
        assert xformer._state.blocks is None
        assert xformer._state.weights is not None


def test_block_diagonal_non_last_axis():
    """Test block-diagonal with the target axis not being the last axis."""
    n_times = 30
    n_features = 5
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights([64, 64], rng=rng)

    # Data shape: (n_times, n_chans, n_features) -- ch is the middle axis
    in_dat = rng.standard_normal((n_times, 128, n_features))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch", "feat"])

    # Expected: move ch to last, matmul, move back
    data_perm = np.transpose(in_dat, (0, 2, 1))
    expected = np.transpose(data_perm @ weights, (0, 2, 1))

    out = _blocked(weights, msg_in)
    assert out.data.shape == expected.shape
    assert np.allclose(out.data, expected)


def test_block_diagonal_channel_major():
    """(ch, time) chunks, as offline pipelines produce."""
    rng = np.random.default_rng(42)
    weights = _make_block_diagonal_weights([64, 64], rng=rng)
    in_dat = rng.standard_normal((128, 200))
    msg_in = AxisArray(data=in_dat, dims=["ch", "time"])

    out = _blocked(weights, msg_in)
    assert out.data.shape == in_dat.shape
    assert np.allclose(out.data, (in_dat.T @ weights).T)


def test_block_diagonal_right_multiply_false():
    """Test block-diagonal with right_multiply=False."""
    n_times = 30
    rng = np.random.default_rng(42)

    raw_weights = _make_block_diagonal_weights([64, 64], rng=rng)
    in_dat = rng.standard_normal((n_times, 128))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    out = _blocked(raw_weights, msg_in, right_multiply=False)
    assert np.allclose(out.data, in_dat @ raw_weights.T)


def test_block_diagonal_identity_preserves_data():
    """Test that block-diagonal identity matrices act as identity."""
    n_times = 20
    n_chans = 128
    rng = np.random.default_rng(42)

    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    for kernel in ("auto", "blocks"):
        xformer = AffineTransformTransformer(AffineTransformSettings(weights=np.eye(n_chans), axis="ch", kernel=kernel))
        assert np.allclose(xformer(msg_in).data, in_dat)


def test_block_diagonal_all_zero_channels():
    """Channels with all-zero weight rows and columns output zero, and the
    surrounding block structure is still found."""
    n_times = 30
    n_chans = 6
    rng = np.random.default_rng(42)

    weights = np.zeros((n_chans, n_chans))
    weights[:2, :2] = rng.standard_normal((2, 2))
    weights[4:, 4:] = rng.standard_normal((2, 2))

    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])
    expected = in_dat @ weights  # channels 2,3 output should be zero

    out = _blocked(weights, msg_in)
    assert out.data.shape == expected.shape
    assert np.allclose(out.data, expected)
    assert np.all(out.data[:, 2] == 0)
    assert np.all(out.data[:, 3] == 0)


def test_block_diagonal_streaming():
    """Test that block-diagonal works across multiple messages (streaming)."""
    n_chans = 128
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights([64, 64], rng=rng)
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel="blocks"))

    for _ in range(5):
        in_dat = rng.standard_normal((30, n_chans))
        msg_in = AxisArray(data=in_dat, dims=["time", "ch"])
        assert np.allclose(xformer(msg_in).data, in_dat @ weights)


def test_set_weights_keeps_structure():
    """An adaptive filter refreshing values under a fixed sparsity pattern
    should not have to re-derive the blocking."""
    rng = np.random.default_rng(17)
    weights = _make_block_diagonal_weights([64, 64], rng=rng)
    in_dat = rng.standard_normal((30, 128))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel="blocks"))
    xformer(msg_in)
    n_blocks = len(xformer._state.blocks)

    updated = _make_block_diagonal_weights([64, 64], rng=rng)
    xformer.set_weights(updated)
    assert len(xformer._state.blocks) == n_blocks
    assert np.allclose(xformer(msg_in).data, in_dat @ updated)

    # Re-deriving from a now-dense matrix drops back to the dense kernel.
    xformer.set_weights(rng.standard_normal((128, 128)), recalc_structure=True)
    assert xformer._state.blocks is None


# --- Non-square block-diagonal tests ---


def _make_block_diagonal_weights_nonsquare(block_shapes: list[tuple[int, int]], rng=None) -> np.ndarray:
    """Helper: create a non-square block-diagonal weight matrix."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_in = sum(s[0] for s in block_shapes)
    n_out = sum(s[1] for s in block_shapes)
    weights = np.zeros((n_in, n_out))
    in_offset = 0
    out_offset = 0
    for rows, cols in block_shapes:
        weights[in_offset : in_offset + rows, out_offset : out_offset + cols] = rng.standard_normal((rows, cols))
        in_offset += rows
        out_offset += cols
    return weights


def test_nonsquare_blocks():
    """Non-square block-diagonal matrix: 4 blocks of 64 input -> 10 output."""
    n_times = 30
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights_nonsquare([(64, 10)] * 4, rng=rng)
    assert weights.shape == (256, 40)

    in_dat = rng.standard_normal((n_times, 256))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    out = _blocked(weights, msg_in)
    assert out.data.shape == (n_times, 40)
    assert np.allclose(out.data, in_dat @ weights)


def test_nonsquare_unequal_blocks():
    """Test non-square with blocks of different shapes."""
    n_times = 30
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights_nonsquare([(64, 10), (96, 20), (32, 5)], rng=rng)
    assert weights.shape == (192, 35)

    in_dat = rng.standard_normal((n_times, 192))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    out = _blocked(weights, msg_in)
    assert out.data.shape == (n_times, 35)
    assert np.allclose(out.data, in_dat @ weights)


def test_nonsquare_shuffled():
    """Test non-square block-diagonal with shuffled input/output channels."""
    n_times = 30
    rng = np.random.default_rng(42)

    sorted_weights = _make_block_diagonal_weights_nonsquare([(64, 10)] * 2, rng=rng)
    in_perm = np.arange(128)
    rng.shuffle(in_perm)
    out_perm = np.arange(20)
    rng.shuffle(out_perm)
    weights = sorted_weights[np.ix_(in_perm, out_perm)]

    in_dat = rng.standard_normal((n_times, 128))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    out = _blocked(weights, msg_in)
    assert out.data.shape == (n_times, 20)
    assert np.allclose(out.data, in_dat @ weights)


def test_nonsquare_non_last_axis():
    """Test non-square block-diagonal with the target axis not being the last axis."""
    n_times = 30
    n_features = 5
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights_nonsquare([(64, 10)] * 2, rng=rng)

    in_dat = rng.standard_normal((n_times, 128, n_features))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch", "feat"])

    data_perm = np.transpose(in_dat, (0, 2, 1))
    expected = np.transpose(data_perm @ weights, (0, 2, 1))

    out = _blocked(weights, msg_in)
    assert out.data.shape == expected.shape
    assert np.allclose(out.data, expected)


def test_offset_row_weights_use_dense():
    """[A|B] weights augment the input with a ones column, which the block
    kernel does not implement -- it must not be selected."""
    rng = np.random.default_rng(21)
    n_chans = 8
    weights = np.vstack([np.eye(n_chans), rng.standard_normal(n_chans)])
    in_dat = rng.standard_normal((10, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    for kernel in ("auto", "blocks"):
        xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel=kernel))
        msg_out = xformer(msg_in)
        assert xformer._state.blocks is None
        assert np.allclose(msg_out.data, in_dat @ weights[:-1] + weights[-1])


# --- Callable weights tests ---


def test_affine_callable_weights():
    """Test that a callable weights factory produces correct results."""
    n_times = 13
    n_chans = 8
    n_out = 4
    rng = np.random.default_rng(42)

    # Pre-generate a fixed weight matrix so we can verify the result
    fixed_weights = rng.standard_normal((n_chans, n_out))

    def weight_factory(n_in: int) -> np.ndarray:
        assert n_in == n_chans
        return fixed_weights

    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])
    expected = in_dat @ fixed_weights

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weight_factory, axis="ch"))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == (n_times, n_out)
    assert np.allclose(msg_out.data, expected)


def test_affine_callable_weights_receives_groups():
    """A two-argument factory is handed the resolved channel groups, so it can
    build weights from metadata it could not otherwise see."""
    banks = ["A", "A", "B", "B"]
    seen = {}

    def weight_factory(n_in: int, groups) -> np.ndarray:
        seen["n_in"] = n_in
        seen["groups"] = groups
        return np.eye(n_in)

    msg_in = AxisArray(data=np.ones((5, 4)), dims=["time", "ch"], axes={"ch": _banked_ch_axis(banks)})
    xformer = AffineTransformTransformer(
        AffineTransformSettings(weights=weight_factory, axis="ch", channel_groups="bank")
    )
    xformer(msg_in)
    assert seen["n_in"] == 4
    assert seen["groups"] == [[0, 1], [2, 3]]


def test_affine_callable_weights_dimension_change():
    """Test that changing the axis length triggers state reset and re-calls the callable."""
    call_log = []

    def weight_factory(n_in: int) -> np.ndarray:
        call_log.append(n_in)
        return np.eye(n_in)

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weight_factory, axis="ch"))

    # First message: 8 channels
    msg_8 = AxisArray(data=np.ones((5, 8)), dims=["time", "ch"])
    out_8 = xformer(msg_8)
    assert out_8.data.shape == (5, 8)
    assert np.allclose(out_8.data, msg_8.data)
    assert call_log == [8]

    # Second message: still 8 channels — should NOT re-call the factory
    out_8b = xformer(msg_8)
    assert np.allclose(out_8b.data, msg_8.data)
    assert call_log == [8]

    # Third message: 10 channels — dimension change triggers reset + re-call
    msg_10 = AxisArray(data=np.ones((5, 10)), dims=["time", "ch"])
    out_10 = xformer(msg_10)
    assert out_10.data.shape == (5, 10)
    assert np.allclose(out_10.data, msg_10.data)
    assert call_log == [8, 10]


def test_affine_callable_identity_factory():
    """Test a simple identity-matrix factory across multiple streaming messages."""
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=lambda n: np.eye(n), axis="ch"))

    rng = np.random.default_rng(42)
    for _ in range(5):
        in_dat = rng.standard_normal((10, 16))
        msg_in = AxisArray(data=in_dat, dims=["time", "ch"])
        msg_out = xformer(msg_in)
        assert np.allclose(msg_out.data, in_dat)


def test_affine_callable_with_right_multiply_false():
    """Test callable weights with right_multiply=False."""
    n_chans = 8
    rng = np.random.default_rng(42)
    # Factory returns (n_out, n_in) shaped weights; right_multiply=False transposes them
    raw = rng.standard_normal((4, n_chans))

    xformer = AffineTransformTransformer(
        AffineTransformSettings(weights=lambda n: raw, axis="ch", right_multiply=False)
    )

    in_dat = rng.standard_normal((10, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])
    expected = in_dat @ raw.T

    msg_out = xformer(msg_in)
    assert msg_out.data.shape == expected.shape
    assert np.allclose(msg_out.data, expected)


def test_affine_kind_weights_invalid_groups():
    """Groups that build the matrix are still validated."""
    msg_in = AxisArray(data=np.zeros((10, 4)), dims=["time", "ch"])
    xformer = AffineTransformTransformer(
        AffineTransformSettings(weights="car", axis="ch", channel_groups=[[0, 1], [4]])
    )
    with pytest.raises(ValueError, match="out-of-range"):
        xformer(msg_in)


def test_affine_empty_square():
    weights = np.eye(N_CH)
    proc = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_affine_empty_nonsquare():
    weights = np.random.randn(N_CH, 2)
    proc = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)


def test_affine_empty_passthrough():
    proc = AffineTransformTransformer(AffineTransformSettings(weights="passthrough", axis="ch"))
    empty = make_empty_msg()
    result = proc(empty)
    check_empty_result(result)


def test_affine_empty_first():
    weights = np.eye(N_CH)
    proc = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_affine_empty_blocks():
    """An empty chunk through the block kernel returns a correctly shaped empty."""
    weights = _make_block_diagonal_weights([N_CH // 2, N_CH - N_CH // 2])
    proc = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel="blocks"))
    normal = make_msg()
    _ = proc(normal)
    assert proc._state.blocks is not None
    check_empty_result(proc(make_empty_msg()))
    check_state_not_corrupted(proc, normal)


def test_common_rereference_empty_mean():
    proc = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_common_rereference_empty_grouped():
    proc = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", channel_groups=[[0, 2]]))
    normal = make_msg()
    _ = proc(normal)
    check_empty_result(proc(make_empty_msg()))
    check_state_not_corrupted(proc, normal)


def test_common_rereference_empty_passthrough():
    proc = CommonRereferenceTransformer(CommonRereferenceSettings(mode="passthrough", axis="ch"))
    empty = make_empty_msg()
    result = proc(empty)
    check_empty_result(result)


def test_common_rereference_empty_first():
    proc = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


class _NDArraySubclass(np.ndarray):
    """Stand-in for message data that is not a plain ndarray.

    `np.asarray` on a subclass returns a *distinct* base-class object that still
    shares the buffer, so any "did asarray copy?" identity check silently fails
    here -- which is exactly the case an input-mutation bug would hide in.
    """


def _no_mutation_cases():
    """(label, factory, data, dims) for every kernel that could write in place."""
    rng = np.random.default_rng(0)
    block_weights = _make_block_diagonal_weights([64, 64], rng=rng)

    scattered = np.zeros((128, 128))
    for group in ([*range(0, 32), *range(96, 128)], list(range(32, 96))):
        scattered[np.ix_(group, group)] = rng.standard_normal((len(group), len(group)))

    def affine(weights, kernel):
        return lambda: AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel=kernel))

    def car(**kwargs):
        return lambda: CommonRereferenceTransformer(CommonRereferenceSettings(axis="ch", **kwargs))

    wide = rng.standard_normal((30, 128))
    narrow = rng.standard_normal((50, 8))
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    return [
        ("affine blocks", affine(block_weights, "blocks"), wide, ["time", "ch"]),
        ("affine dense", affine(block_weights, "dense"), wide, ["time", "ch"]),
        ("affine blocks channel-major", affine(block_weights, "blocks"), wide.T.copy(), ["ch", "time"]),
        ("affine blocks permuted", affine(scattered, "blocks"), wide, ["time", "ch"]),
        ("car mean global", car(mode="mean"), narrow, ["time", "ch"]),
        ("car mean grouped", car(mode="mean", channel_groups=groups), narrow, ["time", "ch"]),
        (
            "car mean grouped loo",
            car(mode="mean", channel_groups=groups, include_current=False),
            narrow,
            ["time", "ch"],
        ),
        ("car median global", car(mode="median"), narrow, ["time", "ch"]),
        ("car median grouped", car(mode="median", channel_groups=groups), narrow, ["time", "ch"]),
        ("car median partial", car(mode="median", channel_groups=[[0, 1]]), narrow, ["time", "ch"]),
        ("car median float32", car(mode="median", channel_groups=groups), narrow.astype(np.float32), ["time", "ch"]),
    ]


@pytest.mark.parametrize("label,factory,data,dims", _no_mutation_cases(), ids=lambda v: v if isinstance(v, str) else "")
@pytest.mark.parametrize("subclass", [False, True], ids=["ndarray", "subclass"])
def test_does_not_mutate_input(label, factory, data, dims, subclass):
    """Message data may be a view shared with other branches of the graph, so no
    processor may write into it -- not even the paths that fill a buffer in place.

    A read-only array turns any in-place write into a ValueError instead of a
    silent corruption two nodes downstream.
    """
    data = np.ascontiguousarray(data)
    before = data.copy()
    if subclass:
        data = data.view(_NDArraySubclass)
    data.setflags(write=False)

    out = factory()(AxisArray(data, dims=dims))

    assert np.array_equal(np.asarray(data), before)
    assert not np.may_share_memory(out.data, data)


@requires_mlx
def test_affine_block_diagonal_mlx():
    """MLX has no ``matmul(out=)``, so the block kernel concatenates instead of
    filling in place; an empty message must survive both paths."""
    import mlx.core as mx

    rng = np.random.default_rng(42)
    n_ch = 64
    weights = _make_block_diagonal_weights([32, 32], rng=rng)

    def _mlx_msg(n_time: int) -> AxisArray:
        return AxisArray(
            data=mx.array(rng.standard_normal((n_time, n_ch)).astype(np.float32)),
            dims=["time", "ch"],
            axes={
                "time": AxisArray.TimeAxis(fs=100.0),
                "ch": AxisArray.CoordinateAxis(data=np.arange(n_ch).astype(str), dims=["ch"]),
            },
            key="test_affine_block_diagonal_mlx",
        )

    proc = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", kernel="blocks"))

    # Empty startup message arrives first; state initializes from it.
    empty_result = proc(_mlx_msg(0))
    assert proc._state.blocks is not None and len(proc._state.blocks) == 2
    assert not proc._state.fill_in_place
    assert isinstance(empty_result.data, mx.array)
    assert empty_result.data.shape == (0, n_ch)
    assert empty_result.data.dtype == mx.float32
    assert empty_result.dims == ["time", "ch"]
    assert np.array_equal(np.asarray(empty_result.axes["ch"].data), np.arange(n_ch).astype(str))
    check_empty_result(empty_result)

    # Non-empty messages through the same processor still compute correctly.
    normal = _mlx_msg(4)
    out = proc(normal)
    assert isinstance(out.data, mx.array)
    expected = np.asarray(normal.data) @ weights
    assert np.allclose(np.asarray(out.data), expected, atol=1e-4)

    # And another empty message after real data also passes through.
    check_empty_result(proc(_mlx_msg(0)))


@requires_mlx
def test_common_rereference_mlx():
    import mlx.core as mx

    rng = np.random.default_rng(3)
    in_dat = rng.standard_normal((20, 8)).astype(np.float32)
    msg = AxisArray(mx.array(in_dat), dims=["time", "ch"], key="mlx_car")

    proc = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch"))
    out = proc(msg)
    assert isinstance(out.data, mx.array)
    assert np.allclose(np.asarray(out.data), in_dat - in_dat.mean(axis=1, keepdims=True), atol=1e-5)

    groups = [[0, 2, 4, 6], [1, 3, 5, 7]]
    expected = np.zeros_like(in_dat)
    for group in groups:
        block = in_dat[:, group]
        expected[:, group] = block - block.mean(axis=1, keepdims=True)
    proc = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", channel_groups=groups))
    out = proc(msg)
    assert isinstance(out.data, mx.array)
    assert np.allclose(np.asarray(out.data), expected, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_stacked_bias_does_not_mutate_input(dtype):
    """The stacked ``A|B`` path adds the bias in place; it must not touch the message.

    ``_matmul_add`` mutates the buffer ``matmul`` just allocated, which nothing
    else references. If that ever became the caller's array instead, every other
    subscriber to the same message would silently see transformed data.
    """
    rng = np.random.default_rng(0)
    n_t, n_ch = 40, 64
    weights = rng.standard_normal((n_ch + 1, n_ch))
    data = (rng.standard_normal((n_t, n_ch)) * 10).astype(dtype)
    before = data.copy()

    msg = AxisArray(data, dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=1000.0)}, key="a")
    out = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))(msg)

    assert np.array_equal(msg.data, before), "input message was mutated"
    assert not np.shares_memory(out.data, msg.data)

    # And the result still equals the ones-column formulation it replaced.
    augmented = np.concatenate((data, np.ones((n_t, 1), dtype=data.dtype)), axis=-1)
    expected = augmented.astype(np.result_type(dtype, np.float64)) @ weights
    assert np.allclose(np.asarray(out.data), expected, rtol=1e-5, atol=1e-5)


def test_stacked_bias_repeat_processing_is_stable():
    """Feeding the same message twice must give the same answer both times."""
    rng = np.random.default_rng(1)
    n_t, n_ch = 32, 48
    weights = rng.standard_normal((n_ch + 1, n_ch))
    msg = AxisArray(
        rng.standard_normal((n_t, n_ch)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=1000.0)},
        key="a",
    )
    proc = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    first = np.asarray(proc(msg).data).copy()
    second = np.asarray(proc(msg).data)
    assert np.array_equal(first, second)
