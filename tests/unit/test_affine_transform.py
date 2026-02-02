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
    _find_block_diagonal_clusters,
    _max_cross_cluster_weight,
    _merge_small_clusters,
)
from tests.helpers.util import assert_messages_equal


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


def test_common_rereference():
    n_times = 300
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    backup = [copy.deepcopy(msg_in)]

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch", include_current=True))
    msg_out = xformer(msg_in)
    assert np.array_equal(
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
    msg_out = xformer(msg_in)  # 41 us
    assert np.allclose(msg_out.data, expected_out)

    # Instead of CAR, we could use AffineTransformTransformer with weights that reproduce CAR.
    # However, this method is 30x slower than above. (Actual difference varies depending on data shape).
    if False:
        weights = -np.ones((n_chans, n_chans)) / (n_chans - 1)
        np.fill_diagonal(weights, 1)
        xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
        msg_out = xformer(msg_in)
        assert np.allclose(msg_out.data, expected_out)


def test_car_passthrough():
    n_times = 300
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    xformer = CommonRereferenceTransformer(CommonRereferenceSettings(mode="passthrough"))
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, in_dat)
    assert np.may_share_memory(msg_out.data, in_dat)


# --- Block-diagonal optimization tests ---


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


def test_find_block_diagonal_clusters():
    """Test the cluster detection helper function directly."""
    weights = _make_block_diagonal_weights([64, 64])
    clusters = _find_block_diagonal_clusters(weights)
    assert clusters is not None
    assert len(clusters) == 2
    assert np.array_equal(clusters[0], np.arange(64))
    assert np.array_equal(clusters[1], np.arange(64, 128))

    # Dense matrix should not be detected as block-diagonal
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((64, 64))
    assert _find_block_diagonal_clusters(dense) is None

    # Non-square should return None
    assert _find_block_diagonal_clusters(rng.standard_normal((64, 32))) is None

    # 1x1 should return None
    assert _find_block_diagonal_clusters(np.array([[1.0]])) is None


def test_max_cross_cluster_weight():
    """Test the cross-cluster weight magnitude checker."""
    weights = _make_block_diagonal_weights([64, 64])
    clusters = [np.arange(64), np.arange(64, 128)]
    assert _max_cross_cluster_weight(weights, clusters) == 0.0

    # Add a small cross-cluster weight
    weights[0, 64] = 0.001
    assert abs(_max_cross_cluster_weight(weights, clusters) - 0.001) < 1e-10


def test_block_diagonal_auto_detect():
    """Test that block-diagonal weight matrices are auto-detected and give correct results."""
    n_times = 30
    n_chans = 128
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights([64, 64], rng=rng)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == expected.shape
    assert np.allclose(msg_out.data, expected)
    # Verify cluster optimization was actually used
    assert xformer._state.cluster_perm is not None
    assert xformer._state.weights is None


def test_block_diagonal_explicit_clusters():
    """Test explicit channel_clusters setting."""
    n_times = 30
    n_chans = 128
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights([64, 64], rng=rng)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    xformer = AffineTransformTransformer(
        AffineTransformSettings(
            weights=weights,
            axis="ch",
            channel_clusters=[list(range(64)), list(range(64, 128))],
        )
    )
    msg_out = xformer(msg_in)

    assert np.allclose(msg_out.data, expected)
    assert xformer._state.cluster_perm is not None


def test_block_diagonal_unsorted_channels():
    """Test with channels interleaved across clusters (not sorted by block)."""
    n_times = 30
    n_chans = 128
    rng = np.random.default_rng(42)

    # Create block-diagonal weights in sorted order
    sorted_weights = _make_block_diagonal_weights([64, 64], rng=rng)

    # Permute channels to interleave clusters
    perm = np.arange(n_chans)
    rng.shuffle(perm)

    # Permuted weights: W_perm[i,j] = W_sorted[perm[i], perm[j]]
    weights = sorted_weights[np.ix_(perm, perm)]

    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == expected.shape
    assert np.allclose(msg_out.data, expected)
    assert xformer._state.cluster_perm is not None


def test_block_diagonal_many_clusters():
    """Test with many small clusters (8 clusters of 32 channels)."""
    n_times = 30
    n_clusters = 8
    cluster_size = 32
    n_chans = n_clusters * cluster_size
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights([cluster_size] * n_clusters, rng=rng)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    msg_out = xformer(msg_in)

    assert np.allclose(msg_out.data, expected)
    assert xformer._state.cluster_perm is not None


def test_block_diagonal_unequal_cluster_sizes():
    """Test with clusters of different sizes."""
    n_times = 30
    block_sizes = [32, 64, 96]
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights(block_sizes, rng=rng)
    n_chans = sum(block_sizes)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    msg_out = xformer(msg_in)

    assert np.allclose(msg_out.data, expected)
    assert xformer._state.cluster_perm is not None


def test_block_diagonal_not_triggered_for_dense():
    """Test that a fully-connected weight matrix falls back to standard matmul."""
    n_chans = 64
    rng = np.random.default_rng(42)
    weights = rng.standard_normal((n_chans, n_chans))

    in_dat = rng.standard_normal((10, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    msg_out = xformer(msg_in)

    assert np.allclose(msg_out.data, expected)
    # Should NOT use cluster optimization
    assert xformer._state.cluster_perm is None
    assert xformer._state.weights is not None


def test_block_diagonal_non_last_axis():
    """Test block-diagonal with the target axis not being the last axis."""
    n_times = 30
    n_chans = 128
    n_features = 5
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights([64, 64], rng=rng)

    # Data shape: (n_times, n_chans, n_features) -- ch is the middle axis
    in_dat = rng.standard_normal((n_times, n_chans, n_features))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch", "feat"])

    # Expected: move ch to last, matmul, move back
    data_perm = np.transpose(in_dat, (0, 2, 1))  # (time, feat, ch)
    expected_perm = data_perm @ weights  # (time, feat, ch)
    expected = np.transpose(expected_perm, (0, 2, 1))  # (time, ch, feat)

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == expected.shape
    assert np.allclose(msg_out.data, expected)
    assert xformer._state.cluster_perm is not None


def test_block_diagonal_right_multiply_false():
    """Test block-diagonal with right_multiply=False."""
    n_times = 30
    n_chans = 128
    rng = np.random.default_rng(42)

    # Weights will be transposed internally when right_multiply=False
    raw_weights = _make_block_diagonal_weights([64, 64], rng=rng)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ raw_weights.T

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=raw_weights, axis="ch", right_multiply=False))
    msg_out = xformer(msg_in)

    assert np.allclose(msg_out.data, expected)
    assert xformer._state.cluster_perm is not None


def test_block_diagonal_identity_preserves_data():
    """Test that block-diagonal identity matrices act as identity."""
    n_times = 20
    n_chans = 128
    rng = np.random.default_rng(42)

    # Identity matrix is block-diagonal (each channel is its own cluster)
    weights = np.eye(n_chans)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    msg_out = xformer(msg_in)

    assert np.allclose(msg_out.data, in_dat)


def test_block_diagonal_invalid_clusters():
    """Test that channel_clusters missing channels raises ValueError."""
    n_chans = 128
    weights = np.eye(n_chans)

    # Only covers half the channels
    with pytest.raises(ValueError, match="channel_clusters must cover all channel indices"):
        xformer = AffineTransformTransformer(
            AffineTransformSettings(
                weights=weights,
                axis="ch",
                channel_clusters=[list(range(64))],
            )
        )
        msg_in = AxisArray(data=np.zeros((10, n_chans)), dims=["time", "ch"])
        xformer(msg_in)

    # Covers 0..63 twice but never 64..127
    with pytest.raises(ValueError, match="channel_clusters must cover all channel indices"):
        xformer = AffineTransformTransformer(
            AffineTransformSettings(
                weights=weights,
                axis="ch",
                channel_clusters=[list(range(64)), list(range(64))],
            )
        )
        msg_in = AxisArray(data=np.zeros((10, n_chans)), dims=["time", "ch"])
        xformer(msg_in)


def test_block_diagonal_streaming():
    """Test that block-diagonal works across multiple messages (streaming)."""
    n_chans = 128
    rng = np.random.default_rng(42)

    weights = _make_block_diagonal_weights([64, 64], rng=rng)

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))

    for _ in range(5):
        in_dat = rng.standard_normal((30, n_chans))
        msg_in = AxisArray(data=in_dat, dims=["time", "ch"])
        expected = in_dat @ weights
        msg_out = xformer(msg_in)
        assert np.allclose(msg_out.data, expected)


# --- Cluster merging tests ---


def test_merge_small_clusters_unit():
    """Test _merge_small_clusters helper directly."""
    # All clusters already large enough — no change
    clusters = [np.arange(32), np.arange(32, 64)]
    result = _merge_small_clusters(clusters, min_size=32)
    assert len(result) == 2

    # Many tiny clusters get merged greedily
    clusters = [np.array([i]) for i in range(64)]  # 64 clusters of size 1
    result = _merge_small_clusters(clusters, min_size=32)
    # 64 channels / 32 min_size = 2 merged groups
    assert len(result) == 2
    assert all(len(c) == 32 for c in result)

    # Mix of large and small
    clusters = [np.arange(64), np.array([64]), np.array([65]), np.array([66])]
    result = _merge_small_clusters(clusters, min_size=32)
    # 1 large (64) + 1 merged remainder (3 channels, below threshold but grouped)
    assert len(result) == 2
    assert 64 in [len(c) for c in result]
    assert 3 in [len(c) for c in result]

    # min_size=1 disables merging
    clusters = [np.array([i]) for i in range(10)]
    result = _merge_small_clusters(clusters, min_size=1)
    assert len(result) == 10


def test_merge_small_clusters_correctness():
    """Test that merging small clusters still produces correct matmul results."""
    n_times = 30
    rng = np.random.default_rng(42)

    # 1 large cluster of 64 + 32 tiny clusters of 2 = 128 channels total
    block_sizes = [64] + [2] * 32
    weights = _make_block_diagonal_weights(block_sizes, rng=rng)
    n_chans = sum(block_sizes)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    # With default min_cluster_size=32, the 32 tiny clusters should be merged
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    msg_out = xformer(msg_in)

    assert np.allclose(msg_out.data, expected)
    assert xformer._state.cluster_perm is not None
    # Should have far fewer than 33 clusters after merging
    assert len(xformer._state.cluster_weights) <= 3


def test_merge_collapses_to_dense():
    """Test that if all clusters are tiny, merging collapses to 1 group → falls back to dense."""
    n_times = 30
    n_chans = 16
    rng = np.random.default_rng(42)

    # 16 clusters of 1 channel each (identity matrix)
    weights = np.eye(n_chans)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    # min_cluster_size=32 > 16 total channels, so everything merges into 1 group → dense fallback
    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", min_cluster_size=32))
    msg_out = xformer(msg_in)

    assert np.allclose(msg_out.data, in_dat)
    # Merged into 1 cluster → should fall back to dense (no cluster optimization)
    assert xformer._state.cluster_perm is None


def test_min_cluster_size_1_disables_merging():
    """Test that min_cluster_size=1 keeps all detected clusters separate."""
    n_times = 30
    rng = np.random.default_rng(42)

    # 8 clusters of 4 channels each
    block_sizes = [4] * 8
    weights = _make_block_diagonal_weights(block_sizes, rng=rng)
    n_chans = sum(block_sizes)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", min_cluster_size=1))
    msg_out = xformer(msg_in)

    assert np.allclose(msg_out.data, expected)
    assert xformer._state.cluster_perm is not None
    assert len(xformer._state.cluster_weights) == 8
