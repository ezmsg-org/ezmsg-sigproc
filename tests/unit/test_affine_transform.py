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
from tests.helpers.empty_time import N_CH, check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg
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


def test_common_rereference_clusters():
    n_times = 300
    n_chans = 8
    rng = np.random.default_rng(42)
    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    cluster_a = [0, 1, 2, 3]
    cluster_b = [4, 5, 6, 7]
    clusters = [cluster_a, cluster_b]

    # --- include_current=True ---
    xformer = CommonRereferenceTransformer(
        CommonRereferenceSettings(mode="mean", axis="ch", include_current=True, channel_clusters=clusters)
    )
    msg_out = xformer(msg_in)

    # Expected: per-cluster CAR
    expected = np.zeros_like(in_dat)
    for cluster in clusters:
        cluster_data = in_dat[:, cluster]
        ref = np.mean(cluster_data, axis=1, keepdims=True)
        expected[:, cluster] = cluster_data - ref

    assert np.allclose(msg_out.data, expected)
    assert not np.may_share_memory(msg_out.data, in_dat)

    # --- include_current=False ---
    xformer = CommonRereferenceTransformer(
        CommonRereferenceSettings(mode="mean", axis="ch", include_current=False, channel_clusters=clusters)
    )
    msg_out = xformer(msg_in)

    # Expected: per-cluster CAR excluding current channel (slow deliberate way)
    expected = np.zeros_like(in_dat)
    for cluster in clusters:
        cluster_data = in_dat[:, cluster]
        N = len(cluster)
        for i, ch in enumerate(cluster):
            others = [j for j in range(N) if j != i]
            ref = np.mean(cluster_data[:, others], axis=1)
            expected[:, ch] = cluster_data[:, i] - ref

    assert np.allclose(msg_out.data, expected)


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
    # Square block-diagonal
    weights = _make_block_diagonal_weights([64, 64])
    clusters = _find_block_diagonal_clusters(weights)
    assert clusters is not None
    assert len(clusters) == 2
    in0, out0 = clusters[0]
    in1, out1 = clusters[1]
    assert np.array_equal(in0, np.arange(64))
    assert np.array_equal(out0, np.arange(64))
    assert np.array_equal(in1, np.arange(64, 128))
    assert np.array_equal(out1, np.arange(64, 128))

    # Non-square block-diagonal
    weights_ns = np.zeros((128, 20))
    weights_ns[:64, :10] = 1.0
    weights_ns[64:, 10:] = 1.0
    clusters_ns = _find_block_diagonal_clusters(weights_ns)
    assert clusters_ns is not None
    assert len(clusters_ns) == 2
    assert np.array_equal(clusters_ns[0][0], np.arange(64))
    assert np.array_equal(clusters_ns[0][1], np.arange(10))
    assert np.array_equal(clusters_ns[1][0], np.arange(64, 128))
    assert np.array_equal(clusters_ns[1][1], np.arange(10, 20))

    # Dense matrix should not be detected as block-diagonal
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((64, 64))
    assert _find_block_diagonal_clusters(dense) is None

    # 1x1 should return None
    assert _find_block_diagonal_clusters(np.array([[1.0]])) is None


def test_max_cross_cluster_weight():
    """Test the cross-cluster weight magnitude checker."""
    weights = _make_block_diagonal_weights([64, 64])
    clusters = [(np.arange(64), np.arange(64)), (np.arange(64, 128), np.arange(64, 128))]
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
    assert xformer._state.clusters is not None
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
    assert xformer._state.clusters is not None


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
    assert xformer._state.clusters is not None


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
    assert xformer._state.clusters is not None


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
    assert xformer._state.clusters is not None


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
    assert xformer._state.clusters is None
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
    assert xformer._state.clusters is not None


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
    assert xformer._state.clusters is not None


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
    """Test that out-of-range channel indices raise ValueError."""
    n_chans = 64
    weights = np.eye(n_chans)

    with pytest.raises(ValueError, match="out-of-range input indices"):
        xformer = AffineTransformTransformer(
            AffineTransformSettings(
                weights=weights,
                axis="ch",
                channel_clusters=[[0, 1, 2], [64]],  # 64 is out of range for 64-channel matrix
            )
        )
        msg_in = AxisArray(data=np.zeros((10, n_chans)), dims=["time", "ch"])
        xformer(msg_in)

    with pytest.raises(ValueError, match="out-of-range input indices"):
        xformer = AffineTransformTransformer(
            AffineTransformSettings(
                weights=weights,
                axis="ch",
                channel_clusters=[[-1, 0, 1]],
            )
        )
        msg_in = AxisArray(data=np.zeros((10, n_chans)), dims=["time", "ch"])
        xformer(msg_in)


def test_block_diagonal_omitted_zero_channels():
    """Test that channels with all-zero weights can be omitted from channel_clusters."""
    n_times = 30
    n_chans = 6
    rng = np.random.default_rng(42)

    # Channels 0,1 form cluster A; channels 4,5 form cluster B;
    # channels 2,3 have all-zero rows and columns.
    weights = np.zeros((n_chans, n_chans))
    weights[:2, :2] = rng.standard_normal((2, 2))
    weights[4:, 4:] = rng.standard_normal((2, 2))

    in_dat = rng.standard_normal((n_times, n_chans))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights  # channels 2,3 output should be zero

    xformer = AffineTransformTransformer(
        AffineTransformSettings(
            weights=weights,
            axis="ch",
            channel_clusters=[[0, 1], [4, 5]],  # channels 2,3 omitted
            min_cluster_size=1,
        )
    )
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == expected.shape
    assert np.allclose(msg_out.data, expected)
    # Verify omitted channels are indeed zero
    assert np.all(msg_out.data[:, 2] == 0)
    assert np.all(msg_out.data[:, 3] == 0)


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
    clusters = [(np.arange(32), np.arange(32)), (np.arange(32, 64), np.arange(32, 64))]
    result = _merge_small_clusters(clusters, min_size=32)
    assert len(result) == 2

    # Many tiny clusters get merged greedily
    clusters = [(np.array([i]), np.array([i])) for i in range(64)]  # 64 clusters of size 1
    result = _merge_small_clusters(clusters, min_size=32)
    # 64 channels / 32 min_size = 2 merged groups
    assert len(result) == 2
    assert all(len(c[0]) == 32 for c in result)

    # Mix of large and small
    clusters = [
        (np.arange(64), np.arange(64)),
        (np.array([64]), np.array([64])),
        (np.array([65]), np.array([65])),
        (np.array([66]), np.array([66])),
    ]
    result = _merge_small_clusters(clusters, min_size=32)
    # 1 large (64) + 1 merged remainder (3 channels, below threshold but grouped)
    assert len(result) == 2
    assert 64 in [len(c[0]) for c in result]
    assert 3 in [len(c[0]) for c in result]

    # min_size=1 disables merging
    clusters = [(np.array([i]), np.array([i])) for i in range(10)]
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
    assert xformer._state.clusters is not None
    # Should have far fewer than 33 clusters after merging
    assert len(xformer._state.clusters) <= 3


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
    assert xformer._state.clusters is None


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
    assert xformer._state.clusters is not None
    assert len(xformer._state.clusters) == 8


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


def test_nonsquare_auto_detect():
    """Test auto-detection on a non-square block-diagonal matrix."""
    n_times = 30
    rng = np.random.default_rng(42)

    # 4 blocks of 64 input → 10 output
    block_shapes = [(64, 10)] * 4
    weights = _make_block_diagonal_weights_nonsquare(block_shapes, rng=rng)
    assert weights.shape == (256, 40)

    in_dat = rng.standard_normal((n_times, 256))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", min_cluster_size=1))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == (n_times, 40)
    assert np.allclose(msg_out.data, expected)
    assert xformer._state.clusters is not None
    assert len(xformer._state.clusters) == 4


def test_nonsquare_explicit_clusters():
    """Test explicit input-only channel_clusters for non-square matrices."""
    n_times = 30
    rng = np.random.default_rng(42)

    block_shapes = [(64, 10)] * 4
    weights = _make_block_diagonal_weights_nonsquare(block_shapes, rng=rng)

    in_dat = rng.standard_normal((n_times, 256))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    # Only specify input clusters; output indices derived from weight matrix
    xformer = AffineTransformTransformer(
        AffineTransformSettings(
            weights=weights,
            axis="ch",
            channel_clusters=[
                list(range(0, 64)),
                list(range(64, 128)),
                list(range(128, 192)),
                list(range(192, 256)),
            ],
            min_cluster_size=1,
        )
    )
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == (n_times, 40)
    assert np.allclose(msg_out.data, expected)


def test_nonsquare_unequal_blocks():
    """Test non-square with blocks of different shapes."""
    n_times = 30
    rng = np.random.default_rng(42)

    block_shapes = [(64, 10), (96, 20), (32, 5)]
    weights = _make_block_diagonal_weights_nonsquare(block_shapes, rng=rng)
    assert weights.shape == (192, 35)

    in_dat = rng.standard_normal((n_times, 192))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", min_cluster_size=1))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == (n_times, 35)
    assert np.allclose(msg_out.data, expected)
    assert xformer._state.clusters is not None


def test_nonsquare_shuffled():
    """Test non-square block-diagonal with shuffled input/output channels."""
    n_times = 30
    rng = np.random.default_rng(42)

    # Sorted block-diagonal
    block_shapes = [(64, 10)] * 2
    sorted_weights = _make_block_diagonal_weights_nonsquare(block_shapes, rng=rng)

    # Shuffle input channels
    in_perm = np.arange(128)
    rng.shuffle(in_perm)
    # Shuffle output channels
    out_perm = np.arange(20)
    rng.shuffle(out_perm)

    weights = sorted_weights[np.ix_(in_perm, out_perm)]

    in_dat = rng.standard_normal((n_times, 128))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch"])

    expected = in_dat @ weights

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", min_cluster_size=1))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == (n_times, 20)
    assert np.allclose(msg_out.data, expected)
    assert xformer._state.clusters is not None


def test_nonsquare_non_last_axis():
    """Test non-square block-diagonal with the target axis not being the last axis."""
    n_times = 30
    n_features = 5
    rng = np.random.default_rng(42)

    block_shapes = [(64, 10)] * 2
    weights = _make_block_diagonal_weights_nonsquare(block_shapes, rng=rng)

    # Data shape: (n_times, n_in, n_features) — ch is the middle axis
    in_dat = rng.standard_normal((n_times, 128, n_features))
    msg_in = AxisArray(data=in_dat, dims=["time", "ch", "feat"])

    # Expected: move ch to last, matmul, move back
    data_perm = np.transpose(in_dat, (0, 2, 1))  # (time, feat, 128)
    expected_perm = data_perm @ weights  # (time, feat, 20)
    expected = np.transpose(expected_perm, (0, 2, 1))  # (time, 20, feat)

    xformer = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch", min_cluster_size=1))
    msg_out = xformer(msg_in)

    assert msg_out.data.shape == expected.shape
    assert np.allclose(msg_out.data, expected)


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


def test_affine_empty_square():
    from ezmsg.sigproc.affinetransform import AffineTransformSettings, AffineTransformTransformer

    weights = np.eye(N_CH)
    proc = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_affine_empty_nonsquare():
    from ezmsg.sigproc.affinetransform import AffineTransformSettings, AffineTransformTransformer

    weights = np.random.randn(N_CH, 2)
    proc = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)


def test_affine_empty_passthrough():
    from ezmsg.sigproc.affinetransform import AffineTransformSettings, AffineTransformTransformer

    proc = AffineTransformTransformer(AffineTransformSettings(weights="passthrough", axis="ch"))
    empty = make_empty_msg()
    result = proc(empty)
    check_empty_result(result)


def test_affine_empty_first():
    from ezmsg.sigproc.affinetransform import AffineTransformSettings, AffineTransformTransformer

    weights = np.eye(N_CH)
    proc = AffineTransformTransformer(AffineTransformSettings(weights=weights, axis="ch"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_common_rereference_empty_mean():
    from ezmsg.sigproc.affinetransform import CommonRereferenceSettings, CommonRereferenceTransformer

    proc = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_common_rereference_empty_passthrough():
    from ezmsg.sigproc.affinetransform import CommonRereferenceSettings, CommonRereferenceTransformer

    proc = CommonRereferenceTransformer(CommonRereferenceSettings(mode="passthrough", axis="ch"))
    empty = make_empty_msg()
    result = proc(empty)
    check_empty_result(result)


def test_common_rereference_empty_first():
    from ezmsg.sigproc.affinetransform import CommonRereferenceSettings, CommonRereferenceTransformer

    proc = CommonRereferenceTransformer(CommonRereferenceSettings(mode="mean", axis="ch"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)
