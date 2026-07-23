import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.affinetransform import AffineTransformSettings, AffineTransformTransformer
from ezmsg.sigproc.util.rereference import RereferenceKind, car_matrix, rereference_matrix


def _rng():
    return np.random.default_rng(2718)


class TestCarMatrix:
    def test_single_cluster_car(self):
        A = car_matrix(6)
        X = _rng().normal(size=(20, 6))
        expected = X - X.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(X @ A, expected, atol=1e-12)

    def test_single_cluster_leave_one_out(self):
        n = 6
        A = car_matrix(n, include_current=False)
        X = _rng().normal(size=(20, n))
        loo = (X.sum(axis=1, keepdims=True) - X) / (n - 1)
        np.testing.assert_allclose(X @ A, X - loo, atol=1e-12)

    def test_per_cluster_independence(self):
        clusters = [[0, 1, 2], [3, 4, 5, 6, 7]]
        A = car_matrix(8, clusters=clusters, include_current=False)
        X = _rng().normal(size=(20, 8))
        expected = X.copy()
        for cl in clusters:
            block = X[:, cl]
            loo = (block.sum(axis=1, keepdims=True) - block) / (len(cl) - 1)
            expected[:, cl] = block - loo
        np.testing.assert_allclose(X @ A, expected, atol=1e-12)
        # No cross-cluster terms
        assert np.all(A[np.ix_(clusters[0], clusters[1])] == 0)
        assert np.all(A[np.ix_(clusters[1], clusters[0])] == 0)

    def test_min_reref_size_leaves_small_clusters_identity(self):
        clusters = [[0, 1], [2, 3, 4, 5]]
        A = car_matrix(6, clusters=clusters, include_current=False, min_reref_size=3)
        np.testing.assert_array_equal(A[np.ix_([0, 1], [0, 1])], np.eye(2))
        assert A[2, 3] != 0

    def test_unclustered_channels_stay_identity(self):
        A = car_matrix(5, clusters=[[0, 1, 2]])
        np.testing.assert_array_equal(A[3:, :], np.eye(5)[3:, :])
        np.testing.assert_array_equal(A[:, 3:], np.eye(5)[:, 3:])

    def test_leave_one_out_singleton_cluster_is_identity(self):
        # k == 1 has no "other" channels; must not divide by zero.
        A = car_matrix(4, clusters=[[0], [1, 2, 3]], include_current=False)
        np.testing.assert_array_equal(A[0, :], np.eye(4)[0, :])

    def test_leave_one_out_pair_is_bipolar(self):
        A = car_matrix(2, include_current=False)
        X = _rng().normal(size=(10, 2))
        np.testing.assert_allclose((X @ A)[:, 0], X[:, 0] - X[:, 1], atol=1e-12)

    def test_symmetric(self):
        A = car_matrix(8, clusters=[[0, 1, 2], [3, 4, 5, 6, 7]], include_current=False)
        np.testing.assert_array_equal(A, A.T)

    def test_zero_channels(self):
        assert car_matrix(0).shape == (0, 0)
        assert car_matrix(0, clusters=[]).shape == (0, 0)

    def test_out_of_range_indices_raise(self):
        with pytest.raises(ValueError, match="out-of-range"):
            car_matrix(4, clusters=[[0, 1, 4]])
        with pytest.raises(ValueError, match="out-of-range"):
            car_matrix(4, clusters=[[-1, 0, 1]])

    def test_dtype(self):
        assert car_matrix(4, dtype=np.float32).dtype == np.float32


class TestRereferenceMatrix:
    def test_identity(self):
        np.testing.assert_array_equal(rereference_matrix(RereferenceKind.IDENTITY, 5), np.eye(5))

    def test_car_dispatch(self):
        expected = car_matrix(6, clusters=[[0, 1, 2]], include_current=False)
        got = rereference_matrix(RereferenceKind.CAR, 6, clusters=[[0, 1, 2]], include_current=False)
        np.testing.assert_array_equal(got, expected)

    def test_accepts_plain_strings(self):
        np.testing.assert_array_equal(rereference_matrix("identity", 3), np.eye(3))
        np.testing.assert_array_equal(rereference_matrix("car", 4), car_matrix(4))

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            rereference_matrix("laplacian", 4)


class TestAffineKindWeights:
    """AffineTransformTransformer accepts a RereferenceKind (or its string value)
    as ``weights`` and builds the matrix over ``channel_clusters`` at reset."""

    def test_identity_kind(self):
        X = _rng().normal(size=(10, 6))
        xf = AffineTransformTransformer(AffineTransformSettings(weights=RereferenceKind.IDENTITY, axis="ch"))
        out = xf(AxisArray(X, dims=["time", "ch"]))
        np.testing.assert_allclose(out.data, X, atol=1e-12)

    def test_car_kind_with_clusters(self):
        clusters = [[0, 1, 2], [3, 4, 5]]
        X = _rng().normal(size=(10, 6))
        xf = AffineTransformTransformer(
            AffineTransformSettings(weights=RereferenceKind.CAR, axis="ch", channel_clusters=clusters)
        )
        out = xf(AxisArray(X, dims=["time", "ch"]))
        expected = X.copy()
        for cl in clusters:
            expected[:, cl] -= X[:, cl].mean(axis=1, keepdims=True)
        np.testing.assert_allclose(out.data, expected, atol=1e-12)

    def test_car_kind_as_plain_string(self):
        X = _rng().normal(size=(10, 4))
        xf = AffineTransformTransformer(AffineTransformSettings(weights="car", axis="ch"))
        out = xf(AxisArray(X, dims=["time", "ch"]))
        np.testing.assert_allclose(out.data, X - X.mean(axis=1, keepdims=True), atol=1e-12)

    def test_kind_weights_rebuild_on_channel_count_change(self):
        xf = AffineTransformTransformer(AffineTransformSettings(weights="car", axis="ch"))
        for n_ch in (4, 8):
            X = _rng().normal(size=(5, n_ch))
            out = xf(AxisArray(X, dims=["time", "ch"], key="k"))
            np.testing.assert_allclose(out.data, X - X.mean(axis=1, keepdims=True), atol=1e-12)
