import numpy as np
import pytest

from ezmsg.sigproc.util.blockdiag import (
    BlockPlan,
    contiguous_block_partition,
    plan_block_matmul,
)


def _block_diag(sizes: list[int], seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    weights = np.zeros((n, n))
    offset = 0
    for size in sizes:
        weights[offset : offset + size, offset : offset + size] = rng.standard_normal((size, size))
        offset += size
    return weights


def _shapes(blocks) -> list[tuple[int, int]]:
    return [(r.stop - r.start, c.stop - c.start) for r, c in blocks]


def _tiles(blocks, n_in: int, n_out: int) -> bool:
    """Blocks cover both axes exactly once, with no gaps -- the property that
    lets the output buffer be allocated uninitialized."""
    rows = [(r.start, r.stop) for r, _ in blocks]
    cols = [(c.start, c.stop) for _, c in blocks]
    return (
        rows[0][0] == 0
        and cols[0][0] == 0
        and rows[-1][1] == n_in
        and cols[-1][1] == n_out
        and all(a[1] == b[0] for a, b in zip(rows, rows[1:]))
        and all(a[1] == b[0] for a, b in zip(cols, cols[1:]))
    )


class TestContiguousBlockPartition:
    def test_square_blocks(self):
        assert _shapes(contiguous_block_partition(_block_diag([64, 64]))) == [(64, 64), (64, 64)]

    def test_unequal_blocks(self):
        assert _shapes(contiguous_block_partition(_block_diag([32, 64, 96]))) == [(32, 32), (64, 64), (96, 96)]

    def test_finest_partition(self):
        """Identity is n blocks of 1, not one block of n."""
        assert len(contiguous_block_partition(np.eye(16))) == 16

    def test_dense_is_one_block(self):
        dense = np.random.default_rng(0).standard_normal((64, 64))
        assert _shapes(contiguous_block_partition(dense)) == [(64, 64)]

    def test_non_contiguous_is_one_block(self):
        """Interleaved groups have no contiguous tiling; the caller falls back."""
        weights = np.zeros((8, 8))
        for group in ([0, 2, 4, 6], [1, 3, 5, 7]):
            weights[np.ix_(group, group)] = 1.0
        assert _shapes(contiguous_block_partition(weights)) == [(8, 8)]

    def test_non_square(self):
        weights = np.zeros((128, 20))
        weights[:64, :10] = 1.0
        weights[64:, 10:] = 1.0
        blocks = contiguous_block_partition(weights)
        assert _shapes(blocks) == [(64, 10), (64, 10)]
        assert _tiles(blocks, 128, 20)

    def test_zero_rows_and_cols_are_absorbed(self):
        """All-zero channels join a neighbour rather than leaving a gap."""
        weights = np.zeros((6, 6))
        weights[:2, :2] = 1.0
        weights[4:, 4:] = 1.0
        blocks = contiguous_block_partition(weights)
        assert _tiles(blocks, 6, 6)
        assert len(blocks) == 2

    def test_all_zero_matrix(self):
        assert _shapes(contiguous_block_partition(np.zeros((8, 8)))) == [(8, 8)]

    def test_empty_matrix(self):
        assert contiguous_block_partition(np.zeros((0, 4))) == [(slice(0, 0), slice(0, 4))]

    def test_swapped_blocks_are_not_contiguous(self):
        """Rows 0-3 feeding cols 4-7 (and vice versa) has no contiguous tiling."""
        weights = np.zeros((8, 8))
        weights[:4, 4:] = 1.0
        weights[4:, :4] = 1.0
        assert _shapes(contiguous_block_partition(weights)) == [(8, 8)]

    @pytest.mark.parametrize("sizes", [[64, 64], [32, 64, 96], [4] * 8, [1] * 16, [7, 1, 12]])
    def test_partition_covers_every_nonzero(self, sizes):
        weights = _block_diag(sizes)
        blocks = contiguous_block_partition(weights)
        assert _tiles(blocks, *weights.shape)
        covered = np.zeros(weights.shape, dtype=bool)
        for rows, cols in blocks:
            covered[rows, cols] = True
        assert not np.any(weights[~covered])


class TestPlanBlockMatmul:
    def test_dense_weights_get_no_plan(self):
        dense = np.random.default_rng(0).standard_normal((512, 512))
        assert plan_block_matmul(dense, 30) is None

    def test_small_matrix_prefers_dense(self):
        """Below a few hundred channels a dense matmul beats the block loop even
        though the block loop does fewer FLOPs (ezmsg-org/ezmsg-sigproc#210)."""
        assert plan_block_matmul(_block_diag([16] * 8), 30) is None

    def test_large_matrix_uses_blocks(self):
        plan = plan_block_matmul(_block_diag([64] * 16), 30)
        assert plan is not None and plan.n_blocks > 1
        assert plan.in_perm is None and plan.out_perm is None

    def test_longer_chunks_favor_finer_blocks(self):
        """More samples amortize the per-block call overhead."""
        weights = _block_diag([64] * 4)
        short = plan_block_matmul(weights, 30)
        long = plan_block_matmul(weights, 30000)
        assert long is not None
        assert short is None or short.n_blocks <= long.n_blocks

    def test_many_tiny_blocks_collapse(self):
        """256 single-channel blocks would cost more in call overhead than they
        save, so they merge (or fall through to dense)."""
        plan = plan_block_matmul(np.eye(256), 30)
        assert plan is None or plan.n_blocks < 256

    def test_force_uses_finest_structure(self):
        plan = plan_block_matmul(_block_diag([16] * 8), 30, force=True)
        assert plan is not None and plan.n_blocks == 8

    def test_force_finds_permuted_structure(self):
        n = 128
        groups = [list(range(0, 32)) + list(range(96, 128)), list(range(32, 96))]
        weights = np.zeros((n, n))
        rng = np.random.default_rng(0)
        for group in groups:
            weights[np.ix_(group, group)] = rng.standard_normal((len(group), len(group)))
        plan = plan_block_matmul(weights, 30, force=True)
        assert plan is not None and plan.n_blocks == 2
        assert plan.in_perm is not None
        permuted = weights[plan.in_perm][:, plan.out_perm]
        covered = np.zeros(permuted.shape, dtype=bool)
        for rows, cols in plan.blocks:
            covered[rows, cols] = True
        assert not np.any(permuted[~covered])

    def test_dispatched_backends_tolerate_fewer_blocks(self):
        """Heavier per-op dispatch shifts the crossover toward dense."""
        weights = _block_diag([64] * 8)
        eager = plan_block_matmul(weights, 30, dispatched=False)
        dispatched = plan_block_matmul(weights, 30, dispatched=True)
        n_eager = 0 if eager is None else eager.n_blocks
        n_dispatched = 0 if dispatched is None else dispatched.n_blocks
        assert n_dispatched <= n_eager

    def test_plan_tiles_the_matrix(self):
        plan = plan_block_matmul(_block_diag([64] * 16), 30)
        assert isinstance(plan, BlockPlan)
        assert _tiles(plan.blocks, 1024, 1024)

    def test_non_2d_returns_none(self):
        assert plan_block_matmul(np.zeros((4, 4, 4)), 30) is None
