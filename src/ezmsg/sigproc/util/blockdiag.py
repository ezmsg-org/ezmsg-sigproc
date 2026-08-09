"""Find block-diagonal structure in a weight matrix, then decide whether to use it.

For ``y = x @ W`` with ``W`` of shape ``(n_in, n_out)``: when ``W`` is
block-diagonal the product decomposes into independent per-block matmuls that
touch only the weights inside the blocks. That is *fewer* FLOPs, but it is not
automatically *faster* — each block costs a separate kernel launch, and below a
few hundred channels a dense matmul against an L2-resident weight matrix wins
outright (see ezmsg-org/ezmsg-sigproc#210).

So this module separates two questions:

1. **What structure does W have?** :func:`contiguous_block_partition` answers it
   with numpy alone, in one pass over the nonzero mask. Contiguous blocks are
   the case worth optimizing: they slice into views, so there is no gather, no
   scatter, and the output can be filled in place.
2. **Is that structure worth exploiting?** :func:`plan_block_matmul` answers it
   with the cost model below, and returns ``None`` for "just do a dense matmul".

Structure is always read off ``W`` itself, never taken on a caller's word — a
hint that disagrees with the weights used to silently compute the wrong answer
(ezmsg-org/ezmsg-sigproc#198).

Cost model
----------
Runtime of one matmul formulation is modelled as::

    (n_samples + WEIGHT_LOAD_SAMPLES) * (weight elements touched)
        + (number of kernel calls) * CALL_COST_MACS
        + (elements gathered) * GATHER_COST_MACS

in units of multiply-accumulates. The first term charges both the arithmetic
(``n_samples`` MACs per weight) and the one-off cost of streaming the weights
into cache (worth about ``WEIGHT_LOAD_SAMPLES`` samples of arithmetic at
typical FLOPs-per-byte ratios). The constants are ratios, so only their
relative size matters; they were fit to ``benchmarks/benchmark_affine_kernels.py``
on an Apple M-series CPU with float32 data, and the decisions they drive are
insensitive to moderate error — the model only has to get the *ordering* right,
and the formulations are within ~2x of each other near every crossover.

Known limitation: the model counts weights touched, not access patterns, so it
does not see that a many-block loop re-walks a strided view of a chunk too large
to cache. Above ~2000 channels with several thousand samples per chunk it picks
a finer blocking than optimal, costing up to ~40% against the best merge (still
several times faster than dense). Pass ``kernel="dense"`` if that combination is
your hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WEIGHT_LOAD_SAMPLES = 108
"""Streaming the weight matrix once costs about this many samples of arithmetic."""

CALL_COST_MACS = 1.6e6
"""Cost of one extra matmul call, in MACs, for eager CPU backends (numpy)."""

CALL_COST_MACS_DISPATCHED = 8.0e6
"""Ditto for backends with heavier per-op dispatch (MLX, torch, cupy)."""

GATHER_COST_MACS = 1.6e3
"""Cost of moving one element through a fancy-index gather, in MACs.

Large on purpose: fancy indexing runs at a small fraction of streaming
bandwidth, which is why permuting channels to make blocks contiguous only pays
off for wide, short chunks.
"""

PERMUTED_SEARCH_MIN_WEIGHTS = 1 << 20
"""Skip the connected-components search (and its scipy import) for matrices
smaller than this. Below ~1024x1024 a dense matmul beats a permuted block
matmul at every chunk length measured, so the search could only cost time."""


@dataclass(frozen=True)
class BlockPlan:
    """How to evaluate ``x @ W`` as a sequence of per-block matmuls.

    Each entry of :attr:`blocks` pairs a contiguous input slice with the
    contiguous output slice it writes. The slices tile ``0..n_in`` and
    ``0..n_out`` exactly, so the output buffer needs no zero-fill.

    When the blocks are only contiguous *after* reordering channels, the
    permutations say how: ``weights[in_perm][:, out_perm]`` is what the blocks
    tile. At runtime that means gathering the input by ``in_perm`` and undoing
    ``out_perm`` on the result.
    """

    blocks: tuple[tuple[slice, slice], ...]
    in_perm: np.ndarray | None = None
    out_perm: np.ndarray | None = None

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def contiguous_block_partition(weights: np.ndarray) -> list[tuple[slice, slice]]:
    """Split ``weights`` into the finest tiling of contiguous diagonal blocks.

    Every nonzero of *weights* lies inside one returned ``(rows, cols)`` block,
    and the blocks tile the full row and column ranges. A single block spanning
    everything means "no exploitable contiguous structure".

    All-zero rows and columns are absorbed into a neighbouring block rather than
    dropped, which keeps the tiling gap-free: an omitted output column would
    otherwise have to be zero-filled separately.
    """
    n_in, n_out = weights.shape
    whole = [(slice(0, n_in), slice(0, n_out))]
    if n_in == 0 or n_out == 0:
        return whole

    nz = weights != 0
    # Rightmost nonzero column of each row, and lowest nonzero row of each
    # column; -1 for all-zero rows/columns so they constrain nothing.
    last_col = np.where(nz.any(axis=1), n_out - 1 - np.argmax(nz[:, ::-1], axis=1), -1)
    last_row = np.where(nz.any(axis=0), n_in - 1 - np.argmax(nz[::-1, :], axis=0), -1)
    reach_col = np.maximum.accumulate(last_col)  # cols touched by rows 0..i
    reach_row = np.maximum.accumulate(last_row)  # rows touching cols 0..j

    # Rows 0..i and cols 0..reach_col[i] form a closed block exactly when no
    # later row reaches back into those columns.
    rows = np.arange(n_in)
    cuts = np.flatnonzero((reach_col >= 0) & (reach_row[np.maximum(reach_col, 0)] <= rows))
    cuts = cuts.tolist()
    if not cuts or cuts[-1] != n_in - 1:
        cuts.append(n_in - 1)
    if len(cuts) == 1:
        return whole

    blocks: list[tuple[slice, slice]] = []
    row_start = col_start = 0
    for k, row_end in enumerate(cuts):
        col_end = n_out - 1 if k == len(cuts) - 1 else int(reach_col[row_end])
        block = (slice(row_start, row_end + 1), slice(col_start, col_end + 1))
        if block[1].stop == block[1].start and blocks:
            # Trailing all-zero rows: no outputs of their own, so fold them into
            # the previous block, where they multiply against zeros.
            prev_rows, prev_cols = blocks[-1]
            blocks[-1] = (slice(prev_rows.start, block[0].stop), prev_cols)
        else:
            blocks.append(block)
        row_start, col_start = row_end + 1, col_end + 1
    return blocks if len(blocks) > 1 else whole


def _merge_to(blocks: list[tuple[slice, slice]], min_rows: int) -> list[tuple[slice, slice]]:
    """Fuse adjacent blocks until each spans at least *min_rows* input channels."""
    merged: list[tuple[slice, slice]] = []
    rows = cols = None
    for block_rows, block_cols in blocks:
        rows = block_rows if rows is None else slice(rows.start, block_rows.stop)
        cols = block_cols if cols is None else slice(cols.start, block_cols.stop)
        if rows.stop - rows.start >= min_rows:
            merged.append((rows, cols))
            rows = cols = None
    if rows is not None:
        if merged:
            last_rows, last_cols = merged[-1]
            merged[-1] = (slice(last_rows.start, rows.stop), slice(last_cols.start, cols.stop))
        else:
            merged.append((rows, cols))
    return merged


def _weight_elements(blocks: list[tuple[slice, slice]]) -> int:
    return sum((r.stop - r.start) * (c.stop - c.start) for r, c in blocks)


def _cost(blocks: list[tuple[slice, slice]], n_samples: int, call_cost: float) -> float:
    return (n_samples + WEIGHT_LOAD_SAMPLES) * _weight_elements(blocks) + len(blocks) * call_cost


def _best_merge(
    blocks: list[tuple[slice, slice]], n_samples: int, call_cost: float
) -> tuple[list[tuple[slice, slice]], float]:
    """Cheapest way to fuse *blocks*, over merge granularities in powers of two."""
    best, best_cost = blocks, _cost(blocks, n_samples, call_cost)
    n_rows = blocks[-1][0].stop
    min_rows = 2
    while min_rows <= n_rows:
        candidate = _merge_to(blocks, min_rows)
        cost = _cost(candidate, n_samples, call_cost)
        if cost < best_cost:
            best, best_cost = candidate, cost
        min_rows *= 2
    return best, best_cost


def _permuted_partition(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[slice, slice]]] | None:
    """Find block structure that is only contiguous after reordering channels.

    Connected components of the bipartite graph of nonzero weights (input
    channels and output channels as the two node sets) give the finest possible
    decomposition, regardless of channel order.

    Returns ``(row_perm, col_perm, blocks)`` where ``weights[row_perm][:, col_perm]``
    is block-diagonal with the returned tiling, or ``None`` if there is nothing
    to find.
    """
    n_in, n_out = weights.shape
    rows, cols = np.nonzero(weights)
    if rows.size == 0:
        return None

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    shifted = cols + n_in
    adjacency = coo_matrix(
        (np.ones(rows.size * 2, dtype=bool), (np.concatenate([rows, shifted]), np.concatenate([shifted, rows]))),
        shape=(n_in + n_out, n_in + n_out),
    )
    n_components, labels = connected_components(adjacency, directed=False)
    if n_components <= 1:
        return None

    groups = []
    for component in range(n_components):
        members = np.flatnonzero(labels == component)
        in_idx = members[members < n_in]
        out_idx = members[members >= n_in] - n_in
        if in_idx.size and out_idx.size:
            groups.append((in_idx, out_idx))
    if len(groups) <= 1:
        return None

    # All-zero rows/columns belong to no component; park them in the last block
    # so the permutations stay complete and the tiling stays gap-free.
    used_in = np.concatenate([g[0] for g in groups])
    used_out = np.concatenate([g[1] for g in groups])
    spare_in = np.setdiff1d(np.arange(n_in), used_in, assume_unique=False)
    spare_out = np.setdiff1d(np.arange(n_out), used_out, assume_unique=False)
    groups[-1] = (np.concatenate([groups[-1][0], spare_in]), np.concatenate([groups[-1][1], spare_out]))

    row_perm = np.concatenate([g[0] for g in groups]).astype(np.intp)
    col_perm = np.concatenate([g[1] for g in groups]).astype(np.intp)
    blocks = []
    row_start = col_start = 0
    for in_idx, out_idx in groups:
        blocks.append((slice(row_start, row_start + in_idx.size), slice(col_start, col_start + out_idx.size)))
        row_start += in_idx.size
        col_start += out_idx.size
    return row_perm, col_perm, blocks


def plan_block_matmul(
    weights: np.ndarray,
    n_samples: int,
    *,
    force: bool = False,
    dispatched: bool = False,
) -> BlockPlan | None:
    """Choose between a dense matmul and a block-diagonal one.

    Args:
        weights: 2-D weight matrix in ``(n_in, n_out)`` orientation.
        n_samples: Representative number of samples per message (everything on
            the message except the channel axis). Feeds the cost model; short
            chunks favour the dense kernel because they cannot amortize the
            per-block call overhead.
        force: Return a block plan whenever *any* structure exists, ignoring
            the cost model. For tests and benchmarks.
        dispatched: Set for backends with heavier per-op overhead than numpy.

    Returns:
        A :class:`BlockPlan`, or ``None`` meaning "use a dense matmul".
    """
    if weights.ndim != 2:
        return None
    n_in, n_out = weights.shape
    call_cost = CALL_COST_MACS_DISPATCHED if dispatched else CALL_COST_MACS
    dense_cost = _cost([(slice(0, n_in), slice(0, n_out))], n_samples, call_cost)

    blocks = contiguous_block_partition(weights)
    if len(blocks) > 1:
        if force:
            return BlockPlan(tuple(blocks))
        best, cost = _best_merge(blocks, n_samples, call_cost)
        if len(best) > 1 and cost < dense_cost:
            return BlockPlan(tuple(best))

    if not force and n_in * n_out < PERMUTED_SEARCH_MIN_WEIGHTS:
        return None
    permuted = _permuted_partition(weights)
    if permuted is None:
        return None
    row_perm, col_perm, perm_blocks = permuted
    best, cost = (perm_blocks, 0.0) if force else _best_merge(perm_blocks, n_samples, call_cost)
    if len(best) <= 1:
        return None

    in_perm = None if np.array_equal(row_perm, np.arange(n_in)) else row_perm
    out_perm = None if np.array_equal(col_perm, np.arange(n_out)) else col_perm
    gathered = (n_in if in_perm is not None else 0) + (n_out if out_perm is not None else 0)
    cost += n_samples * GATHER_COST_MACS * gathered
    if not force and cost >= dense_cost:
        return None
    return BlockPlan(tuple(best), in_perm, out_perm)
