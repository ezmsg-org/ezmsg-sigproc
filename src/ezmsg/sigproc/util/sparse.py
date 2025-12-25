"""Methods for sparse array signal processing operations."""

import numpy as np
import sparse


def sliding_win_oneaxis_old(s: sparse.SparseArray, nwin: int, axis: int, step: int = 1) -> sparse.SparseArray:
    """
    Like `ezmsg.util.messages.axisarray.sliding_win_oneaxis` but for sparse arrays.
    This approach is about 4x slower than the version that uses coordinate arithmetic below.

    Args:
        s: The input sparse array.
        nwin: The size of the sliding window.
        axis: The axis along which the sliding window will be applied.
        step: The size of the step between windows. If > 1, the strided window will be sliced with `slice_along_axis`.

    Returns:

    """
    if -s.ndim <= axis < 0:
        axis = s.ndim + axis
    targ_slices = [slice(_, _ + nwin) for _ in range(0, s.shape[axis] - nwin + 1, step)]
    s = s.reshape(s.shape[:axis] + (1,) + s.shape[axis:])
    full_slices = (slice(None),) * s.ndim
    full_slices = [full_slices[: axis + 1] + (sl,) + full_slices[axis + 2 :] for sl in targ_slices]
    result = sparse.concatenate([s[_] for _ in full_slices], axis=axis)
    return result


def sliding_win_oneaxis(s: sparse.SparseArray, nwin: int, axis: int, step: int = 1) -> sparse.SparseArray:
    """
    Generates a view-like sparse array using a sliding window of specified length along a specified axis.
    Sparse analog of an optimized dense as_strided-based implementation with these properties:

    - Accepts a single `nwin` and a single `axis`.
    - Inserts a new 'win' axis immediately BEFORE the original target axis.
      Output shape:
        s.shape[:axis] + (W,) + (nwin,) + s.shape[axis+1:]
      where W = s.shape[axis] - (nwin - 1).
    - If `step > 1`, stepping is applied by slicing along the new windows axis (same observable behavior
      as doing `slice_along_axis(result, slice(None, None, step), axis)` in the dense version).

    Args:
        s: Input sparse array (pydata/sparse COO-compatible).
        nwin: Sliding window size (must be > 0).
        axis: Axis of `s` along which the window slides (supports negative indexing).
        step: Stride between windows. If > 1, applied by slicing the windows axis after construction.

    Returns:
        A sparse array with a new windows axis inserted before the original axis.

    Notes:
        - Mirrors the dense function’s known edge case: when nwin == shape[axis] + 1, W becomes 0 and
          an empty windows axis is returned.
        - Built by coordinate arithmetic; no per-window indexing or concatenation.
    """
    if -s.ndim <= axis < 0:
        axis = s.ndim + axis
    if not (0 <= axis < s.ndim):
        raise ValueError(f"Invalid axis {axis} for array with {s.ndim} dimensions")
    if nwin <= 0:
        raise ValueError("nwin must be > 0")
    dim = s.shape[axis]

    last_win_start = dim - nwin
    win_starts = list(range(0, last_win_start + 1, step))
    n_win_out = len(win_starts)
    if n_win_out <= 0:
        # Return array with proper shape except empty along windows axis
        return sparse.zeros(s.shape[:axis] + (0,) + (nwin,) + s.shape[axis + 1 :], dtype=s.dtype)

    coo = s.asformat("coo")
    coords = coo.coords  # shape: (ndim, nnz)
    data = coo.data  # shape: (nnz,)
    ia = coords[axis]  # indices along sliding axis, shape: (nnz,)

    # We emit contributions for each offset o in [0, nwin-1].
    # For a nonzero at index i, it contributes to window start w = i - o when 0 <= w < W.
    out_coords_blocks = []
    out_data_blocks = []

    # Small speed/memory tweak: reuse dtypes and pre-allocate o-array once per loop.
    idx_dtype = coords.dtype

    for win_ix, win_start in enumerate(win_starts):
        w = ia - win_start
        # Valid window starts are those within [0, nwin]
        mask = (w >= 0) & (w < nwin)
        if not mask.any():
            continue

        sel = np.nonzero(mask)[0]
        w_sel = w[sel]

        # Build new coords with windows axis inserted at `axis` and the original axis
        # becoming the next axis with fixed offset value `o`.
        # Output ndim = s.ndim + 1
        before = coords[:axis, sel]  # unchanged
        after_other = coords[axis + 1 :, sel]  # dims after original axis
        win_idx_row = np.full((1, sel.size), win_ix, dtype=idx_dtype)

        new_coords = np.vstack([before, win_idx_row, w_sel[None, :], after_other])

        out_coords_blocks.append(new_coords)
        out_data_blocks.append(data[sel])

    if not out_coords_blocks:
        return sparse.zeros(s.shape[:axis] + (n_win_out,) + (nwin,) + s.shape[axis + 1 :], dtype=s.dtype)

    out_coords = np.hstack(out_coords_blocks)
    out_data = np.hstack(out_data_blocks)
    out_shape = s.shape[:axis] + (n_win_out,) + (nwin,) + s.shape[axis + 1 :]

    return sparse.COO(out_coords, out_data, shape=out_shape)
