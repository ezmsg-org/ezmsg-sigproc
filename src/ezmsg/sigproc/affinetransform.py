"""Affine transformations via matrix multiplication: y = Ax or y = Ax + B.

For full matrix transformations where channels are mixed (off-diagonal weights),
use :obj:`AffineTransformTransformer` or the `AffineTransform` unit.

For simple per-channel scaling and offset (diagonal weights only), use
:obj:`LinearTransformTransformer` from :mod:`ezmsg.sigproc.linear` instead,
which is more efficient as it avoids matrix multiplication.
"""

import os
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from array_api_compat import get_namespace
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, AxisBase
from ezmsg.util.messages.util import replace


def _find_block_diagonal_clusters(weights: np.ndarray) -> list[np.ndarray] | None:
    """Detect block-diagonal structure in a square weight matrix.

    Finds connected components in the non-zero structure of the matrix.
    If the matrix has more than one component, returns a list of index
    arrays (one per block/cluster).

    Args:
        weights: Weight matrix. Must be square for detection to apply.

    Returns:
        List of sorted 1-D index arrays, one per cluster, or None if the matrix
        is not block-diagonal (single connected component or non-square).
    """
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        return None

    n = weights.shape[0]
    if n <= 1:
        return None

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    # Symmetric adjacency: i and j are in the same cluster if W[i,j] or W[j,i] is non-zero
    nonzero = weights != 0
    adjacency = nonzero | nonzero.T
    n_components, labels = connected_components(csr_matrix(adjacency), directed=False)

    if n_components <= 1:
        return None

    return [np.sort(np.where(labels == c)[0]) for c in range(n_components)]


def _max_cross_cluster_weight(weights: np.ndarray, clusters: list[np.ndarray]) -> float:
    """Return the maximum absolute weight between different clusters."""
    mask = np.zeros(weights.shape, dtype=bool)
    for indices in clusters:
        mask[np.ix_(indices, indices)] = True
    cross = np.abs(weights[~mask])
    return float(cross.max()) if cross.size > 0 else 0.0


def _merge_small_clusters(clusters: list[np.ndarray], min_size: int) -> list[np.ndarray]:
    """Merge clusters smaller than *min_size* into combined groups.

    Small clusters are greedily concatenated until each merged group has
    at least *min_size* channels. Any leftover small clusters that don't
    reach the threshold are combined into a final group.

    The merged group's sub-weight-matrix will contain the original small
    diagonal blocks with zeros between them — a dense matmul on that
    sub-matrix is cheaper than iterating over many tiny matmuls.
    """
    if min_size <= 1:
        return clusters

    large = []
    small = []
    for c in clusters:
        if len(c) >= min_size:
            large.append(c)
        else:
            small.append(c)

    if not small:
        return clusters

    current: list[np.ndarray] = []
    current_size = 0
    for c in small:
        current.append(c)
        current_size += len(c)
        if current_size >= min_size:
            large.append(np.sort(np.concatenate(current)))
            current = []
            current_size = 0

    if current:
        large.append(np.sort(np.concatenate(current)))

    return large


class AffineTransformSettings(ez.Settings):
    """
    Settings for :obj:`AffineTransform`.
    """

    weights: np.ndarray | str | Path
    """An array of weights or a path to a file with weights compatible with np.loadtxt."""

    axis: str | None = None
    """The name of the axis to apply the transformation to. Defaults to the leading (0th) axis in the array."""

    right_multiply: bool = True
    """Set False to transpose the weights before applying."""

    channel_clusters: list[list[int]] | None = None
    """Optional explicit channel cluster specification for block-diagonal optimization.
    Each inner list contains channel indices belonging to one cluster. When provided,
    the weight matrix is decomposed into per-cluster sub-matrices and multiplied
    separately, which is faster when cross-cluster weights are zero.

    If None and the weight matrix is square, block-diagonal structure is
    auto-detected from the zero pattern of the weights."""

    min_cluster_size: int = 32
    """Minimum number of channels per cluster for the block-diagonal optimization.
    Clusters smaller than this are greedily merged together to avoid excessive
    Python loop overhead. Set to 1 to disable merging."""


@processor_state
class AffineTransformState:
    weights: npt.NDArray | None = None
    new_axis: AxisBase | None = None
    cluster_perm: npt.NDArray | None = None
    cluster_inv_perm: npt.NDArray | None = None
    cluster_weights: list | None = None


class AffineTransformTransformer(
    BaseStatefulTransformer[AffineTransformSettings, AxisArray, AxisArray, AffineTransformState]
):
    """Apply affine transformation via matrix multiplication: y = Ax or y = Ax + B.

    Use this transformer when you need full matrix transformations that mix
    channels (off-diagonal weights), such as spatial filters or projections.

    For simple per-channel scaling and offset where each output channel depends
    only on its corresponding input channel (diagonal weight matrix), use
    :obj:`LinearTransformTransformer` instead, which is more efficient.

    The weights matrix can include an offset row (stacked as [A|B]) where the
    input is automatically augmented with a column of ones to compute y = Ax + B.
    """

    def __call__(self, message: AxisArray) -> AxisArray:
        # Override __call__ so we can shortcut if weights are None.
        if self.settings.weights is None or (
            isinstance(self.settings.weights, str) and self.settings.weights == "passthrough"
        ):
            return message
        return super().__call__(message)

    def _hash_message(self, message: AxisArray) -> int:
        return hash(message.key)

    def _reset_state(self, message: AxisArray) -> None:
        weights = self.settings.weights
        if isinstance(weights, str):
            weights = Path(os.path.abspath(os.path.expanduser(weights)))
        if isinstance(weights, Path):
            weights = np.loadtxt(weights, delimiter=",")
        if not self.settings.right_multiply:
            weights = weights.T
        if weights is not None:
            weights = np.ascontiguousarray(weights)

        self._state.weights = weights

        # Note: If weights were scipy.sparse BSR then maybe we could use automate this next part.
        #  However, that would break compatibility with Array API.

        # --- Block-diagonal cluster detection ---
        clusters = None
        if self.settings.channel_clusters is not None:
            if weights.shape[0] != weights.shape[1]:
                ez.logger.warning(
                    "channel_clusters ignored: weight matrix is not square " f"({weights.shape[0]}x{weights.shape[1]})"
                )
            else:
                clusters = [np.asarray(group) for group in self.settings.channel_clusters]
                all_idx = np.concatenate(clusters)
                n_unique = len(np.unique(all_idx))
                if n_unique != weights.shape[0]:
                    raise ValueError(
                        "channel_clusters must cover all channel indices 0..n-1 "
                        f"(expected {weights.shape[0]} unique indices, got {n_unique})"
                    )
                max_cross = _max_cross_cluster_weight(weights, clusters)
                if max_cross > 0:
                    ez.logger.warning(
                        f"Non-zero cross-cluster weights detected (max abs: {max_cross:.2e}). "
                        "These will be ignored in block-diagonal multiplication."
                    )
        elif weights.shape[0] == weights.shape[1]:
            clusters = _find_block_diagonal_clusters(weights)
            if clusters is not None:
                ez.logger.info(
                    f"Auto-detected {len(clusters)} block-diagonal clusters " f"(sizes: {[len(c) for c in clusters]})"
                )

        # Merge small clusters to avoid excessive loop overhead
        if clusters is not None:
            clusters = _merge_small_clusters(clusters, self.settings.min_cluster_size)

        if clusters is not None and len(clusters) > 1:
            perm = np.concatenate(clusters)
            inv_perm = np.empty_like(perm)
            inv_perm[perm] = np.arange(len(perm))
            self._state.cluster_weights = [np.ascontiguousarray(weights[np.ix_(idx, idx)]) for idx in clusters]
            self._state.cluster_perm = perm
            self._state.cluster_inv_perm = inv_perm
            self._state.weights = None
        else:
            self._state.cluster_perm = None
            self._state.cluster_inv_perm = None
            self._state.cluster_weights = None

        # --- Axis label handling (for non-square transforms) ---
        axis = self.settings.axis or message.dims[-1]
        if axis in message.axes and hasattr(message.axes[axis], "data") and weights.shape[0] != weights.shape[1]:
            in_labels = message.axes[axis].data
            new_labels = []
            n_in, n_out = weights.shape
            if len(in_labels) != n_in:
                ez.logger.warning(f"Received {len(in_labels)} for {n_in} inputs. Check upstream labels.")
            else:
                b_filled_outputs = np.any(weights, axis=0)
                b_used_inputs = np.any(weights, axis=1)
                if np.all(b_used_inputs) and np.all(b_filled_outputs):
                    new_labels = []
                elif np.all(b_used_inputs):
                    in_ix = 0
                    new_labels = []
                    for out_ix in range(n_out):
                        if b_filled_outputs[out_ix]:
                            new_labels.append(in_labels[in_ix])
                            in_ix += 1
                        else:
                            new_labels.append("")
                elif np.all(b_filled_outputs):
                    new_labels = np.array(in_labels)[b_used_inputs]

            self._state.new_axis = replace(message.axes[axis], data=np.array(new_labels))

        # Convert to match message.data namespace for efficient operations in _process
        xp = get_namespace(message.data)
        if self._state.weights is not None:
            self._state.weights = xp.asarray(self._state.weights)
        if self._state.cluster_perm is not None:
            self._state.cluster_perm = xp.asarray(self._state.cluster_perm)
            self._state.cluster_inv_perm = xp.asarray(self._state.cluster_inv_perm)
            self._state.cluster_weights = [xp.asarray(w) for w in self._state.cluster_weights]

    def _block_diagonal_matmul(self, xp, data, axis_idx):
        """Perform matmul using block-diagonal decomposition.

        Permutes channels into cluster-sorted order, performs per-cluster
        matmuls on contiguous slices, then restores the original channel order.
        """
        needs_permute = axis_idx not in [-1, data.ndim - 1]
        if needs_permute:
            dim_perm = list(range(data.ndim))
            dim_perm.append(dim_perm.pop(axis_idx))
            data = xp.permute_dims(data, dim_perm)

        # Sort channels by cluster for contiguous access (last axis)
        sorted_data = xp.take(data, self._state.cluster_perm, axis=data.ndim - 1)

        # Matmul each cluster block
        pieces = []
        offset = 0
        for sub_weights in self._state.cluster_weights:
            n = sub_weights.shape[0]
            pieces.append(xp.matmul(sorted_data[..., offset : offset + n], sub_weights))
            offset += n

        # Concatenate and restore original channel order
        sorted_result = xp.concat(pieces, axis=data.ndim - 1)
        result = xp.take(sorted_result, self._state.cluster_inv_perm, axis=data.ndim - 1)

        if needs_permute:
            inv_dim_perm = list(range(data.ndim))
            inv_dim_perm.insert(axis_idx, inv_dim_perm.pop(-1))
            result = xp.permute_dims(result, inv_dim_perm)

        return result

    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        data = message.data

        if self._state.cluster_perm is not None:
            data = self._block_diagonal_matmul(xp, data, axis_idx)
        else:
            if data.shape[axis_idx] == (self._state.weights.shape[0] - 1):
                # The weights are stacked A|B where A is the transform and B is a single row
                #  in the equation y = Ax + B. This supports NeuroKey's weights matrices.
                sample_shape = data.shape[:axis_idx] + (1,) + data.shape[axis_idx + 1 :]
                data = xp.concat((data, xp.ones(sample_shape, dtype=data.dtype)), axis=axis_idx)

            if axis_idx in [-1, len(message.dims) - 1]:
                data = xp.matmul(data, self._state.weights)
            else:
                perm = list(range(data.ndim))
                perm.append(perm.pop(axis_idx))
                data = xp.permute_dims(data, perm)
                data = xp.matmul(data, self._state.weights)
                inv_perm = list(range(data.ndim))
                inv_perm.insert(axis_idx, inv_perm.pop(-1))
                data = xp.permute_dims(data, inv_perm)

        replace_kwargs = {"data": data}
        if self._state.new_axis is not None:
            replace_kwargs["axes"] = {**message.axes, axis: self._state.new_axis}

        return replace(message, **replace_kwargs)


class AffineTransform(BaseTransformerUnit[AffineTransformSettings, AxisArray, AxisArray, AffineTransformTransformer]):
    SETTINGS = AffineTransformSettings


def affine_transform(
    weights: np.ndarray | str | Path,
    axis: str | None = None,
    right_multiply: bool = True,
    channel_clusters: list[list[int]] | None = None,
    min_cluster_size: int = 32,
) -> AffineTransformTransformer:
    """
    Perform affine transformations on streaming data.

    Args:
        weights: An array of weights or a path to a file with weights compatible with np.loadtxt.
        axis: The name of the axis to apply the transformation to. Defaults to the leading (0th) axis in the array.
        right_multiply: Set False to transpose the weights before applying.
        channel_clusters: Optional explicit channel cluster specification. See
            :attr:`AffineTransformSettings.channel_clusters`.
        min_cluster_size: Minimum channels per cluster; smaller clusters are merged. See
            :attr:`AffineTransformSettings.min_cluster_size`.

    Returns:
        :obj:`AffineTransformTransformer`.
    """
    return AffineTransformTransformer(
        AffineTransformSettings(
            weights=weights,
            axis=axis,
            right_multiply=right_multiply,
            channel_clusters=channel_clusters,
            min_cluster_size=min_cluster_size,
        )
    )


def zeros_for_noop(data, **ignore_kwargs):
    xp = get_namespace(data)
    return xp.zeros_like(data)


class CommonRereferenceSettings(ez.Settings):
    """
    Settings for :obj:`CommonRereference`
    """

    mode: str = "mean"
    """The statistical mode to apply -- either "mean" or "median"."""

    axis: str | None = None
    """The name of the axis to apply the transformation to."""

    include_current: bool = True
    """Set False to exclude each channel from participating in the calculation of its reference."""


class CommonRereferenceTransformer(BaseTransformer[CommonRereferenceSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        if self.settings.mode == "passthrough":
            return message

        xp = get_namespace(message.data)
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)

        func = {"mean": xp.mean, "median": np.median, "passthrough": zeros_for_noop}[self.settings.mode]

        ref_data = func(message.data, axis=axis_idx, keepdims=True)

        if not self.settings.include_current:
            # Typical `CAR = x[0]/N + x[1]/N + ... x[i-1]/N + x[i]/N + x[i+1]/N + ... + x[N-1]/N`
            # and is the same for all i, so it is calculated only once in `ref_data`.
            # However, if we had excluded the current channel,
            # then we would have omitted the contribution of the current channel:
            # `CAR[i] = x[0]/(N-1) + x[1]/(N-1) + ... x[i-1]/(N-1) + x[i+1]/(N-1) + ... + x[N-1]/(N-1)`
            # The majority of the calculation is the same as when the current channel is included;
            # we need only rescale CAR so the divisor is `N-1` instead of `N`, then subtract the contribution
            # from the current channel (i.e., `x[i] / (N-1)`)
            #  i.e., `CAR[i] = (N / (N-1)) * common_CAR - x[i]/(N-1)`
            # We can use broadcasting subtraction instead of looping over channels.
            N = message.data.shape[axis_idx]
            ref_data = (N / (N - 1)) * ref_data - message.data / (N - 1)
            # Note: I profiled using AffineTransformTransformer; it's ~30x slower than this implementation.

        return replace(message, data=message.data - ref_data)


class CommonRereference(
    BaseTransformerUnit[CommonRereferenceSettings, AxisArray, AxisArray, CommonRereferenceTransformer]
):
    SETTINGS = CommonRereferenceSettings


def common_rereference(
    mode: str = "mean", axis: str | None = None, include_current: bool = True
) -> CommonRereferenceTransformer:
    """
    Perform common average referencing (CAR) on streaming data.

    Args:
        mode: The statistical mode to apply -- either "mean" or "median"
        axis: The name of hte axis to apply the transformation to.
        include_current: Set False to exclude each channel from participating in the calculation of its reference.

    Returns:
        :obj:`CommonRereferenceTransformer`
    """
    return CommonRereferenceTransformer(
        CommonRereferenceSettings(mode=mode, axis=axis, include_current=include_current)
    )
