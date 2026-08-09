"""Deterministic rereferencing schemes expressed as affine weight matrices.

These helpers build weight matrices for deterministic, group-aware
rereferencing — identity passthrough and per-group common-average
reference (CAR) — suitable for
:class:`ezmsg.sigproc.affinetransform.AffineTransformTransformer`
(``y = x @ A``). All returned matrices are symmetric, so the
``right_multiply`` orientation does not matter.

Matrices are built with numpy: ``AffineTransformTransformer`` converts its
weights to the message's array namespace, dtype, and device on first use, so
a host-side float64 matrix is the correct interchange format even for
GPU-backed streams.

The matrix form of mean-CAR is equivalent to the streaming
:class:`ezmsg.sigproc.affinetransform.CommonRereferenceTransformer` with
``mode="mean"``. The streaming form is cheaper per sample (mean subtraction
vs. matmul) and also supports ``median``; the matrix form composes with
other affine transforms and provides a deterministic cold-start for adaptive
rereferencing (e.g. LRR in ezmsg-learn).
"""

from __future__ import annotations

import enum

import numpy as np
import numpy.typing as npt

from ezmsg.sigproc.util.channels import validate_channel_groups


class RereferenceKind(str, enum.Enum):
    """Deterministic rereference schemes expressible as a weight matrix.

    ``str`` enum so values round-trip through config as their plain strings.
    """

    IDENTITY = "identity"
    """Pass the signal through unchanged."""

    CAR = "car"
    """Per-group common-average reference."""


def car_matrix(
    n_channels: int,
    *,
    groups: list[list[int]] | None = None,
    include_current: bool = True,
    min_reref_size: int = 1,
    dtype: npt.DTypeLike = np.float64,
) -> np.ndarray:
    """Build a per-group common-average-reference matrix.

    Within each group of ``k`` channels the sub-block subtracts the group
    mean: ``y_i = x_i - mean_j x_j`` (``include_current=True``), or the
    leave-one-out mean ``y_i = x_i - mean_{j != i} x_j``
    (``include_current=False``). Channels outside every group — and all
    cross-group terms — stay identity.

    Args:
        n_channels: Total number of channels (matrix is ``n x n``).
        groups: Disjoint channel-index groups, e.g. from
            :func:`ezmsg.sigproc.util.channels.channel_groups_from_field`.
            ``None`` treats all channels as a single group.
        include_current: Set False for the leave-one-out reference (each
            channel excluded from its own reference).
        min_reref_size: Groups with fewer channels than this stay identity.
        dtype: Result dtype; must be floating-point.

    Returns:
        Symmetric ``(n_channels, n_channels)`` numpy weight matrix.
    """
    A = np.eye(n_channels, dtype=dtype)
    if groups is None:
        groups = [list(range(n_channels))]
    validate_channel_groups(groups, n_channels)
    # Leave-one-out needs at least one *other* channel (k == 1 would divide by zero).
    min_k = max(min_reref_size, 1 if include_current else 2)
    for group in groups:
        idx = np.asarray(group, dtype=np.intp)
        k = idx.size
        if k < min_k:
            continue
        if include_current:
            block = np.eye(k, dtype=dtype) - 1.0 / k
        else:
            block = (k / (k - 1.0)) * np.eye(k, dtype=dtype) - 1.0 / (k - 1.0)
        A[np.ix_(idx, idx)] = block
    return A


def rereference_matrix(
    kind: RereferenceKind | str,
    n_channels: int,
    *,
    groups: list[list[int]] | None = None,
    include_current: bool = True,
    min_reref_size: int = 1,
    dtype: npt.DTypeLike = np.float64,
) -> np.ndarray:
    """Build the weight matrix for a :class:`RereferenceKind`.

    Dispatches to :func:`car_matrix` for ``CAR``; the group/reference
    arguments are ignored for ``IDENTITY``. Accepts the enum or its plain
    string value.
    """
    kind = RereferenceKind(kind)
    if kind == RereferenceKind.IDENTITY:
        return np.eye(n_channels, dtype=dtype)
    return car_matrix(
        n_channels,
        groups=groups,
        include_current=include_current,
        min_reref_size=min_reref_size,
        dtype=dtype,
    )
