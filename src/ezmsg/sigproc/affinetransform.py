"""Affine transformations via matrix multiplication: y = Ax or y = Ax + B.

For full matrix transformations where channels are mixed (off-diagonal weights),
use :obj:`AffineTransformTransformer` or the `AffineTransform` unit.

For simple per-channel scaling and offset (diagonal weights only), use
:obj:`LinearTransformTransformer` from :mod:`ezmsg.sigproc.linear` instead,
which is more efficient as it avoids matrix multiplication.

Both transformers here take a :data:`~ezmsg.sigproc.util.channels.ChannelGroupSpec`
as ``channel_groups``: explicit index groups, or the name of a channel-metadata
field (``"bank"``, ``"array"``, ...) to group by. For :obj:`CommonRereference`
the groups say which channels share a reference. For :obj:`AffineTransform` they
only shape the *construction* of deterministic weights — the block structure the
matmul exploits is always read off the weight matrix itself.
"""

import inspect
import math
import os
from collections.abc import Callable
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from array_api_compat import get_namespace
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, AxisBase
from ezmsg.util.messages.util import replace

from ezmsg.sigproc.util.array import array_device, is_float_dtype, xp_asarray, xp_copy, xp_create, xp_empty
from ezmsg.sigproc.util.blockdiag import plan_block_matmul
from ezmsg.sigproc.util.channels import ChannelGroupSpec, group_spec_fingerprint, resolve_channel_groups
from ezmsg.sigproc.util.rereference import RereferenceKind, rereference_matrix

KERNELS = ("auto", "dense", "blocks")
"""Valid values for :attr:`AffineTransformSettings.kernel`."""


def _is_dispatched(xp) -> bool:
    """True for backends whose per-op overhead is well above numpy's."""
    return "numpy" not in xp.__name__


def _supports_matmul_out(xp, dtype, device) -> bool:
    """Whether ``xp.matmul(..., out=view)`` works, so blocks can fill in place.

    numpy, cupy and torch accept ``out=``; MLX does not. Probed once per state
    reset rather than sniffed from the namespace name, since which backends
    support it is a moving target.
    """
    try:
        a = xp_create(xp.zeros, (1, 1), dtype=dtype, device=device)
        b = xp_create(xp.zeros, (1, 1), dtype=dtype, device=device)
        xp.matmul(a, a, out=b)
    except Exception:
        return False
    return True


def _matmul_add(xp, data, weights, bias):
    """``data @ weights + bias``, using the backend's fused kernel when it has one.

    The obvious formulation for stacked ``A|B`` weights is to glue a column of
    ones onto the data and do one matmul, but that materializes a copy of the
    whole message every cycle just to carry a constant. Broadcasting the bias
    instead is cheaper everywhere, and MLX additionally offers ``addmm``, which
    folds the add into the matmul epilogue: measured on an M4 Pro, concat is
    1.26-1.33x slower than ``addmm`` at 30x256 and 128x512, 1.07x at 512x1024.
    """
    addmm = getattr(xp, "addmm", None)
    if addmm is not None:
        try:
            return addmm(bias, data, weights)
        except (TypeError, ValueError):
            pass
    return xp.matmul(data, weights) + bias


def _call_weight_factory(factory: Callable, n_in: int, groups: list[list[int]] | None):
    """Call a user weights factory as ``f(n_in)`` or ``f(n_in, groups)``.

    The two-argument form lets a factory build weights from channel metadata
    (which it cannot see otherwise) without every caller having to accept it.
    """
    try:
        params = inspect.signature(factory).parameters.values()
        positional = sum(1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        variadic = any(p.kind is p.VAR_POSITIONAL for p in params)
    except (TypeError, ValueError):  # builtins and C callables have no signature
        positional, variadic = 1, False
    if positional >= 2 or variadic:
        return factory(n_in, groups)
    return factory(n_in)


class AffineTransformSettings(ez.Settings):
    """
    Settings for :obj:`AffineTransform`.
    """

    weights: np.ndarray | str | Path | RereferenceKind | Callable[[int], np.ndarray]
    """An array of weights; a path to a file with weights compatible with np.loadtxt;
    a :class:`~ezmsg.sigproc.util.rereference.RereferenceKind` or its string value
    (e.g. ``"car"``) to build a deterministic rereference matrix over
    ``channel_groups``; or a callable accepting ``n_in: int`` (optionally also the
    resolved ``groups``) and returning an ndarray of shape ``(n_in, n_out)``.

    Note: if you simply want streaming CAR, :obj:`CommonRereference` in this module
    is usually the better choice (per-sample mean subtraction instead of a matmul,
    plus ``median`` support). Kind-based weights are useful for discovering the
    available deterministic transforms and for workflows that start from such a
    matrix and later replace it externally via
    :meth:`AffineTransformTransformer.set_weights`. For variants (leave-one-out,
    minimum group size) pass a callable, e.g.
    ``lambda n, groups: car_matrix(n, groups=groups, include_current=False)``."""

    axis: str | None = None
    """The name of the axis to apply the transformation to. Defaults to the leading (0th) axis in the array."""

    right_multiply: bool = True
    """Set False to transpose the weights before applying."""

    channel_groups: ChannelGroupSpec | None = None
    """How to group input channels when *building* the weight matrix.

    Applies only when ``weights`` is a
    :class:`~ezmsg.sigproc.util.rereference.RereferenceKind` or a callable —
    e.g. ``weights="car", channel_groups="bank"`` builds a per-bank common
    average reference from the channel axis metadata. See
    :data:`~ezmsg.sigproc.util.channels.ChannelGroupSpec` for the accepted forms.

    It has **no effect** when ``weights`` is an explicit array or file: the block
    structure exploited by the matmul is always derived from the weight matrix
    itself, so a grouping that disagreed with the weights could never change the
    result (ezmsg-org/ezmsg-sigproc#198)."""

    kernel: str = "auto"
    """Matmul kernel: ``"auto"`` (default) picks between a dense matmul and a
    block-diagonal one from the structure of the weights and the message size;
    ``"dense"`` and ``"blocks"`` force the choice. See
    :mod:`ezmsg.sigproc.util.blockdiag` for the cost model behind ``"auto"``."""


@processor_state
class AffineTransformState:
    weights: npt.NDArray | None = None
    """Full weight matrix for the dense kernel; None when blocks are in use."""
    stacked_split: tuple | None = None
    """``(A, B)`` views of stacked ``A|B`` weights, built on first use."""
    blocks: list | None = None
    """list of (in_slice, out_slice, sub_weights) for the block-diagonal kernel."""
    in_perm: npt.NDArray | None = None
    """Channel gather that makes the blocks contiguous, or None if they already are."""
    out_perm: npt.NDArray | None = None
    """Output-channel counterpart of ``in_perm``, used to slice the weights."""
    out_inv_perm: npt.NDArray | None = None
    """Gather that undoes ``out_perm`` on the result."""
    out_dtype: npt.DTypeLike | None = None
    """Result dtype of a block matmul, resolved once against the message dtype."""
    fill_in_place: bool = False
    """Whether the backend accepts ``matmul(..., out=)``; else blocks are concatenated."""
    device: object = None
    """Device the output buffer is allocated on, resolved once at reset."""
    new_axis: AxisBase | None = None
    n_out: int = 0
    n_in: int = 0
    """Channels expected on the message; blocks require it to equal ``weights.shape[0]``."""
    n_samples: int = 1
    """Representative samples per message, for the kernel cost model."""
    dispatched: bool = False


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
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        return hash(
            (message.key, message.data.shape[axis_idx])
            + group_spec_fingerprint(message, axis, self.settings.channel_groups)
        )

    def _reset_state(self, message: AxisArray) -> None:
        if self.settings.kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {self.settings.kernel!r}")

        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        n_in = message.data.shape[axis_idx]
        xp = get_namespace(message.data)

        weights = self.settings.weights
        if isinstance(weights, str):
            # Bare strings may name a RereferenceKind (e.g. "car" from config);
            # anything else is treated as a weights file path below.
            try:
                weights = RereferenceKind(weights)
            except ValueError:
                pass
        if isinstance(weights, RereferenceKind) or callable(weights):
            groups = resolve_channel_groups(message, axis, self.settings.channel_groups)
            group_lists = None if groups is None else [group.tolist() for group in groups]
            if isinstance(weights, RereferenceKind):
                weights = rereference_matrix(weights, n_in, groups=group_lists)
            else:
                weights = _call_weight_factory(weights, n_in, group_lists)
        if isinstance(weights, str):
            weights = Path(os.path.abspath(os.path.expanduser(weights)))
        if isinstance(weights, Path):
            weights = np.loadtxt(weights, delimiter=",")
        if not self.settings.right_multiply:
            weights = weights.T
        weights = np.ascontiguousarray(weights)

        # Context the kernel planner needs, set before set_weights() consults it.
        self._state.n_in = n_in
        # math.prod(shape), not .size: torch spells size as a *method*, so
        # `data.size // n_in` raises TypeError on a torch-backed message.
        n_elem = math.prod(message.data.shape)
        self._state.n_samples = max(n_elem // n_in, 1) if n_in else 1
        self._state.dispatched = _is_dispatched(xp)
        self.set_weights(weights, recalc_structure=True)

        # --- Axis label handling (for non-square transforms) ---
        n_in, n_out = weights.shape
        if axis in message.axes and hasattr(message.axes[axis], "data") and n_in != n_out:
            in_labels = message.axes[axis].data
            new_labels = []
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

        # Convert to match message.data namespace and device for _process.
        # Weights are numpy float64 up to here; some devices (e.g. MPS) don't
        # support float64, so downcast to the message's dtype when it is floating.
        dev = array_device(message.data)
        msg_dt = message.data.dtype
        w_dt = msg_dt if is_float_dtype(xp, msg_dt) else None
        if self._state.weights is not None:
            self._state.weights = xp_asarray(xp, self._state.weights, dtype=w_dt, device=dev)
            self._state.stacked_split = None
        if self._state.blocks is not None:
            self._state.blocks = [
                (in_slice, out_slice, xp_asarray(xp, sub_w, dtype=w_dt, device=dev))
                for in_slice, out_slice, sub_w in self._state.blocks
            ]
            if w_dt is not None:
                out_dtype = msg_dt  # weights were downcast to the message's float dtype
            else:
                try:
                    out_dtype = np.result_type(msg_dt, np.float64)
                except TypeError:
                    out_dtype = None  # non-numpy integer dtype; concatenate instead
            self._state.out_dtype = out_dtype
            self._state.device = dev
            self._state.fill_in_place = out_dtype is not None and _supports_matmul_out(xp, out_dtype, dev)
            for name in ("in_perm", "out_perm", "out_inv_perm"):
                perm = getattr(self._state, name)
                if perm is not None:
                    setattr(self._state, name, xp_asarray(xp, perm, device=dev))

    def set_weights(self, weights, *, recalc_structure: bool = False) -> None:
        """Replace weight values, optionally re-deriving the matmul kernel.

        *weights* must be in **canonical orientation** (``right_multiply``
        already applied by the caller or by ``_reset_state``). The array may
        live in any Array-API namespace (NumPy, CuPy, etc.).

        Args:
            weights: Weight matrix in canonical orientation.
            recalc_structure: When True, re-derive the block-diagonal structure
                from *weights* and re-choose the kernel. When False (default),
                keep the existing block layout and only refresh the values --
                appropriate for an adaptive filter whose sparsity pattern is
                fixed. Note that re-deriving reads the whole matrix on the host,
                so avoid it on a hot path with device-resident weights.
        """
        if recalc_structure:
            w_np = np.ascontiguousarray(weights)
            n_in, n_out = w_np.shape
            plan = None
            if self.settings.kernel != "dense" and n_in == self._state.n_in:
                # A mismatch means the [A|B] offset form, whose ones-column
                # augmentation the block kernel does not implement.
                plan = plan_block_matmul(
                    w_np,
                    self._state.n_samples,
                    force=self.settings.kernel == "blocks",
                    dispatched=self._state.dispatched,
                )
            self._state.n_out = n_out
            self._state.blocks = None if plan is None else [(in_sl, out_sl, None) for in_sl, out_sl in plan.blocks]
            self._state.in_perm = None if plan is None else plan.in_perm
            self._state.out_perm = None if plan is None else plan.out_perm
            self._state.out_inv_perm = None
            if self._state.out_perm is not None:
                inverse = np.empty(n_out, dtype=np.intp)
                inverse[plan.out_perm] = np.arange(n_out)
                self._state.out_inv_perm = inverse
            if plan is not None:
                ez.logger.info(
                    f"AffineTransform: block-diagonal kernel with {len(plan.blocks)} blocks "
                    f"(sizes: {[(r.stop - r.start, c.stop - c.start) for r, c in plan.blocks]})"
                )

        if self._state.blocks is None:
            self._state.weights = weights
            self._state.stacked_split = None
            return

        xp = get_namespace(weights)
        permuted = weights
        if self._state.in_perm is not None:
            permuted = xp.take(permuted, self._state.in_perm, axis=0)
        if self._state.out_perm is not None:
            permuted = xp.take(permuted, self._state.out_perm, axis=1)
        # Copy each sub-block rather than keeping a view into *weights*. Views
        # would pin the whole dense matrix alive to hold the (much smaller) block
        # diagonal, and would let a caller that recycles its weight buffer mutate
        # our state between messages.
        self._state.blocks = [
            (in_slice, out_slice, xp_copy(permuted[in_slice, out_slice]))
            for in_slice, out_slice, _ in self._state.blocks
        ]
        self._state.weights = None
        self._state.stacked_split = None

    def _block_matmul(self, xp, data, axis_idx):
        """Multiply by a block-diagonal weight matrix, one contiguous block at a time.

        Basic slicing gives views on both sides, so there is no gather and no
        scatter: each block reads a strided window of the input and writes its
        own window of the output. The blocks tile the output exactly, so the
        buffer never needs zeroing.
        """
        state = self._state
        needs_permute = axis_idx not in (-1, data.ndim - 1)
        if needs_permute:
            dim_perm = list(range(data.ndim))
            dim_perm.append(dim_perm.pop(axis_idx))
            data = xp.permute_dims(data, dim_perm)
        if state.in_perm is not None:
            data = xp.take(data, state.in_perm, axis=data.ndim - 1)

        if state.fill_in_place:
            result = xp_empty(xp, data.shape[:-1] + (state.n_out,), dtype=state.out_dtype, device=state.device)
            for in_slice, out_slice, sub_weights in state.blocks:
                xp.matmul(data[..., in_slice], sub_weights, out=result[..., out_slice])
        else:
            result = xp.concat(
                [xp.matmul(data[..., in_slice], sub_weights) for in_slice, _, sub_weights in state.blocks],
                axis=-1,
            )

        if state.out_inv_perm is not None:
            result = xp.take(result, state.out_inv_perm, axis=result.ndim - 1)
        if needs_permute:
            inv_dim_perm = list(range(result.ndim))
            inv_dim_perm.insert(axis_idx, inv_dim_perm.pop(-1))
            result = xp.permute_dims(result, inv_dim_perm)
        return result

    def _stacked_split(self, xp):
        """Split stacked ``A|B`` weights into ``(A, B)``, once per weight matrix.

        Both are views on the stored matrix, so this costs nothing to keep.
        """
        if self._state.stacked_split is None:
            weights = self._state.weights
            self._state.stacked_split = (weights[:-1], weights[-1:])
        return self._state.stacked_split

    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        data = message.data

        if self._state.blocks is not None:
            data = self._block_matmul(xp, data, axis_idx)
        else:
            # Weights stacked A|B express y = xA + B, where B is the last row and
            # the input is notionally augmented with a column of ones. This
            # supports NeuroKey's weights matrices.
            stacked = data.shape[axis_idx] == (self._state.weights.shape[0] - 1)

            needs_permute = axis_idx not in (-1, data.ndim - 1)
            if needs_permute:
                perm = list(range(data.ndim))
                perm.append(perm.pop(axis_idx))
                data = xp.permute_dims(data, perm)

            if stacked:
                a, b = self._stacked_split(xp)
                data = _matmul_add(xp, data, a, b)
            else:
                data = xp.matmul(data, self._state.weights)

            if needs_permute:
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
    weights: np.ndarray | str | Path | RereferenceKind | Callable[[int], np.ndarray],
    axis: str | None = None,
    right_multiply: bool = True,
    channel_groups: ChannelGroupSpec | None = None,
    kernel: str = "auto",
) -> AffineTransformTransformer:
    """
    Perform affine transformations on streaming data.

    Args:
        weights: An array of weights, a path to a file with weights compatible with np.loadtxt,
            a :class:`~ezmsg.sigproc.util.rereference.RereferenceKind` (or its string value),
            or a callable that accepts ``n_in`` and returns an ndarray of shape ``(n_in, n_out)``.
            See :attr:`AffineTransformSettings.weights`; for streaming CAR prefer
            :func:`common_rereference`.
        axis: The name of the axis to apply the transformation to. Defaults to the leading (0th) axis in the array.
        right_multiply: Set False to transpose the weights before applying.
        channel_groups: Channel grouping used to build kind- or callable-based weights. See
            :attr:`AffineTransformSettings.channel_groups`.
        kernel: Matmul kernel selection. See :attr:`AffineTransformSettings.kernel`.

    Returns:
        :obj:`AffineTransformTransformer`.
    """
    return AffineTransformTransformer(
        AffineTransformSettings(
            weights=weights,
            axis=axis,
            right_multiply=right_multiply,
            channel_groups=channel_groups,
            kernel=kernel,
        )
    )


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

    channel_groups: ChannelGroupSpec | None = None
    """Which channels share a reference. The common reference is computed
    independently within each group; channels in no group pass through unchanged.

    ``None`` (default) references every channel against one common average.
    Pass explicit index groups, or the name of a channel-metadata field to group
    by -- ``channel_groups="bank"`` rereferences within each electrode bank. See
    :data:`~ezmsg.sigproc.util.channels.ChannelGroupSpec`.

    A field-derived grouping is resolved when the stream gains or loses that
    field, not when its values change: the channel-to-field map is expected to be
    static for a given stream key and channel count."""


@processor_state
class CommonRereferenceState:
    single: bool = False
    """Whether one reference covers every channel -- the no-allocation fast path."""
    passthrough: bool = False
    """Leave-one-out with nothing to reference against; emit the input unchanged."""
    project: npt.NDArray | None = None
    """(n_ch, n_groups) group-mean projector; None on the ``single`` path."""
    spread: npt.NDArray | None = None
    """(n_groups, n_ch) indicator that broadcasts each group's reference back."""
    scale: npt.NDArray | float = 1.0
    """Per-channel N/(N-1) leave-one-out gain, or 1.0 when it is uniform."""
    groups: list | None = None
    """Per-group index arrays; used only by the ``median`` path."""
    out_dtype: npt.DTypeLike | None = None


class CommonRereferenceTransformer(
    BaseStatefulTransformer[CommonRereferenceSettings, AxisArray, AxisArray, CommonRereferenceState]
):
    """Subtract a common reference, computed over all channels or within groups.

    ``mode="mean"`` is expressed as two skinny matmuls rather than a per-group
    gather/scatter loop, so cost is independent of whether a group's channels are
    contiguous and there is no Python loop per message.

    Channels belonging to no group pass through unchanged, matching
    :func:`~ezmsg.sigproc.util.rereference.car_matrix`, which leaves them identity.

    Floating-point input keeps its dtype; integer input promotes to float. (An
    earlier version promoted float32 to float64, doubling the bandwidth of every
    downstream stage.)
    """

    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        return hash(
            (message.key, message.data.shape[axis_idx])
            + group_spec_fingerprint(message, axis, self.settings.channel_groups)
        )

    def _reset_state(self, message: AxisArray) -> None:
        xp = get_namespace(message.data)
        dev = array_device(message.data)
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        n_ch = message.data.shape[axis_idx]
        include_current = self.settings.include_current

        msg_dt = message.data.dtype
        out_dt = msg_dt if is_float_dtype(xp, msg_dt) else getattr(xp, "float64", None) or xp.float32
        self._state.out_dtype = out_dt

        groups = resolve_channel_groups(message, axis, self.settings.channel_groups)
        if groups is None:
            groups = [np.arange(n_ch, dtype=np.intp)]
        # A lone channel has no "other" channels to form a leave-one-out
        # reference from, so it passes through rather than dividing by N - 1 == 0.
        groups = [g for g in groups if g.size >= (1 if include_current else 2)]

        self._state.groups = [xp_asarray(xp, g, device=dev) for g in groups]
        self._state.single = len(groups) == 1 and groups[0].size == n_ch
        self._state.passthrough = not groups
        self._state.project = None
        self._state.spread = None
        self._state.scale = 1.0

        if self._state.passthrough:
            return

        sizes = np.array([g.size for g in groups], dtype=np.float64)
        if not include_current:
            scale = np.ones(n_ch, dtype=np.float64)
            for group, size in zip(groups, sizes):
                scale[group] = size / (size - 1.0)
            uniform = float(scale[groups[0][0]])
            self._state.scale = uniform if np.all(scale == uniform) else xp_asarray(xp, scale, dtype=out_dt, device=dev)

        if self._state.single:
            return

        project = np.zeros((n_ch, len(groups)), dtype=np.float64)
        spread = np.zeros((len(groups), n_ch), dtype=np.float64)
        for g, (group, size) in enumerate(zip(groups, sizes)):
            project[group, g] = 1.0 / size
            spread[g, group] = 1.0
        self._state.project = xp_asarray(xp, project, dtype=out_dt, device=dev)
        self._state.spread = xp_asarray(xp, spread, dtype=out_dt, device=dev)

    def _process(self, message: AxisArray) -> AxisArray:
        if self.settings.mode == "passthrough" or self._state.passthrough:
            return message

        xp = get_namespace(message.data)
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        state = self._state
        data = message.data

        if self.settings.mode == "median":
            return replace(message, data=self._median_rereference(xp, data, axis_idx))

        if state.single:
            # Only ever one group here, so the leave-one-out gain is a scalar.
            output = data - xp.mean(data, axis=axis_idx, keepdims=True)
            if state.scale != 1.0:
                output = output * state.scale
            return replace(message, data=output)

        # Grouped: reference = (x @ project) @ spread gives every channel its
        # group's mean (and zero for ungrouped channels) without any gather.
        needs_permute = axis_idx not in (-1, data.ndim - 1)
        if needs_permute:
            dim_perm = list(range(data.ndim))
            dim_perm.append(dim_perm.pop(axis_idx))
            data = xp.permute_dims(data, dim_perm)
        output = data - xp.matmul(xp.matmul(data, state.project), state.spread)
        # Channels are last here, so a per-channel gain broadcasts as-is.
        if isinstance(state.scale, float):
            if state.scale != 1.0:
                output = output * state.scale
        else:
            output = output * state.scale
        if needs_permute:
            inv_dim_perm = list(range(output.ndim))
            inv_dim_perm.insert(axis_idx, inv_dim_perm.pop(-1))
            output = xp.permute_dims(output, inv_dim_perm)
        return replace(message, data=output)

    def _median_rereference(self, xp, data, axis_idx):
        """Per-group median reference.

        Unlike the mean, a median is not a linear functional of the channels, so
        it has no matmul form -- this stays a gather/scatter loop. The output
        starts as a copy of the input so ungrouped channels pass through.

        This is the only path here that writes into an array elementwise, so the
        copy has to be a real one. ``asarray`` is not enough: given an
        ``ndarray`` *subclass* it returns a distinct base-class object that still
        shares the caller's buffer, and message data may be a view shared with
        other branches of the graph. Branching on dtype avoids the guesswork --
        a dtype change always allocates, and otherwise we copy outright.
        """
        median = getattr(xp, "median", np.median)
        scale = self._state.scale
        out_dtype = self._state.out_dtype
        output = xp_copy(data) if data.dtype == out_dtype else xp_asarray(xp, data, dtype=out_dtype)
        index: list = [slice(None)] * data.ndim
        bcast: list = [1] * data.ndim
        for group in self._state.groups:
            index[axis_idx] = group
            group_data = xp.take(data, group, axis=axis_idx)
            centered = group_data - median(group_data, axis=axis_idx, keepdims=True)
            if isinstance(scale, float):
                if scale != 1.0:
                    centered = centered * scale
            else:
                bcast[axis_idx] = group.shape[0]
                centered = centered * xp.reshape(xp.take(scale, group, axis=0), tuple(bcast))
            output[tuple(index)] = centered
        return output


class CommonRereference(
    BaseTransformerUnit[CommonRereferenceSettings, AxisArray, AxisArray, CommonRereferenceTransformer]
):
    SETTINGS = CommonRereferenceSettings


def common_rereference(
    mode: str = "mean",
    axis: str | None = None,
    include_current: bool = True,
    channel_groups: ChannelGroupSpec | None = None,
) -> CommonRereferenceTransformer:
    """
    Perform common average referencing (CAR) on streaming data.

    Args:
        mode: The statistical mode to apply -- either "mean" or "median"
        axis: The name of the axis to apply the transformation to.
        include_current: Set False to exclude each channel from participating in the calculation of its reference.
        channel_groups: Which channels share a reference -- explicit index groups or a
            channel-metadata field name (e.g. ``"bank"``). See
            :attr:`CommonRereferenceSettings.channel_groups`.

    Returns:
        :obj:`CommonRereferenceTransformer`
    """
    return CommonRereferenceTransformer(
        CommonRereferenceSettings(
            mode=mode,
            axis=axis,
            include_current=include_current,
            channel_groups=channel_groups,
        )
    )
