import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, AxisBase
from ezmsg.util.messages.util import replace

from .base import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    BaseTransformer,
    processor_state,
)


class AffineTransformSettings(ez.Settings):
    """
    Settings for :obj:`AffineTransform`.
    See :obj:`affine_transform` for argument details.
    """

    weights: np.ndarray | str | Path
    """An array of weights or a path to a file with weights compatible with np.loadtxt."""

    axis: str | None = None
    """The name of the axis to apply the transformation to. Defaults to the leading (0th) axis in the array."""

    right_multiply: bool = True
    """Set False to transpose the weights before applying."""


@processor_state
class AffineTransformState:
    weights: npt.NDArray | None = None
    new_axis: AxisBase | None = None


class AffineTransformTransformer(
    BaseStatefulTransformer[
        AffineTransformSettings, AxisArray, AxisArray, AffineTransformState
    ]
):
    def __call__(self, message: AxisArray) -> AxisArray:
        # Override __call__ so we can shortcut if weights are None.
        if self.settings.weights is None or (
            isinstance(self.settings.weights, str)
            and self.settings.weights == "passthrough"
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

        axis = self.settings.axis or message.dims[-1]
        if (
            axis in message.axes
            and hasattr(message.axes[axis], "data")
            and weights.shape[0] != weights.shape[1]
        ):
            in_labels = message.axes[axis].data
            new_labels = []
            n_in, n_out = weights.shape
            if len(in_labels) != n_in:
                ez.logger.warning(
                    f"Received {len(in_labels)} for {n_in} inputs. Check upstream labels."
                )
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

            self._state.new_axis = replace(
                message.axes[axis], data=np.array(new_labels)
            )

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        data = message.data

        if data.shape[axis_idx] == (self._state.weights.shape[0] - 1):
            # The weights are stacked A|B where A is the transform and B is a single row
            #  in the equation y = Ax + B. This supports NeuroKey's weights matrices.
            sample_shape = data.shape[:axis_idx] + (1,) + data.shape[axis_idx + 1 :]
            data = np.concatenate(
                (data, np.ones(sample_shape).astype(data.dtype)), axis=axis_idx
            )

        if axis_idx in [-1, len(message.dims) - 1]:
            data = np.matmul(data, self._state.weights)
        else:
            data = np.moveaxis(data, axis_idx, -1)
            data = np.matmul(data, self._state.weights)
            data = np.moveaxis(data, -1, axis_idx)

        replace_kwargs = {"data": data}
        if self._state.new_axis is not None:
            replace_kwargs["axes"] = {**message.axes, axis: self._state.new_axis}

        return replace(message, **replace_kwargs)


class AffineTransform(
    BaseTransformerUnit[
        AffineTransformSettings, AxisArray, AxisArray, AffineTransformTransformer
    ]
):
    SETTINGS = AffineTransformSettings


def affine_transform(
    weights: np.ndarray | str | Path,
    axis: str | None = None,
    right_multiply: bool = True,
) -> AffineTransformTransformer:
    """
    Perform affine transformations on streaming data.

    Args:
        weights: An array of weights or a path to a file with weights compatible with np.loadtxt.
        axis: The name of the axis to apply the transformation to. Defaults to the leading (0th) axis in the array.
        right_multiply: Set False to transpose the weights before applying.

    Returns:
        :obj:`AffineTransformTransformer`.
    """
    return AffineTransformTransformer(
        AffineTransformSettings(
            weights=weights, axis=axis, right_multiply=right_multiply
        )
    )


def zeros_for_noop(data: npt.NDArray, **ignore_kwargs) -> npt.NDArray:
    return np.zeros_like(data)


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


class CommonRereferenceTransformer(
    BaseTransformer[CommonRereferenceSettings, AxisArray, AxisArray]
):
    def _process(self, message: AxisArray) -> AxisArray:
        if self.settings.mode == "passthrough":
            return message

        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)

        func = {"mean": np.mean, "median": np.median, "passthrough": zeros_for_noop}[
            self.settings.mode
        ]

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
    BaseTransformerUnit[
        CommonRereferenceSettings, AxisArray, AxisArray, CommonRereferenceTransformer
    ]
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
