import os
from pathlib import Path
import typing

import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, AxisBase, replace
from ezmsg.util.generator import consumer

from .base import GenAxisArray


@consumer
def affine_transform(
    weights: typing.Union[np.ndarray, str, Path],
    axis: typing.Optional[str] = None,
    right_multiply: bool = True,
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Perform affine transformations on streaming data.

    Args:
        weights: An array of weights or a path to a file with weights compatible with np.loadtxt.
        axis: The name of the axis to apply the transformation to. Defaults to the leading (0th) axis in the array.
        right_multiply: Set False to transpose the weights before applying.

    Returns:
        A primed generator object that yields an :obj:`AxisArray` object for every
        :obj:`AxisArray` it receives via `send`.
    """
    msg_out = AxisArray(np.array([]), dims=[""])

    # Check parameters
    if isinstance(weights, str):
        if weights == "passthrough":
            weights = None
        else:
            weights = Path(os.path.abspath(os.path.expanduser(weights)))
    if isinstance(weights, Path):
        weights = np.loadtxt(weights, delimiter=",")
    if not right_multiply:
        weights = weights.T
    if weights is not None:
        weights = np.ascontiguousarray(weights)

    # State variables
    # New axis with transformed labels, if required
    new_axis: typing.Optional[AxisBase] = None

    # Reset if any of these change.
    check_input = {"key": None}
    # We assume key change catches labels change; we don't want to check labels every message
    # We don't need to check if input size has changed because weights multiplication will fail if so.

    while True:
        msg_in: AxisArray = yield msg_out

        if weights is None:
            msg_out = msg_in
            continue

        axis = axis or msg_in.dims[-1]  # Note: Most nodes default do dim[0]
        axis_idx = msg_in.get_axis_idx(axis)

        b_reset = msg_in.key != check_input["key"]
        if b_reset:
            # First sample or key has changed. Reset the state.
            check_input["key"] = msg_in.key
            # Determine if we need to modify the transformed axis.
            if (
                axis in msg_in.axes
                and hasattr(msg_in.axes[axis], "data")
                and weights.shape[0] != weights.shape[1]
            ):
                in_labels = msg_in.axes[axis].data
                new_labels = []
                n_in, n_out = weights.shape
                if len(in_labels) != n_in:
                    # Something upstream did something it wasn't supposed to. We will drop the labels.
                    ez.logger.warning(
                        f"Received {len(in_labels)} for {n_in} inputs. Check upstream labels."
                    )
                else:
                    b_filled_outputs = np.any(weights, axis=0)
                    b_used_inputs = np.any(weights, axis=1)
                    if np.all(b_used_inputs) and np.all(b_filled_outputs):
                        # All inputs are used and all outputs are used, but n_in != n_out.
                        # Mapping cannot be determined.
                        new_labels = []
                    elif np.all(b_used_inputs):
                        # Strange scenario: New outputs are filled with empty data.
                        in_ix = 0
                        new_labels = []
                        for out_ix in range(n_out):
                            if b_filled_outputs[out_ix]:
                                new_labels.append(in_labels[in_ix])
                                in_ix += 1
                            else:
                                new_labels.append("")
                    elif np.all(b_filled_outputs):
                        # Transform is dropping some of the inputs.
                        new_labels = np.array(in_labels)[b_used_inputs]
                new_axis = replace(msg_in.axes[axis], data=np.array(new_labels))

        data = msg_in.data

        if data.shape[axis_idx] == (weights.shape[0] - 1):
            # The weights are stacked A|B where A is the transform and B is a single row
            #  in the equation y = Ax + B. This supports NeuroKey's weights matrices.
            sample_shape = data.shape[:axis_idx] + (1,) + data.shape[axis_idx + 1 :]
            data = np.concatenate(
                (data, np.ones(sample_shape).astype(data.dtype)), axis=axis_idx
            )

        if axis_idx in [-1, len(msg_in.dims) - 1]:
            data = np.matmul(data, weights)
        else:
            data = np.moveaxis(data, axis_idx, -1)
            data = np.matmul(data, weights)
            data = np.moveaxis(data, -1, axis_idx)

        replace_kwargs = {"data": data}
        if new_axis is not None:
            replace_kwargs["axes"] = {**msg_in.axes, axis: new_axis}
        msg_out = replace(msg_in, **replace_kwargs)


class AffineTransformSettings(ez.Settings):
    """
    Settings for :obj:`AffineTransform`.
    See :obj:`affine_transform` for argument details.
    """

    weights: typing.Union[np.ndarray, str, Path]
    axis: typing.Optional[str] = None
    right_multiply: bool = True


class AffineTransform(GenAxisArray):
    """:obj:`Unit` for :obj:`affine_transform`"""

    SETTINGS = AffineTransformSettings

    def construct_generator(self):
        self.STATE.gen = affine_transform(
            weights=self.SETTINGS.weights,
            axis=self.SETTINGS.axis,
            right_multiply=self.SETTINGS.right_multiply,
        )


def zeros_for_noop(data: npt.NDArray, **ignore_kwargs) -> npt.NDArray:
    return np.zeros_like(data)


@consumer
def common_rereference(
    mode: str = "mean", axis: typing.Optional[str] = None, include_current: bool = True
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Perform common average referencing (CAR) on streaming data.

    Args:
        mode: The statistical mode to apply -- either "mean" or "median"
        axis: The name of hte axis to apply the transformation to.
        include_current: Set False to exclude each channel from participating in the calculation of its reference.

    Returns:
        A primed generator object that yields an :obj:`AxisArray` object
        for every :obj:`AxisArray` it receives via `send`.
    """
    msg_out = AxisArray(np.array([]), dims=[""])

    if mode == "passthrough":
        include_current = True

    func = {"mean": np.mean, "median": np.median, "passthrough": zeros_for_noop}[mode]

    while True:
        msg_in: AxisArray = yield msg_out

        if axis is None:
            axis = msg_in.dims[-1]
            axis_idx = -1
        else:
            axis_idx = msg_in.get_axis_idx(axis)

        ref_data = func(msg_in.data, axis=axis_idx, keepdims=True)

        if not include_current:
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
            N = msg_in.data.shape[axis_idx]
            ref_data = (N / (N - 1)) * ref_data - msg_in.data / (N - 1)
            # Side note: I profiled using affine_transform and it's about 30x slower than this implementation.

        msg_out = replace(msg_in, data=msg_in.data - ref_data)


class CommonRereferenceSettings(ez.Settings):
    """
    Settings for :obj:`CommonRereference`
    See :obj:`common_rereference` for argument details.
    """

    mode: str = "mean"
    axis: typing.Optional[str] = None
    include_current: bool = True


class CommonRereference(GenAxisArray):
    """
    :obj:`Unit` for :obj:`common_rereference`.
    """

    SETTINGS = CommonRereferenceSettings

    def construct_generator(self):
        self.STATE.gen = common_rereference(
            mode=self.SETTINGS.mode,
            axis=self.SETTINGS.axis,
            include_current=self.SETTINGS.include_current,
        )
