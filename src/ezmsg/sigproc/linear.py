"""
Apply a linear transformation: output = scale * input + offset.

Supports per-element scale and offset along a specified axis.
For full matrix transformations, use :obj:`AffineTransformTransformer` instead.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from array_api_compat import get_namespace
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class LinearTransformSettings(ez.Settings):
    scale: float | list[float] | npt.ArrayLike = 1.0
    """Scale factor(s). Can be a scalar (applied to all elements) or an array
    matching the size of the specified axis for per-element scaling."""

    offset: float | list[float] | npt.ArrayLike = 0.0
    """Offset value(s). Can be a scalar (applied to all elements) or an array
    matching the size of the specified axis for per-element offset."""

    axis: str | None = None
    """Axis along which to apply per-element scale/offset. If None, scalar
    scale/offset are broadcast to all elements."""


@processor_state
class LinearTransformState:
    scale: npt.NDArray = None
    """Prepared scale array for broadcasting."""

    offset: npt.NDArray = None
    """Prepared offset array for broadcasting."""


class LinearTransformTransformer(
    BaseStatefulTransformer[LinearTransformSettings, AxisArray, AxisArray, LinearTransformState]
):
    """Apply linear transformation: output = scale * input + offset.

    This transformer is optimized for element-wise linear operations with
    optional per-channel (or per-axis) coefficients. For full matrix
    transformations, use :obj:`AffineTransformTransformer` instead.

    Examples:
        # Uniform scaling and offset
        >>> transformer = LinearTransformTransformer(LinearTransformSettings(scale=2.0, offset=1.0))

        # Per-channel scaling (e.g., for 3-channel data along "ch" axis)
        >>> transformer = LinearTransformTransformer(LinearTransformSettings(
        ...     scale=[0.5, 1.0, 2.0],
        ...     offset=[0.0, 0.1, 0.2],
        ...     axis="ch"
        ... ))
    """

    def _hash_message(self, message: AxisArray) -> int:
        """Hash based on shape and axis to detect when broadcast shapes need recalculation."""
        axis = self.settings.axis
        if axis is not None:
            axis_idx = message.get_axis_idx(axis)
            return hash((message.data.ndim, axis_idx, message.data.shape[axis_idx]))
        return hash(message.data.ndim)

    def _reset_state(self, message: AxisArray) -> None:
        """Prepare scale/offset arrays with proper broadcast shapes."""
        xp = get_namespace(message.data)
        ndim = message.data.ndim

        scale = self.settings.scale
        offset = self.settings.offset

        # Convert settings to arrays
        if isinstance(scale, (list, np.ndarray)):
            scale = xp.asarray(scale, dtype=xp.float64)
        else:
            # Scalar: create a 0-d array
            scale = xp.asarray(float(scale), dtype=xp.float64)

        if isinstance(offset, (list, np.ndarray)):
            offset = xp.asarray(offset, dtype=xp.float64)
        else:
            # Scalar: create a 0-d array
            offset = xp.asarray(float(offset), dtype=xp.float64)

        # If axis is specified and we have 1-d arrays, reshape for proper broadcasting
        if self.settings.axis is not None and ndim > 0:
            axis_idx = message.get_axis_idx(self.settings.axis)

            if scale.ndim == 1:
                # Create shape for broadcasting: all 1s except at axis_idx
                broadcast_shape = [1] * ndim
                broadcast_shape[axis_idx] = scale.shape[0]
                scale = xp.reshape(scale, broadcast_shape)

            if offset.ndim == 1:
                broadcast_shape = [1] * ndim
                broadcast_shape[axis_idx] = offset.shape[0]
                offset = xp.reshape(offset, broadcast_shape)

        self._state.scale = scale
        self._state.offset = offset

    def _process(self, message: AxisArray) -> AxisArray:
        result = message.data * self._state.scale + self._state.offset
        return replace(message, data=result)


class LinearTransform(BaseTransformerUnit[LinearTransformSettings, AxisArray, AxisArray, LinearTransformTransformer]):
    """Unit wrapper for LinearTransformTransformer."""

    SETTINGS = LinearTransformSettings
