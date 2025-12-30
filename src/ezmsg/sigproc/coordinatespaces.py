"""
Coordinate space transformations for streaming data.

This module provides utilities and ezmsg nodes for transforming between
Cartesian (x, y) and polar (r, theta) coordinate systems.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

from enum import Enum
from typing import Tuple

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from array_api_compat import get_namespace, is_array_api_obj
from ezmsg.baseproc import (
    BaseTransformer,
    BaseTransformerUnit,
)
from ezmsg.util.messages.axisarray import AxisArray, replace

# -- Utility functions for coordinate transformations --


def _get_namespace_or_numpy(*args: npt.ArrayLike):
    """Get array namespace if any arg is an array, otherwise return numpy."""
    for arg in args:
        if is_array_api_obj(arg):
            return get_namespace(arg)
    return np


def polar2z(r: npt.ArrayLike, theta: npt.ArrayLike) -> npt.ArrayLike:
    """Convert polar coordinates to complex number representation."""
    xp = _get_namespace_or_numpy(r, theta)
    return r * xp.exp(1j * theta)


def z2polar(z: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Convert complex number to polar coordinates (r, theta)."""
    xp = _get_namespace_or_numpy(z)
    return xp.abs(z), xp.atan2(xp.imag(z), xp.real(z))


def cart2z(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.ArrayLike:
    """Convert Cartesian coordinates to complex number representation."""
    return x + 1j * y


def z2cart(z: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Convert complex number to Cartesian coordinates (x, y)."""
    xp = _get_namespace_or_numpy(z)
    return xp.real(z), xp.imag(z)


def cart2pol(x: npt.ArrayLike, y: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Convert Cartesian coordinates (x, y) to polar coordinates (r, theta)."""
    return z2polar(cart2z(x, y))


def pol2cart(r: npt.ArrayLike, theta: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)."""
    return z2cart(polar2z(r, theta))


# -- ezmsg transformer classes --


class CoordinateMode(str, Enum):
    """Transformation mode for coordinate conversion."""

    CART2POL = "cart2pol"
    """Convert Cartesian (x, y) to polar (r, theta)."""

    POL2CART = "pol2cart"
    """Convert polar (r, theta) to Cartesian (x, y)."""


class CoordinateSpacesSettings(ez.Settings):
    """
    Settings for :obj:`CoordinateSpaces`.

    See :obj:`coordinate_spaces` for argument details.
    """

    mode: CoordinateMode = CoordinateMode.CART2POL
    """The transformation mode: 'cart2pol' or 'pol2cart'."""

    axis: str | None = None
    """
    The name of the axis containing the coordinate components.
    Defaults to the last axis. Must have exactly 2 elements (x,y or r,theta).
    """


class CoordinateSpacesTransformer(BaseTransformer[CoordinateSpacesSettings, AxisArray, AxisArray]):
    """
    Transform between Cartesian and polar coordinate systems.

    The input must have exactly 2 elements along the specified axis:
    - For cart2pol: expects (x, y), outputs (r, theta)
    - For pol2cart: expects (r, theta), outputs (x, y)
    """

    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)

        if message.data.shape[axis_idx] != 2:
            raise ValueError(
                f"Coordinate transformation requires exactly 2 elements along axis '{axis}', "
                f"got {message.data.shape[axis_idx]}."
            )

        # Extract components along the specified axis
        slices_a = [slice(None)] * message.data.ndim
        slices_b = [slice(None)] * message.data.ndim
        slices_a[axis_idx] = 0
        slices_b[axis_idx] = 1

        component_a = message.data[tuple(slices_a)]
        component_b = message.data[tuple(slices_b)]

        if self.settings.mode == CoordinateMode.CART2POL:
            # Input: x, y -> Output: r, theta
            out_a, out_b = cart2pol(component_a, component_b)
        else:
            # Input: r, theta -> Output: x, y
            out_a, out_b = pol2cart(component_a, component_b)

        # Stack results back along the same axis
        result = xp.stack([out_a, out_b], axis=axis_idx)

        # Update axis labels if present (use numpy for string labels)
        axes = message.axes
        if axis in axes and hasattr(axes[axis], "data"):
            if self.settings.mode == CoordinateMode.CART2POL:
                new_labels = np.array(["r", "theta"])
            else:
                new_labels = np.array(["x", "y"])
            axes = {**axes, axis: replace(axes[axis], data=new_labels)}

        return replace(message, data=result, axes=axes)


class CoordinateSpaces(
    BaseTransformerUnit[CoordinateSpacesSettings, AxisArray, AxisArray, CoordinateSpacesTransformer]
):
    """
    Unit for transforming between Cartesian and polar coordinate systems.

    See :obj:`CoordinateSpacesSettings` for configuration options.
    """

    SETTINGS = CoordinateSpacesSettings
