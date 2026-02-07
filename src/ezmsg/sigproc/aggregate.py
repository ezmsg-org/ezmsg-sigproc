"""
Aggregation operations over arrays.

.. note::
    :obj:`AggregateTransformer` and :obj:`RangedAggregateTransformer` support the
    :doc:`Array API standard </guides/explanations/array_api>`, enabling use with
    NumPy, CuPy, PyTorch, and other compatible array libraries.
    Operations not available on a given backend (nan-variants, trapezoid) fall back
    to NumPy automatically.
"""

import typing

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
from ezmsg.util.messages.axisarray import (
    AxisArray,
    AxisBase,
    replace,
    slice_along_axis,
)

from .spectral import OptionsEnum


class AggregationFunction(OptionsEnum):
    """Enum for aggregation functions available to be used in :obj:`ranged_aggregate` operation."""

    NONE = "None (all)"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    SUM = "sum"
    NANMAX = "nanmax"
    NANMIN = "nanmin"
    NANMEAN = "nanmean"
    NANMEDIAN = "nanmedian"
    NANSTD = "nanstd"
    NANSUM = "nansum"
    ARGMIN = "argmin"
    ARGMAX = "argmax"
    TRAPEZOID = "trapezoid"


AGGREGATORS = {
    AggregationFunction.NONE: np.all,
    AggregationFunction.MAX: np.max,
    AggregationFunction.MIN: np.min,
    AggregationFunction.MEAN: np.mean,
    AggregationFunction.MEDIAN: np.median,
    AggregationFunction.STD: np.std,
    AggregationFunction.SUM: np.sum,
    AggregationFunction.NANMAX: np.nanmax,
    AggregationFunction.NANMIN: np.nanmin,
    AggregationFunction.NANMEAN: np.nanmean,
    AggregationFunction.NANMEDIAN: np.nanmedian,
    AggregationFunction.NANSTD: np.nanstd,
    AggregationFunction.NANSUM: np.nansum,
    AggregationFunction.ARGMIN: np.argmin,
    AggregationFunction.ARGMAX: np.argmax,
    # Note: Some methods require x-coordinates and
    #  are handled specially in `_process`.
    AggregationFunction.TRAPEZOID: np.trapezoid,
}


class RangedAggregateSettings(ez.Settings):
    """
    Settings for ``RangedAggregate``.
    """

    axis: str | None = None
    """The name of the axis along which to apply the bands."""

    bands: list[tuple[float, float]] | None = None
    """
    [(band1_min, band1_max), (band2_min, band2_max), ...]
    If not set then this acts as a passthrough node.
    """

    operation: AggregationFunction = AggregationFunction.MEAN
    """:obj:`AggregationFunction` to apply to each band."""


@processor_state
class RangedAggregateState:
    slices: list[tuple[typing.Any, ...]] | None = None
    out_axis: AxisBase | None = None
    ax_vec: npt.NDArray | None = None


class RangedAggregateTransformer(
    BaseStatefulTransformer[RangedAggregateSettings, AxisArray, AxisArray, RangedAggregateState]
):
    def __call__(self, message: AxisArray) -> AxisArray:
        # Override for shortcut passthrough mode.
        if self.settings.bands is None:
            return message
        return super().__call__(message)

    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[0]
        target_axis = message.get_axis(axis)

        hash_components = (message.key,)
        if hasattr(target_axis, "data"):
            hash_components += (len(target_axis.data),)
        elif isinstance(target_axis, AxisArray.LinearAxis):
            hash_components += (target_axis.gain, target_axis.offset)
        return hash(hash_components)

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[0]
        target_axis = message.get_axis(axis)
        ax_idx = message.get_axis_idx(axis)

        if hasattr(target_axis, "data"):
            self._state.ax_vec = target_axis.data
        else:
            self._state.ax_vec = target_axis.value(np.arange(message.data.shape[ax_idx]))

        ax_dat = []
        slices = []
        for start, stop in self.settings.bands:
            inds = np.where(np.logical_and(self._state.ax_vec >= start, self._state.ax_vec <= stop))[0]
            slices.append(slice(int(inds[0]), int(inds[-1]) + 1))
            if hasattr(target_axis, "data"):
                if self._state.ax_vec.dtype.type is np.str_:
                    sl_dat = f"{self._state.ax_vec[start]} - {self._state.ax_vec[stop]}"
                else:
                    ax_dat.append(np.mean(self._state.ax_vec[inds]))
            else:
                sl_dat = target_axis.value(np.mean(inds))
            ax_dat.append(sl_dat)

        self._state.slices = slices
        self._state.out_axis = AxisArray.CoordinateAxis(
            data=np.array(ax_dat),
            dims=[axis],
            unit=target_axis.unit,
        )

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis or message.dims[0]
        ax_idx = message.get_axis_idx(axis)
        xp = get_namespace(message.data)
        op = self.settings.operation

        if op == AggregationFunction.TRAPEZOID:
            # Trapezoid requires x-coordinates and has no Array API equivalent;
            # fall back to numpy.
            np_data = np.asarray(message.data)
            out_data = [
                np.trapezoid(
                    slice_along_axis(np_data, sl, axis=ax_idx),
                    x=self._state.ax_vec[sl],
                    axis=ax_idx,
                )
                for sl in self._state.slices
            ]
            stacked = np.stack(out_data, axis=ax_idx)
            # Convert back to original backend if needed
            if xp is not np:
                stacked = xp.asarray(stacked)
        else:
            # Use Array API function when available, fall back to numpy AGGREGATORS
            func_name = op.value
            if hasattr(xp, func_name):
                agg_func = getattr(xp, func_name)
                out_data = [
                    agg_func(slice_along_axis(message.data, sl, axis=ax_idx), axis=ax_idx) for sl in self._state.slices
                ]
                stacked = xp.stack(out_data, axis=ax_idx)
            else:
                # nan-variants etc. — fall back to numpy
                np_agg = AGGREGATORS[op]
                np_data = np.asarray(message.data)
                out_data = [
                    np_agg(slice_along_axis(np_data, sl, axis=ax_idx), axis=ax_idx) for sl in self._state.slices
                ]
                stacked = np.stack(out_data, axis=ax_idx)
                if xp is not np:
                    stacked = xp.asarray(stacked)

        msg_out = replace(
            message,
            data=stacked,
            axes={**message.axes, axis: self._state.out_axis},
        )

        if op in (AggregationFunction.ARGMIN, AggregationFunction.ARGMAX):
            # Post-process: convert indices to axis coordinate values.
            # ax_vec is always numpy; offsets must be numpy for fancy indexing.
            out_data = []
            for sl_ix, sl in enumerate(self._state.slices):
                offsets = np.asarray(slice_along_axis(msg_out.data, sl_ix, axis=ax_idx))
                out_data.append(self._state.ax_vec[sl][offsets])
            msg_out.data = np.stack(out_data, axis=ax_idx)

        return msg_out


class RangedAggregate(BaseTransformerUnit[RangedAggregateSettings, AxisArray, AxisArray, RangedAggregateTransformer]):
    SETTINGS = RangedAggregateSettings


def ranged_aggregate(
    axis: str | None = None,
    bands: list[tuple[float, float]] | None = None,
    operation: AggregationFunction = AggregationFunction.MEAN,
) -> RangedAggregateTransformer:
    """
    Apply an aggregation operation over one or more bands.

    Args:
        axis: The name of the axis along which to apply the bands.
        bands: [(band1_min, band1_max), (band2_min, band2_max), ...]
            If not set then this acts as a passthrough node.
        operation: :obj:`AggregationFunction` to apply to each band.

    Returns:
        :obj:`RangedAggregateTransformer`
    """
    return RangedAggregateTransformer(RangedAggregateSettings(axis=axis, bands=bands, operation=operation))


class AggregateSettings(ez.Settings):
    """Settings for :obj:`Aggregate`."""

    axis: str
    """The name of the axis to aggregate over. This axis will be removed from the output."""

    operation: AggregationFunction = AggregationFunction.MEAN
    """:obj:`AggregationFunction` to apply."""


class AggregateTransformer(BaseTransformer[AggregateSettings, AxisArray, AxisArray]):
    """
    Transformer that aggregates an entire axis using a specified operation.

    Unlike :obj:`RangedAggregateTransformer` which aggregates over specific ranges/bands
    and preserves the axis (with one value per band), this transformer aggregates the
    entire axis and removes it from the output, reducing dimensionality by one.
    """

    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        axis_idx = message.get_axis_idx(self.settings.axis)
        op = self.settings.operation

        if op == AggregationFunction.NONE:
            raise ValueError("AggregationFunction.NONE is not supported for full-axis aggregation")

        if op == AggregationFunction.TRAPEZOID:
            # Trapezoid integration requires x-coordinates
            target_axis = message.get_axis(self.settings.axis)
            if hasattr(target_axis, "data"):
                x = target_axis.data
            else:
                x = target_axis.value(np.arange(message.data.shape[axis_idx]))
            agg_data = np.trapezoid(np.asarray(message.data), x=x, axis=axis_idx)
        else:
            # Try array-API compatible function first, fall back to numpy
            func_name = op.value
            if hasattr(xp, func_name):
                agg_data = getattr(xp, func_name)(message.data, axis=axis_idx)
            else:
                agg_data = AGGREGATORS[op](message.data, axis=axis_idx)

        new_dims = list(message.dims)
        new_dims.pop(axis_idx)

        new_axes = dict(message.axes)
        new_axes.pop(self.settings.axis, None)

        return replace(
            message,
            data=agg_data,
            dims=new_dims,
            axes=new_axes,
        )


class AggregateUnit(BaseTransformerUnit[AggregateSettings, AxisArray, AxisArray, AggregateTransformer]):
    """Unit that aggregates an entire axis using a specified operation."""

    SETTINGS = AggregateSettings
