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
    #  are handled specially in `aggregate_slices`.
    AggregationFunction.TRAPEZOID: np.trapezoid,
}

# Operations that cannot be evaluated from the values alone.
_NEEDS_COORDINATES = frozenset(
    {
        AggregationFunction.TRAPEZOID,
        AggregationFunction.ARGMIN,
        AggregationFunction.ARGMAX,
    }
)


def needs_coordinates(operation: typing.Union["AggregationFunction", typing.Iterable["AggregationFunction"]]) -> bool:
    """Whether ``operation`` requires the axis's x-coordinates.

    Lets a caller skip building a coordinate vector it will not use, which is
    worth doing where that vector would have to be constructed per chunk.
    Accepts a single function or an iterable of them.
    """
    if isinstance(operation, AggregationFunction):
        return operation in _NEEDS_COORDINATES
    return any(op in _NEEDS_COORDINATES for op in operation)


def axis_coordinates(message: AxisArray, axis_name: str) -> npt.NDArray:
    """The coordinate value of every element along ``axis_name``.

    Reads a coordinate axis's values directly; evaluates a linear axis over its
    own length. Callers whose data does not line up with the message -- anything
    carrying samples across message boundaries -- must build their own vector
    instead.
    """
    target_axis = message.get_axis(axis_name)
    if hasattr(target_axis, "data"):
        return np.asarray(target_axis.data)
    axis_idx = message.get_axis_idx(axis_name)
    return target_axis.value(np.arange(message.data.shape[axis_idx]))


def _apply_one(xp, op: AggregationFunction, segment, axis_idx: int):
    """One aggregation over one already-sliced segment."""
    func_name = op.value
    if hasattr(xp, func_name):
        return getattr(xp, func_name)(segment, axis=axis_idx)
    # nan-variants and friends are not in the Array API standard.
    result = AGGREGATORS[op](np.asarray(segment), axis=axis_idx)
    return xp.asarray(result) if xp is not np else result


def aggregate_slices(
    data,
    slices: typing.Sequence[slice],
    axis_idx: int,
    operation: AggregationFunction,
    *,
    coordinates: typing.Optional[npt.NDArray] = None,
    index_to_coordinate: bool = True,
):
    """Apply ``operation`` to each slice of ``data`` along ``axis_idx``, stacked.

    Where the groups come from is the caller's business -- coordinate bands
    resolved once, or bin boundaries recomputed per chunk -- but *running* the
    aggregation is the same either way, and three operations need more than
    ``f(segment, axis)`` to do it correctly:

    * the nan-variants, which the Array API does not define, so they need a
      numpy fallback and a conversion back to the caller's namespace;
    * ``TRAPEZOID``, which integrates and so needs the axis's x-coordinates, or
      it silently returns an integral in units of *samples* rather than of the
      axis;
    * ``ARGMIN``/``ARGMAX``, which return a position within the slice. An index
      is rarely what anyone wants -- "the peak is at 10.5 Hz" is useful, "the
      peak is at offset 7 of this band" is not -- so it is converted back to the
      axis coordinate here.

    The result has the same rank as ``data``, with ``axis_idx`` reduced to one
    entry per slice, so a caller can drop it into the message it came from
    having replaced only that axis.

    :param data: The array to slice. Any Array API namespace.
    :param slices: One slice per output group, in output order. Slices index
        ``data`` along ``axis_idx``; they need not be contiguous, and an empty
        sequence yields a zero-length result.
    :param axis_idx: Index of the axis being grouped.
    :param operation: The :obj:`AggregationFunction` to apply within each slice.
    :param coordinates: The axis's coordinate values, one per element of ``data``
        along ``axis_idx``. Required when :func:`needs_coordinates` is True for
        ``operation``, ignored otherwise.
    :param index_to_coordinate: Whether ``ARGMIN``/``ARGMAX`` results are
        converted from a within-slice index to an axis coordinate. False leaves
        the raw index, which :obj:`AggregateTransformer` relies on; it needs no
        ``coordinates``.

    :raises ValueError: if ``operation`` needs coordinates and none were given,
        or if ``coordinates`` does not span ``data`` along ``axis_idx``. Either
        would otherwise produce a plausible-looking but wrong number.
    """
    xp = get_namespace(data)
    is_arg = operation in (AggregationFunction.ARGMIN, AggregationFunction.ARGMAX)
    uses_coordinates = operation == AggregationFunction.TRAPEZOID or (is_arg and index_to_coordinate)

    if uses_coordinates:
        if coordinates is None:
            raise ValueError(f"AggregationFunction.{operation.name} needs the axis coordinates; pass coordinates=...")
        coordinates = np.asarray(coordinates)
        n = data.shape[axis_idx]
        if coordinates.shape[0] != n:
            raise ValueError(f"coordinates has {coordinates.shape[0]} entries but data spans {n} along axis {axis_idx}")

    if not len(slices):
        return slice_along_axis(data, slice(0, 0), axis=axis_idx)

    if operation == AggregationFunction.TRAPEZOID:
        # No Array API equivalent, and it needs x, so this path is numpy-only.
        np_data = np.asarray(data)
        stacked = np.stack(
            [
                np.trapezoid(slice_along_axis(np_data, sl, axis=axis_idx), x=coordinates[sl], axis=axis_idx)
                for sl in slices
            ],
            axis=axis_idx,
        )
        return xp.asarray(stacked) if xp is not np else stacked

    stacked = xp.stack(
        [_apply_one(xp, operation, slice_along_axis(data, sl, axis=axis_idx), axis_idx) for sl in slices],
        axis=axis_idx,
    )

    if is_arg and index_to_coordinate:
        # Indices are relative to each slice, so each is looked up in that
        # slice's own coordinates rather than in the whole vector.
        out = [
            coordinates[sl][np.asarray(slice_along_axis(stacked, sl_ix, axis=axis_idx))]
            for sl_ix, sl in enumerate(slices)
        ]
        stacked = np.stack(out, axis=axis_idx)
        return xp.asarray(stacked) if xp is not np else stacked

    return stacked


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
            self._state.ax_vec = np.array(target_axis.data)
        else:
            self._state.ax_vec = target_axis.value(np.arange(message.data.shape[ax_idx]))

        ax_dat = []
        slices = []
        for start, stop in self.settings.bands:
            inds = np.where(np.logical_and(self._state.ax_vec >= start, self._state.ax_vec <= stop))[0]
            slices.append(slice(int(inds[0]), int(inds[-1]) + 1))
            if hasattr(target_axis, "data"):
                if self._state.ax_vec.dtype.type is np.str_:
                    sl_dat = f"{self._state.ax_vec[inds[0]]} - {self._state.ax_vec[inds[-1]]}"
                else:
                    sl_dat = np.mean(self._state.ax_vec[inds])
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

        # The bands are already resolved to slices in _reset_state, and ax_vec
        # spans the whole axis, so both are handed over as-is.
        stacked = aggregate_slices(
            message.data,
            self._state.slices,
            ax_idx,
            self.settings.operation,
            coordinates=self._state.ax_vec,
        )

        return replace(
            message,
            data=stacked,
            axes={**message.axes, axis: self._state.out_axis},
        )


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
        axis_idx = message.get_axis_idx(self.settings.axis)
        op = self.settings.operation

        if op == AggregationFunction.NONE:
            raise ValueError("AggregationFunction.NONE is not supported for full-axis aggregation")

        # A whole-axis reduction is the one-slice case, so it shares the runner
        # -- which is what gets TRAPEZOID its x-coordinates here.
        #
        # index_to_coordinate=False keeps ARGMIN/ARGMAX returning a raw index.
        # Unlike the other two aggregators that is a *tested* contract here
        # rather than an oversight, so it is left alone; converting would be more
        # consistent (and arguably more useful, since this drops the axis the
        # index refers to) but it is a separate decision.
        needs_x = op == AggregationFunction.TRAPEZOID
        coordinates = axis_coordinates(message, self.settings.axis) if needs_x else None
        agg_data = aggregate_slices(
            message.data,
            [slice(None)],
            axis_idx,
            op,
            coordinates=coordinates,
            index_to_coordinate=False,
        )
        # One slice in, one entry out: drop the now-degenerate axis.
        agg_data = slice_along_axis(agg_data, 0, axis=axis_idx)

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
