from dataclasses import replace
import typing

import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis

from .spectral import OptionsEnum
from .base import GenAxisArray


class AggregationFunction(OptionsEnum):
    """Enum for aggregation functions available to be used in :obj:`ranged_aggregate` operation."""

    NONE = "None (all)"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    NANMAX = "nanmax"
    NANMIN = "nanmin"
    NANMEAN = "nanmean"
    NANMEDIAN = "nanmedian"
    NANSTD = "nanstd"
    ARGMIN = "argmin"
    ARGMAX = "argmax"


AGGREGATORS = {
    AggregationFunction.NONE: np.all,
    AggregationFunction.MAX: np.max,
    AggregationFunction.MIN: np.min,
    AggregationFunction.MEAN: np.mean,
    AggregationFunction.MEDIAN: np.median,
    AggregationFunction.STD: np.std,
    AggregationFunction.NANMAX: np.nanmax,
    AggregationFunction.NANMIN: np.nanmin,
    AggregationFunction.NANMEAN: np.nanmean,
    AggregationFunction.NANMEDIAN: np.nanmedian,
    AggregationFunction.NANSTD: np.nanstd,
    AggregationFunction.ARGMIN: np.argmin,
    AggregationFunction.ARGMAX: np.argmax,
}


@consumer
def ranged_aggregate(
    axis: typing.Optional[str] = None,
    bands: typing.Optional[typing.List[typing.Tuple[float, float]]] = None,
    operation: AggregationFunction = AggregationFunction.MEAN,
):
    """
    Apply an aggregation operation over one or more bands.

    Args:
        axis: The name of the axis along which to apply the bands.
        bands: [(band1_min, band1_max), (band2_min, band2_max), ...]
            If not set then this acts as a passthrough node.
        operation: :obj:`AggregationFunction` to apply to each band.

    Returns:
        A primed generator object ready to yield an AxisArray for each .send(axis_array)
    """
    msg_out = AxisArray(np.array([]), dims=[""])

    # State variables
    slices: typing.Optional[typing.List[typing.Tuple[typing.Any, ...]]] = None
    out_axis: typing.Optional[AxisArray.Axis] = None
    ax_vec: typing.Optional[npt.NDArray] = None

    # Reset if any of these changes. Key not checked because continuity between chunks not required.
    check_inputs = {"gain": None, "offset": None}

    while True:
        msg_in: AxisArray = yield msg_out
        if bands is None:
            msg_out = msg_in
        else:
            axis = axis or msg_in.dims[0]
            target_axis = msg_in.get_axis(axis)

            b_reset = target_axis.gain != check_inputs["gain"]
            b_reset = b_reset or target_axis.offset != check_inputs["offset"]
            if b_reset:
                check_inputs["gain"] = target_axis.gain
                check_inputs["offset"] = target_axis.offset

                # If the axis we are operating on has changed (e.g., "time" or "win" axis always changes),
                #  or the key has changed, then recalculate slices.

                ax_idx = msg_in.get_axis_idx(axis)

                ax_vec = (
                    target_axis.offset
                    + np.arange(msg_in.data.shape[ax_idx]) * target_axis.gain
                )
                slices = []
                mids = []
                for start, stop in bands:
                    inds = np.where(np.logical_and(ax_vec >= start, ax_vec <= stop))[0]
                    mids.append(np.mean(inds) * target_axis.gain + target_axis.offset)
                    slices.append(np.s_[inds[0] : inds[-1] + 1])
                out_ax_kwargs = {
                    "unit": target_axis.unit,
                    "offset": mids[0],
                    "gain": (mids[1] - mids[0]) if len(mids) > 1 else 1.0,
                }
                if hasattr(target_axis, "labels"):
                    out_ax_kwargs["labels"] = [f"{_[0]} - {_[1]}" for _ in bands]
                out_axis = replace(target_axis, **out_ax_kwargs)

            agg_func = AGGREGATORS[operation]
            out_data = [
                agg_func(slice_along_axis(msg_in.data, sl, axis=ax_idx), axis=ax_idx)
                for sl in slices
            ]

            msg_out = replace(
                msg_in,
                data=np.stack(out_data, axis=ax_idx),
                axes={**msg_in.axes, axis: out_axis},
            )
            if operation in [AggregationFunction.ARGMIN, AggregationFunction.ARGMAX]:
                # Convert indices returned by argmin/argmax into the value along the axis.
                out_data = []
                for sl_ix, sl in enumerate(slices):
                    offsets = np.take(msg_out.data, [sl_ix], axis=ax_idx)
                    out_data.append(ax_vec[sl][offsets])
                msg_out.data = np.concatenate(out_data, axis=ax_idx)


class RangedAggregateSettings(ez.Settings):
    """
    Settings for ``RangedAggregate``.
    See :obj:`ranged_aggregate` for details.
    """

    axis: typing.Optional[str] = None
    bands: typing.Optional[typing.List[typing.Tuple[float, float]]] = None
    operation: AggregationFunction = AggregationFunction.MEAN


class RangedAggregate(GenAxisArray):
    """
    Unit for :obj:`ranged_aggregate`
    """

    SETTINGS = RangedAggregateSettings

    def construct_generator(self):
        self.STATE.gen = ranged_aggregate(
            axis=self.SETTINGS.axis,
            bands=self.SETTINGS.bands,
            operation=self.SETTINGS.operation,
        )
