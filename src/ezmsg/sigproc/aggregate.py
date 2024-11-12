import typing

import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
    AxisBase,
    replace,
)

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
    SUM = "sum"
    NANMAX = "nanmax"
    NANMIN = "nanmin"
    NANMEAN = "nanmean"
    NANMEDIAN = "nanmedian"
    NANSTD = "nanstd"
    NANSUM = "nansum"
    ARGMIN = "argmin"
    ARGMAX = "argmax"


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
        A primed generator object ready to yield an :obj:`AxisArray` for each .send(axis_array)
    """
    msg_out = AxisArray(np.array([]), dims=[""])

    # State variables
    slices: typing.Optional[typing.List[typing.Tuple[typing.Any, ...]]] = None
    out_axis: typing.Optional[AxisBase] = None
    ax_vec: typing.Optional[npt.NDArray] = None

    # Reset if any of these changes. Key not checked because continuity between chunks not required.
    check_inputs = {"gain": None, "offset": None, "len": None, "key": None}

    while True:
        msg_in: AxisArray = yield msg_out
        if bands is None:
            msg_out = msg_in
        else:
            axis = axis or msg_in.dims[0]
            target_axis = msg_in.get_axis(axis)

            # Check if we need to reset state
            b_reset = msg_in.key != check_inputs["key"]
            if hasattr(target_axis, "data"):
                b_reset = b_reset or len(target_axis.data) != check_inputs["len"]
            elif isinstance(target_axis, AxisArray.LinearAxis):
                b_reset = b_reset or target_axis.gain != check_inputs["gain"]
                b_reset = b_reset or target_axis.offset != check_inputs["offset"]

            if b_reset:
                # Update check variables
                check_inputs["key"] = msg_in.key
                if hasattr(target_axis, "data"):
                    check_inputs["len"] = len(target_axis.data)
                else:
                    check_inputs["gain"] = target_axis.gain
                    check_inputs["offset"] = target_axis.offset

                # If the axis we are operating on has changed (e.g., "time" or "win" axis always changes),
                #  or the key has changed, then recalculate slices.

                ax_idx = msg_in.get_axis_idx(axis)

                if hasattr(target_axis, "data"):
                    ax_vec = target_axis.data
                else:
                    ax_vec = target_axis.value(np.arange(msg_in.data.shape[ax_idx]))

                slices = []
                ax_dat = []
                for start, stop in bands:
                    inds = np.where(np.logical_and(ax_vec >= start, ax_vec <= stop))[0]
                    slices.append(np.s_[inds[0] : inds[-1] + 1])
                    if hasattr(target_axis, "data"):
                        if ax_vec.dtype.type is np.str_:
                            sl_dat = f"{ax_vec[start]} - {ax_vec[stop]}"
                        else:
                            sl_dat = ax_dat.append(np.mean(ax_vec[inds]))
                    else:
                        sl_dat = target_axis.value(np.mean(inds))
                    ax_dat.append(sl_dat)

                out_axis = AxisArray.CoordinateAxis(
                    data=np.array(ax_dat),
                    dims=[axis],
                    unit=target_axis.unit,
                )

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
