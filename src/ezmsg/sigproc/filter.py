from dataclasses import dataclass, field
import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from ezmsg.util.generator import consumer
import numpy as np
import numpy.typing as npt
import scipy.signal

from ezmsg.sigproc.base import GenAxisArray


@dataclass
class FilterCoefficients:
    b: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    a: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))


def _normalize_coefs(
    coefs: FilterCoefficients | tuple[npt.NDArray, npt.NDArray] | npt.NDArray,
) -> tuple[str, tuple[npt.NDArray, ...]]:
    coef_type = "ba"
    if coefs is not None:
        # scipy.signal functions called with first arg `*coefs`.
        # Make sure we have a tuple of coefficients.
        if isinstance(coefs, npt.NDArray):
            coef_type = "sos"
            coefs = (coefs,)  # sos funcs just want a single ndarray.
        elif isinstance(coefs, FilterCoefficients):
            coefs = (FilterCoefficients.b, FilterCoefficients.a)
    return coef_type, coefs


@consumer
def filtergen(
    axis: str, coefs: npt.NDArray | tuple[npt.NDArray] | None, coef_type: str
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Filter data using the provided coefficients.

    Args:
        axis: The name of the axis to operate on.
        coefs: The pre-calculated filter coefficients.
        coef_type: The type of filter coefficients. One of "ba" or "sos".

    Returns:
        A primed generator that, when passed an :obj:`AxisArray` via `.send(axis_array)`,
         yields an :obj:`AxisArray` with the data filtered.
    """
    # Massage inputs
    if coefs is not None and not isinstance(coefs, tuple):
        # scipy.signal functions called with first arg `*coefs`, but sos coefs are a single ndarray.
        coefs = (coefs,)

    # Init IO
    msg_out = AxisArray(np.array([]), dims=[""])

    filt_func = {"ba": scipy.signal.lfilter, "sos": scipy.signal.sosfilt}[coef_type]
    zi_func = {"ba": scipy.signal.lfilter_zi, "sos": scipy.signal.sosfilt_zi}[coef_type]

    # State variables
    zi: npt.NDArray | None = None

    # Reset if these change.
    check_input = {"key": None, "shape": None}
    # fs changing will be handled by caller that creates coefficients.

    while True:
        msg_in: AxisArray = yield msg_out

        if coefs is None:
            # passthrough if we do not have a filter design.
            msg_out = msg_in
            continue

        axis = msg_in.dims[0] if axis is None else axis
        axis_idx = msg_in.get_axis_idx(axis)

        # Re-calculate/reset zi if necessary
        samp_shape = msg_in.data.shape[:axis_idx] + msg_in.data.shape[axis_idx + 1 :]
        b_reset = samp_shape != check_input["shape"]
        b_reset = b_reset or msg_in.key != check_input["key"]
        if b_reset:
            check_input["shape"] = samp_shape
            check_input["key"] = msg_in.key

            n_tail = msg_in.data.ndim - axis_idx - 1
            zi = zi_func(*coefs)
            zi_expand = (None,) * axis_idx + (slice(None),) + (None,) * n_tail
            n_tile = (
                msg_in.data.shape[:axis_idx] + (1,) + msg_in.data.shape[axis_idx + 1 :]
            )
            if coef_type == "sos":
                # sos zi must keep its leading dimension (`order / 2` for low|high; `order` for bpass|bstop)
                zi_expand = (slice(None),) + zi_expand
                n_tile = (1,) + n_tile
            zi = np.tile(zi[zi_expand], n_tile)

        if msg_in.data.size > 0:
            dat_out, zi = filt_func(*coefs, msg_in.data, axis=axis_idx, zi=zi)
        else:
            dat_out = msg_in.data
        msg_out = replace(msg_in, data=dat_out)


# Type aliases
BACoeffs = tuple[npt.NDArray, npt.NDArray]
SOSCoeffs = npt.NDArray
FilterCoefsMultiType = BACoeffs | SOSCoeffs


@consumer
def filter_gen_by_design(
    axis: str,
    coef_type: str,
    design_fun: typing.Callable[[float], FilterCoefsMultiType | None],
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Filter data using a filter whose coefficients are calculated using the provided design function.

    Args:
        axis: The name of the axis to filter.
            Note: The axis must be represented in the message .axes and be of type AxisArray.LinearAxis.
        coef_type: "ba" or "sos"
        design_fun: A callable that takes "fs" as its only argument and returns a tuple of filter coefficients.
          If the design_fun returns None then the filter will act as a passthrough.
          Hint: To make a design function that only requires fs, use functools.partial to set other parameters.
          See butterworthfilter for an example.

    Returns:

    """
    msg_out = AxisArray(np.array([]), dims=[""])

    # State variables
    # Initialize filtergen as passthrough until we receive a message that allows us to design the filter.
    filter_gen = filtergen(axis, None, coef_type)

    # Reset if these change.
    check_input = {"gain": None}
    # No need to check parameters that don't affect the design; filter_gen should check most of its parameters.

    while True:
        msg_in: AxisArray = yield msg_out
        axis = axis or msg_in.dims[0]
        b_reset = msg_in.axes[axis].gain != check_input["gain"]
        if b_reset:
            check_input["gain"] = msg_in.axes[axis].gain
            coefs = design_fun(1 / msg_in.axes[axis].gain)
            filter_gen = filtergen(axis, coefs, coef_type)

        msg_out = filter_gen.send(msg_in)


class FilterBaseSettings(ez.Settings):
    axis: str | None = None
    coef_type: str = "ba"


class FilterBase(GenAxisArray):
    SETTINGS = FilterBaseSettings

    # Backwards-compatible with `Filter` unit
    INPUT_FILTER = ez.InputStream(FilterCoefsMultiType)

    def design_filter(
        self,
    ) -> typing.Callable[[float], FilterCoefsMultiType | None]:
        raise NotImplementedError("Must implement 'design_filter' in Unit subclass!")

    def construct_generator(self):
        design_fun = self.design_filter()
        self.STATE.gen = filter_gen_by_design(
            self.SETTINGS.axis, self.SETTINGS.coef_type, design_fun
        )

    @ez.subscriber(INPUT_FILTER)
    async def redesign(self, message: FilterBaseSettings) -> None:
        self.apply_settings(message)
        self.construct_generator()


class FilterSettings(FilterBaseSettings):
    # If you'd like to statically design a filter, define it in settings
    coefs: FilterCoefficients | None = None
    # Note: coef_type = "ba" is assumed for this class.


class Filter(FilterBase):
    SETTINGS = FilterSettings

    INPUT_FILTER = ez.InputStream(FilterCoefficients)

    def design_filter(self) -> typing.Callable[[float], BACoeffs | None]:
        if self.SETTINGS.coefs is None:
            return lambda fs: None
        return lambda fs: (self.SETTINGS.coefs.b, self.SETTINGS.coefs.a)
