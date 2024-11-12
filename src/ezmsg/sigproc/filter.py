import asyncio
from dataclasses import dataclass, field
import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.util.generator import consumer
import numpy as np
import numpy.typing as npt
import scipy.signal


@dataclass
class FilterCoefficients:
    b: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    a: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))


def _normalize_coefs(
    coefs: typing.Union[
        FilterCoefficients, typing.Tuple[npt.NDArray, npt.NDArray], npt.NDArray
    ],
) -> typing.Tuple[str, typing.Tuple[npt.NDArray, ...]]:
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
    axis: str, coefs: typing.Optional[typing.Tuple[np.ndarray]], coef_type: str
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
    zi: typing.Optional[npt.NDArray] = None

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


class FilterSettingsBase(ez.Settings):
    axis: typing.Optional[str] = None
    fs: typing.Optional[float] = None


class FilterSettings(FilterSettingsBase):
    # If you'd like to statically design a filter, define it in settings
    filt: typing.Optional[FilterCoefficients] = None


class FilterState(ez.State):
    axis: typing.Optional[str] = None
    zi: typing.Optional[np.ndarray] = None
    filt_designed: bool = False
    filt: typing.Optional[FilterCoefficients] = None
    filt_set: asyncio.Event = field(default_factory=asyncio.Event)
    samp_shape: typing.Optional[typing.Tuple[int, ...]] = None
    fs: typing.Optional[float] = None  # Hz


class Filter(ez.Unit):
    SETTINGS = FilterSettingsBase
    STATE = FilterState

    INPUT_FILTER = ez.InputStream(FilterCoefficients)
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def design_filter(self) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError("Must implement 'design_filter' in Unit subclass!")

    # Set up filter with static initialization if specified
    async def initialize(self) -> None:
        if self.SETTINGS.axis is not None:
            self.STATE.axis = self.SETTINGS.axis

        if isinstance(self.SETTINGS, FilterSettings):
            if self.SETTINGS.filt is not None:
                self.STATE.filt = self.SETTINGS.filt
                self.STATE.filt_set.set()
        else:
            self.STATE.filt_set.clear()

        if self.SETTINGS.fs is not None:
            try:
                self.update_filter()
            except NotImplementedError:
                ez.logger.debug("Using filter coefficients.")

    @ez.subscriber(INPUT_FILTER)
    async def redesign(self, message: FilterCoefficients):
        self.STATE.filt = message

    def update_filter(self):
        try:
            coefs = self.design_filter()
            self.STATE.filt = (
                FilterCoefficients() if coefs is None else FilterCoefficients(*coefs)
            )
            self.STATE.filt_set.set()
            self.STATE.filt_designed = True
        except NotImplementedError as e:
            raise e
        except Exception as e:
            ez.logger.warning(f"Error when designing filter: {e}")

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_SIGNAL)
    async def apply_filter(self, msg: AxisArray) -> typing.AsyncGenerator:
        axis_name = msg.dims[0] if self.STATE.axis is None else self.STATE.axis
        axis_idx = msg.get_axis_idx(axis_name)
        axis = msg.get_axis(axis_name)
        fs = 1.0 / axis.gain

        if self.STATE.fs != fs and self.STATE.filt_designed is True:
            self.STATE.fs = fs
            self.update_filter()

        # Ensure filter is defined
        # TODO: Maybe have me be a passthrough filter until coefficients are received
        if self.STATE.filt is None:
            self.STATE.filt_set.clear()
            ez.logger.info("Awaiting filter coefficients...")
            await self.STATE.filt_set.wait()
            ez.logger.info("Filter coefficients received.")

        assert self.STATE.filt is not None

        arr_in = msg.data

        # If the array is one dimensional, add a temporary second dimension so that the math works out
        one_dimensional = False
        if arr_in.ndim == 1:
            arr_in = np.expand_dims(arr_in, axis=1)
            one_dimensional = True

        # We will perform filter with time dimension as last axis
        arr_in = np.moveaxis(arr_in, axis_idx, -1)
        samp_shape = arr_in[..., 0].shape

        # Re-calculate/reset zi if necessary
        if self.STATE.zi is None or samp_shape != self.STATE.samp_shape:
            zi: np.ndarray = scipy.signal.lfilter_zi(
                self.STATE.filt.b, self.STATE.filt.a
            )
            self.STATE.samp_shape = samp_shape
            self.STATE.zi = np.array([zi] * np.prod(self.STATE.samp_shape))
            self.STATE.zi = self.STATE.zi.reshape(
                tuple(list(self.STATE.samp_shape) + [zi.shape[0]])
            )

        arr_out, self.STATE.zi = scipy.signal.lfilter(
            self.STATE.filt.b, self.STATE.filt.a, arr_in, zi=self.STATE.zi
        )

        arr_out = np.moveaxis(arr_out, -1, axis_idx)

        # Remove temporary first dimension if necessary
        if one_dimensional:
            arr_out = np.squeeze(arr_out, axis=1)

        yield (
            self.OUTPUT_SIGNAL,
            replace(msg, data=arr_out),
        )
