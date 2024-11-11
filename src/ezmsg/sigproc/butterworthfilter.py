import typing

import ezmsg.core as ez
import scipy.signal
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer

from .filter import filtergen, Filter, FilterState, FilterSettingsBase


class ButterworthFilterSettings(FilterSettingsBase):
    """Settings for :obj:`ButterworthFilter`."""

    order: int = 0
    """
    Filter order
    """

    cuton: typing.Optional[float] = None
    """
    Cuton frequency (Hz). If `cutoff` is not specified then this is the highpass corner. Otherwise,
    if this is lower than `cutoff` then this is the beginning of the bandpass
    or if this is greater than `cutoff` then this is the end of the bandstop. 
    """

    cutoff: typing.Optional[float] = None
    """
    Cutoff frequency (Hz). If `cuton` is not specified then this is the lowpass corner. Otherwise,
    if this is greater than `cuton` then this is the end of the bandpass,
    or if this is less than `cuton` then this is the beginning of the bandstop. 
    """

    def filter_specs(
        self,
    ) -> typing.Optional[
        typing.Tuple[str, typing.Union[float, typing.Tuple[float, float]]]
    ]:
        """
        Determine the filter type given the corner frequencies.

        Returns:
            A tuple with the first element being a string indicating the filter type
            (one of "lowpass", "highpass", "bandpass", "bandstop")
            and the second element being the corner frequency or frequencies.

        """
        if self.cuton is None and self.cutoff is None:
            return None
        elif self.cuton is None and self.cutoff is not None:
            return "lowpass", self.cutoff
        elif self.cuton is not None and self.cutoff is None:
            return "highpass", self.cuton
        elif self.cuton is not None and self.cutoff is not None:
            if self.cuton <= self.cutoff:
                return "bandpass", (self.cuton, self.cutoff)
            else:
                return "bandstop", (self.cutoff, self.cuton)


@consumer
def butter(
    axis: typing.Optional[str],
    order: int = 0,
    cuton: typing.Optional[float] = None,
    cutoff: typing.Optional[float] = None,
    coef_type: str = "ba",
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Apply Butterworth filter to streaming data. Uses :obj:`scipy.signal.butter` to design the filter.
    See :obj:`ButterworthFilterSettings.filter_specs` for an explanation of specifying different
    filter types (lowpass, highpass, bandpass, bandstop) from the parameters.

    Args:
        axis: The name of the axis to filter.
            Note: The axis must be represented in the message .axes and be of type AxisArray.LinearAxis.
        order: Filter order.
        cuton: Corner frequency of the filter in Hz.
        cutoff: Corner frequency of the filter in Hz.
        coef_type: "ba" or "sos"

    Returns:
        A primed generator object which accepts an :obj:`AxisArray` via .send(axis_array)
         and yields an :obj:`AxisArray` with filtered data.

    """
    # IO
    msg_out = AxisArray(np.array([]), dims=[""])

    # Check parameters
    btype, cutoffs = ButterworthFilterSettings(
        order=order, cuton=cuton, cutoff=cutoff
    ).filter_specs()

    # State variables
    # Initialize filtergen as passthrough until we can calculate coefs.
    filter_gen = filtergen(axis, None, coef_type)

    # Reset if these change.
    check_input = {"gain": None}
    # Key not checked because filter_gen will handle resetting if .key changes.

    while True:
        msg_in: AxisArray = yield msg_out
        axis = axis or msg_in.dims[0]

        b_reset = msg_in.axes[axis].gain != check_input["gain"]
        b_reset = b_reset and order > 0  # Not passthrough
        if b_reset:
            check_input["gain"] = msg_in.axes[axis].gain
            coefs = scipy.signal.butter(
                order,
                Wn=cutoffs,
                btype=btype,
                fs=1 / msg_in.axes[axis].gain,
                output=coef_type,
            )
            filter_gen = filtergen(axis, coefs, coef_type)

        msg_out = filter_gen.send(msg_in)


class ButterworthFilterState(FilterState):
    design: ButterworthFilterSettings


class ButterworthFilter(Filter):
    """:obj:`Unit` for :obj:`butterworth`"""

    SETTINGS = ButterworthFilterSettings
    STATE = ButterworthFilterState

    INPUT_FILTER = ez.InputStream(ButterworthFilterSettings)

    async def initialize(self) -> None:
        self.STATE.design = self.SETTINGS
        self.STATE.filt_designed = True
        await super().initialize()

    def design_filter(self) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]:
        specs = self.STATE.design.filter_specs()
        if self.STATE.design.order > 0 and specs is not None:
            btype, cut = specs
            return scipy.signal.butter(
                self.STATE.design.order,
                Wn=cut,
                btype=btype,
                fs=self.STATE.fs,
                output="ba",
            )
        else:
            return None

    @ez.subscriber(INPUT_FILTER)
    async def redesign(self, message: ButterworthFilterSettings) -> None:
        if type(message) is not ButterworthFilterSettings:
            return

        if self.STATE.design.order != message.order:
            self.STATE.zi = None
        self.STATE.design = message
        self.update_filter()
