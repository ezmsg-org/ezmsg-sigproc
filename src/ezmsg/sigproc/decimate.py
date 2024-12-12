import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray

from .cheby import ChebyshevFilter, ChebyshevFilterSettings
from .downsample import Downsample, DownsampleSettings
from .filter import FilterCoefsMultiType


class ChebyForDecimate(ChebyshevFilter):
    """
    A :obj:`ChebyshevFilter` node with a design filter method that additionally accepts a target sampling rate,
     and if the target rate cannot be achieved it returns None, else it returns the filter coefficients.
    """

    def design_filter(
        self,
    ) -> typing.Callable[[float], FilterCoefsMultiType | None]:
        def cheby_opt_design_fun(fs: float) -> FilterCoefsMultiType | None:
            if fs is None:
                return None
            ds_factor = int(fs / (2.5 * self.SETTINGS.Wn))
            if ds_factor < 2:
                return None
            partial_fun = super(ChebyForDecimate, self).design_filter()
            return partial_fun(fs)
        return cheby_opt_design_fun


class Decimate(ez.Collection):
    """
    A :obj:`Collection` chaining a :obj:`Filter` node configured as a lowpass Chebyshev filter
    and a :obj:`Downsample` node.
    """

    SETTINGS = DownsampleSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    FILTER = ChebyForDecimate()
    DOWNSAMPLE = Downsample()

    def configure(self) -> None:

        cheby_settings = ChebyshevFilterSettings(
            order=8,
            ripple_tol=0.05,
            Wn=0.4 * self.SETTINGS.target_rate,
            btype="lowpass",
            axis=self.SETTINGS.axis,
            wn_hz=True,
        )
        self.FILTER.apply_settings(cheby_settings)
        self.DOWNSAMPLE.apply_settings(self.SETTINGS)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.FILTER.INPUT_SIGNAL),
            (self.FILTER.OUTPUT_SIGNAL, self.DOWNSAMPLE.INPUT_SIGNAL),
            (self.DOWNSAMPLE.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
