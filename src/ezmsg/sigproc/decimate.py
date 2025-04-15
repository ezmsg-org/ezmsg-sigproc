import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray

from .base import BaseTransformerUnit
from .cheby import ChebyshevFilterTransformer, ChebyshevFilterSettings
from .downsample import Downsample, DownsampleSettings
from .filter import BACoeffs, SOSCoeffs


class ChebyForDecimateTransformer(ChebyshevFilterTransformer[BACoeffs | SOSCoeffs]):
    """
    A :obj:`ChebyshevFilterTransformer` with a design filter method that additionally accepts a target sampling rate,
     and if the target rate cannot be achieved it returns None, else it returns the filter coefficients.
    """

    def get_design_function(
        self,
    ) -> typing.Callable[[float], BACoeffs | SOSCoeffs | None]:
        def cheby_opt_design_fun(fs: float) -> BACoeffs | SOSCoeffs | None:
            if fs is None:
                return None
            ds_factor = int(fs / (2.5 * self.settings.Wn))
            if ds_factor < 2:
                return None
            partial_fun = super(ChebyForDecimateTransformer, self).get_design_function()
            return partial_fun(fs)

        return cheby_opt_design_fun


class ChebyForDecimate(
    BaseTransformerUnit[
        ChebyshevFilterSettings, AxisArray, AxisArray, ChebyForDecimateTransformer
    ]
):
    SETTINGS = ChebyshevFilterSettings


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
