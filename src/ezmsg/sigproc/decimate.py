import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray

from .cheby import ChebyshevFilter, ChebyshevFilterSettings
from .downsample import Downsample, DownsampleSettings


class Decimate(ez.Collection):
    """
    A :obj:`Collection` chaining a :obj:`Filter` node configured as a lowpass Chebyshev filter
    and a :obj:`Downsample` node.
    """

    SETTINGS = DownsampleSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    FILTER = ChebyshevFilter()
    DOWNSAMPLE = Downsample()

    def configure(self) -> None:
        cheby_settings = ChebyshevFilterSettings(
            order=8 if self.SETTINGS.factor > 1 else 0,
            ripple_tol=0.05,
            Wn=0.8 / self.SETTINGS.factor if self.SETTINGS.factor > 1 else None,
            btype="lowpass",
            axis=self.SETTINGS.axis,
            wn_hz=False,
        )
        self.FILTER.apply_settings(cheby_settings)
        self.DOWNSAMPLE.apply_settings(self.SETTINGS)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.FILTER.INPUT_SIGNAL),
            (self.FILTER.OUTPUT_SIGNAL, self.DOWNSAMPLE.INPUT_SIGNAL),
            (self.DOWNSAMPLE.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
