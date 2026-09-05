"""Decimation (downsample with anti-alias filtering)."""

import typing

import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray

from .cheby import ChebyshevFilterSettings, ChebyshevFilterTransformer
from .downsample import Downsample, DownsampleSettings
from .filter import BACoeffs, SOSCoeffs
from .util.deprecation import suppress_axis_deprecation, warn_axis_deprecated


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


class ChebyForDecimate(BaseTransformerUnit[ChebyshevFilterSettings, AxisArray, AxisArray, ChebyForDecimateTransformer]):
    SETTINGS = ChebyshevFilterSettings


class DecimateSettings(DownsampleSettings):
    """Settings for :obj:`Decimate`.

    Adds the anti-aliasing filter's ``axis`` on top of the downsampler's
    settings. The filter has one because a filter legitimately runs along any
    dimension; the downsampler does not, because its phase counter only means
    something along the dimension messages accumulate along. Leave ``axis``
    matching the stream's chunk dimension -- filtering one dimension and
    decimating another is not decimation.
    """

    axis: str | None = None
    """.. deprecated:: 3.8
        Scheduled for removal in 4.0. The dimension messages accumulate along
        now comes from :attr:`~ezmsg.util.messages.axisarray.AxisArray.chunk_dim`;
        see :mod:`ezmsg.sigproc.util.deprecation`."""

    def __post_init__(self) -> None:
        warn_axis_deprecated(self)


class Decimate(ez.Collection):
    """
    A :obj:`Collection` chaining a :obj:`Filter` node configured as a lowpass Chebyshev filter
    and a :obj:`Downsample` node.
    """

    SETTINGS = DecimateSettings

    INPUT_SIGNAL = ez.InputTopic(AxisArray)
    OUTPUT_SIGNAL = ez.OutputTopic(AxisArray)

    FILTER = ChebyForDecimate()
    DOWNSAMPLE = Downsample()

    def configure(self) -> None:
        # Already warned about on DecimateSettings, whose `axis` exists only to
        # reach this filter.
        with suppress_axis_deprecation():
            cheby_settings = ChebyshevFilterSettings(
                order=8,
                ripple_tol=0.05,
                Wn=0.4 * self.SETTINGS.target_rate,
                btype="lowpass",
                axis=self.SETTINGS.axis,
                wn_hz=True,
            )
        self.FILTER.apply_settings(cheby_settings)
        # `axis` is the filter's, not the downsampler's -- pass only what
        # DownsampleSettings still declares.
        self.DOWNSAMPLE.apply_settings(
            DownsampleSettings(target_rate=self.SETTINGS.target_rate, factor=self.SETTINGS.factor)
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.FILTER.INPUT_SIGNAL),
            (self.FILTER.OUTPUT_SIGNAL, self.DOWNSAMPLE.INPUT_SIGNAL),
            (self.DOWNSAMPLE.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
