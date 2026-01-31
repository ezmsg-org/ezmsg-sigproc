"""
Time-domain single-band power estimation.

Two methods are provided:

1. **RMS Band Power** — Bandpass filter, square, window into bins, take the mean, optionally take the square root.
2. **Square-Law + LPF Band Power** — Bandpass filter, square, lowpass filter (smoothing), downsample.
"""

from dataclasses import field

import ezmsg.core as ez
from ezmsg.baseproc import (
    BaseProcessor,
    BaseStatefulProcessor,
    BaseTransformerUnit,
    CompositeProcessor,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.modify import modify_axis

from .aggregate import AggregateSettings, AggregateTransformer, AggregationFunction
from .butterworthfilter import ButterworthFilterSettings, ButterworthFilterTransformer
from .downsample import DownsampleSettings, DownsampleTransformer
from .math.pow import PowSettings, PowTransformer
from .window import WindowTransformer


class RMSBandPowerSettings(ez.Settings):
    """Settings for :obj:`RMSBandPowerTransformer`."""

    bandpass: ButterworthFilterSettings = field(
        default_factory=lambda: ButterworthFilterSettings(order=4, coef_type="sos")
    )
    """Butterworth bandpass filter settings. Set ``cuton`` and ``cutoff`` to define the band."""

    bin_duration: float = 0.05
    """Duration of each non-overlapping bin in seconds."""

    apply_sqrt: bool = True
    """If True, output is RMS (root-mean-square). If False, output is mean-square power."""


class RMSBandPowerTransformer(CompositeProcessor[RMSBandPowerSettings, AxisArray, AxisArray]):
    """
    RMS band power estimation.

    Pipeline: bandpass -> square -> window(bins) -> mean(time) -> rename bin->time -> [sqrt]
    """

    @staticmethod
    def _initialize_processors(
        settings: RMSBandPowerSettings,
    ) -> dict[str, BaseProcessor | BaseStatefulProcessor]:
        procs: dict[str, BaseProcessor | BaseStatefulProcessor] = {
            "bandpass": ButterworthFilterTransformer(settings.bandpass),
            "square": PowTransformer(PowSettings(exponent=2.0)),
            "window": WindowTransformer(
                axis="time",
                newaxis="bin",
                window_dur=settings.bin_duration,
                window_shift=settings.bin_duration,
                zero_pad_until="none",
            ),
            "aggregate": AggregateTransformer(AggregateSettings(axis="time", operation=AggregationFunction.MEAN)),
            "rename": modify_axis(name_map={"bin": "time"}),
        }
        if settings.apply_sqrt:
            procs["sqrt"] = PowTransformer(PowSettings(exponent=0.5))
        return procs


class RMSBandPower(BaseTransformerUnit[RMSBandPowerSettings, AxisArray, AxisArray, RMSBandPowerTransformer]):
    SETTINGS = RMSBandPowerSettings


class SquareLawBandPowerSettings(ez.Settings):
    """Settings for :obj:`SquareLawBandPowerTransformer`."""

    bandpass: ButterworthFilterSettings = field(
        default_factory=lambda: ButterworthFilterSettings(order=4, coef_type="sos")
    )
    """Butterworth bandpass filter settings. Set ``cuton`` and ``cutoff`` to define the band."""

    lowpass: ButterworthFilterSettings = field(
        default_factory=lambda: ButterworthFilterSettings(order=4, coef_type="sos")
    )
    """Butterworth lowpass filter settings for smoothing the squared signal."""

    downsample: DownsampleSettings = field(default_factory=DownsampleSettings)
    """Downsample settings for rate reduction after lowpass smoothing."""


class SquareLawBandPowerTransformer(CompositeProcessor[SquareLawBandPowerSettings, AxisArray, AxisArray]):
    """
    Square-law + LPF band power estimation.

    Pipeline: bandpass -> square -> lowpass -> downsample
    """

    @staticmethod
    def _initialize_processors(
        settings: SquareLawBandPowerSettings,
    ) -> dict[str, BaseProcessor | BaseStatefulProcessor]:
        return {
            "bandpass": ButterworthFilterTransformer(settings.bandpass),
            "square": PowTransformer(PowSettings(exponent=2.0)),
            "lowpass": ButterworthFilterTransformer(settings.lowpass),
            "downsample": DownsampleTransformer(settings.downsample),
        }


class SquareLawBandPower(
    BaseTransformerUnit[SquareLawBandPowerSettings, AxisArray, AxisArray, SquareLawBandPowerTransformer]
):
    SETTINGS = SquareLawBandPowerSettings
