from dataclasses import field

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray

from .spectrogram import SpectrogramSettings, SpectrogramTransformer
from .aggregate import (
    AggregationFunction,
    RangedAggregateTransformer,
    RangedAggregateSettings,
)
from .base import CompositeProcessor, BaseStatefulProcessor, BaseTransformerUnit


class BandPowerSettings(ez.Settings):
    """
    Settings for ``BandPower``.
    """

    spectrogram_settings: SpectrogramSettings = field(
        default_factory=SpectrogramSettings
    )
    """
    Settings for spectrogram calculation.
    """

    bands: list[tuple[float, float]] | None = field(
        default_factory=lambda: [(17, 30), (70, 170)]
    )
    """
    (min, max) tuples of band limits in Hz.
    """


class BandPowerTransformer(CompositeProcessor[BandPowerSettings, AxisArray]):
    @staticmethod
    def _initialize_processors(
        settings: BandPowerSettings,
    ) -> dict[str, BaseStatefulProcessor]:
        return {
            "spectrogram": SpectrogramTransformer(
                settings=settings.spectrogram_settings
            ),
            "aggregate": RangedAggregateTransformer(
                settings=RangedAggregateSettings(
                    axis="freq",
                    bands=settings.bands,
                    operation=AggregationFunction.MEAN,
                )
            ),
        }


class BandPower(
    BaseTransformerUnit[BandPowerSettings, AxisArray, BandPowerTransformer]
):
    SETTINGS = BandPowerSettings


def bandpower(
    spectrogram_settings: SpectrogramSettings,
    bands: list[tuple[float, float]] | None = [
        (17, 30),
        (70, 170),
    ],
) -> BandPowerTransformer:
    """
    Calculate the average spectral power in each band.

    Returns:
        :obj:`BandPowerTransformer`
    """
    return BandPowerTransformer(
        settings=BandPowerSettings(
            spectrogram_settings=spectrogram_settings, bands=bands
        )
    )
