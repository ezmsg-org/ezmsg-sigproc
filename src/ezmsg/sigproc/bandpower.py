"""Spectral band power estimation via windowed FFT and aggregation."""

from dataclasses import field

import ezmsg.core as ez
from ezmsg.baseproc import (
    BaseProcessor,
    BaseStatefulProcessor,
    BaseTransformerUnit,
    CompositeProcessor,
)
from ezmsg.util.messages.axisarray import AxisArray

from .aggregate import (
    AggregationFunction,
    RangedAggregateSettings,
    RangedAggregateTransformer,
)
from .materialize import MaterializeMode, materialize_array
from .spectrogram import SpectrogramSettings, SpectrogramTransformer


class BandPowerSettings(ez.Settings):
    """
    Settings for ``BandPower``.
    """

    spectrogram_settings: SpectrogramSettings = field(default_factory=SpectrogramSettings)
    """
    Settings for spectrogram calculation.
    """

    bands: list[tuple[float, float]] | None = field(default_factory=lambda: [(17, 30), (70, 170)])
    """
    (min, max) tuples of band limits in Hz.
    """

    aggregation: AggregationFunction = AggregationFunction.MEAN
    """:obj:`AggregationFunction` to apply to each band."""

    materialize: MaterializeMode = MaterializeMode.ASYNC
    """How to evaluate the output on a lazy backend (MLX); see
    :obj:`~ezmsg.sigproc.materialize.MaterializeMode`. No-op elsewhere.

    Defaults to :obj:`~ezmsg.sigproc.materialize.MaterializeMode.ASYNC`: this
    node's output is the end of a spectrogram chain, so evaluating it here keeps
    a lazy graph from accumulating even if nothing downstream forces it -- but
    the caller has no need of the values on the host, so there is nothing to
    block for. Set ``OFF`` if a downstream node already materializes every
    cycle, or ``SYNC`` to time this stage's work as its own."""


class BandPowerTransformer(CompositeProcessor[BandPowerSettings, AxisArray, AxisArray]):
    @staticmethod
    def _initialize_processors(
        settings: BandPowerSettings,
    ) -> dict[str, BaseProcessor | BaseStatefulProcessor]:
        return {
            "spectrogram": SpectrogramTransformer(settings=settings.spectrogram_settings),
            "aggregate": RangedAggregateTransformer(
                settings=RangedAggregateSettings(
                    axis="freq",
                    bands=settings.bands,
                    operation=settings.aggregation,
                )
            ),
        }

    def _post_process(self, result: AxisArray | None) -> AxisArray | None:
        if result is not None:
            materialize_array(result.data, self.settings.materialize)
        return result


class BandPower(BaseTransformerUnit[BandPowerSettings, AxisArray, AxisArray, BandPowerTransformer]):
    SETTINGS = BandPowerSettings


def bandpower(
    spectrogram_settings: SpectrogramSettings,
    bands: list[tuple[float, float]] | None = [
        (17, 30),
        (70, 170),
    ],
    aggregation: AggregationFunction = AggregationFunction.MEAN,
    materialize: MaterializeMode = MaterializeMode.ASYNC,
) -> BandPowerTransformer:
    """
    Calculate the average spectral power in each band.

    Returns:
        :obj:`BandPowerTransformer`
    """
    return BandPowerTransformer(
        settings=BandPowerSettings(
            spectrogram_settings=spectrogram_settings,
            bands=bands,
            aggregation=aggregation,
            materialize=materialize,
        )
    )
