from typing import Generator
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.modify import modify_axis

from .window import Anchor, WindowTransformer
from .spectrum import (
    WindowFunction,
    SpectralTransform,
    SpectralOutput,
    SpectrumTransformer,
)
from .base import (
    CompositeProcessor,
    BaseStatefulProcessor,
    BaseTransformerUnit,
)


class SpectrogramSettings(ez.Settings):
    """
    Settings for :obj:`SpectrogramTransformer`.
    """

    window_dur: float | None = None
    """window duration in seconds."""

    window_shift: float | None = None
    """"window step in seconds. If None, window_shift == window_dur"""

    window_anchor: str | Anchor = Anchor.BEGINNING
    """See :obj"`WindowTransformer`"""

    window: WindowFunction = WindowFunction.HAMMING
    """The :obj:`WindowFunction` to apply to the data slice prior to calculating the spectrum."""

    transform: SpectralTransform = SpectralTransform.REL_DB
    """The :obj:`SpectralTransform` to apply to the spectral magnitude."""

    output: SpectralOutput = SpectralOutput.POSITIVE
    """The :obj:`SpectralOutput` format."""


class SpectrogramTransformer(
    CompositeProcessor[SpectrogramSettings, AxisArray, AxisArray]
):
    @staticmethod
    def _initialize_processors(
        settings: SpectrogramSettings,
    ) -> dict[str, BaseStatefulProcessor | Generator[AxisArray, AxisArray, None]]:
        return {
            "windowing": WindowTransformer(
                axis="time",
                newaxis="win",
                window_dur=settings.window_dur,
                window_shift=settings.window_shift,
                zero_pad_until="shift"
                if settings.window_shift is not None
                else "input",
                anchor=settings.window_anchor,
            ),
            "spectrum": SpectrumTransformer(
                axis="time",
                window=settings.window,
                transform=settings.transform,
                output=settings.output,
            ),
            "modify_axis": modify_axis(name_map={"win": "time"}),
        }


class Spectrogram(
    BaseTransformerUnit[
        SpectrogramSettings, AxisArray, AxisArray, SpectrogramTransformer
    ]
):
    SETTINGS = SpectrogramSettings


def spectrogram(
    window_dur: float | None = None,
    window_shift: float | None = None,
    window_anchor: str | Anchor = Anchor.BEGINNING,
    window: WindowFunction = WindowFunction.HAMMING,
    transform: SpectralTransform = SpectralTransform.REL_DB,
    output: SpectralOutput = SpectralOutput.POSITIVE,
) -> SpectrogramTransformer:
    return SpectrogramTransformer(
        SpectrogramSettings(
            window_dur=window_dur,
            window_shift=window_shift,
            window_anchor=window_anchor,
            window=window,
            transform=transform,
            output=output,
        )
    )
