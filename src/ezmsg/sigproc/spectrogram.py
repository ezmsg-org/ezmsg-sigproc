import pickle
import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.modify import modify_axis

from .window import Anchor, WindowTransformer, WindowState
from .spectrum import WindowFunction, SpectralTransform, SpectralOutput, SpectrumTransformer, SpectrumState
from .base import BaseSignalTransformer, BaseSignalTransformerUnit


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


class SpectrogramState(ez.State):
    """
    State for :obj:`Spectrogram`.
    """
    window_state: WindowState | None = None
    spectrum_state: SpectrumState | None = None


class SpectrogramTransformer(
    BaseSignalTransformer[SpectrogramState, SpectrogramSettings, AxisArray]
):
    def __init__(self, *args, settings: typing.Optional[SpectrogramSettings] = None, **kwargs):
        super().__init__(*args, settings=settings, **kwargs)
        self._windowing = WindowTransformer(
            axis="time",
            newaxis="win",
            window_dur=self.settings.window_dur,
            window_shift=self.settings.window_shift,
            zero_pad_until="shift" if self.settings.window_shift is not None else "input",
            anchor=self.settings.window_anchor,
        )
        self._spectrum = SpectrumTransformer(axis="time", window=self.settings.window, transform=self.settings.transform,
                 output=self.settings.output)
        self._modify_axis = modify_axis(name_map={"win": "time"})

    def check_metadata(self, message: AxisArray) -> bool:
        # Unused because we override __call__
        return False

    def reset(self, message: AxisArray) -> None:
        # Unused because we override __call__
        pass

    def _process(self, message: AxisArray) -> AxisArray:
        # Unused because we override __call__
        # TODO: Maybe _process should be removed from the protocol.
        return message

    def __call__(self, message: AxisArray):
        message = self._windowing(message)
        message = self._spectrum(message)
        message = self._modify_axis.send(message)
        return message

    @property
    def state(self) -> SpectrogramState:
        # Update self._state with the latest state from the sub-transformers
        self._state.window_state = self._windowing.state
        self._state.spectrum_state = self._spectrum.state
        return self._state

    @state.setter
    def state(self, state: SpectrogramState | bytes | None) -> None:
        """
        Restore state from serialized state or SpectrogramState instance.

        Args:
            state: _description_
        """
        # Ignore state if None. This is required for stateful_op calls that do not want to update state.
        if state is not None:
            if isinstance(state, bytes):
                self._state = pickle.loads(state)
            else:
                self._state = state

            # Update sub-transformer states with provided state
            self._windowing.state = self._state.window_state
            self._spectrum.state = self._state.spectrum_state


class Spectrum(
    BaseSignalTransformerUnit[
        SpectrogramState, SpectrogramSettings, AxisArray, SpectrogramTransformer
    ]
):
    SETTINGS = SpectrogramSettings

