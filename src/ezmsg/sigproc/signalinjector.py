import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
import numpy as np
import numpy.typing as npt

from .base import (
    BaseAsyncTransformer,
    BaseTransformerUnit,
    processor_state,
)


class SignalInjectorSettings(ez.Settings):
    time_dim: str = "time"  # Input signal needs a time dimension with units in sec.
    frequency: float | None = None  # Hz
    amplitude: float = 1.0
    mixing_seed: int | None = None


@processor_state
class SignalInjectorState:
    cur_shape: tuple[int, ...] | None = None
    cur_frequency: float | None = None
    cur_amplitude: float | None = None
    mixing: npt.NDArray | None = None


class SignalInjectorTransformer(
    BaseAsyncTransformer[
        SignalInjectorSettings, AxisArray, AxisArray, SignalInjectorState
    ]
):
    def _hash_message(self, message: AxisArray) -> int:
        time_ax_idx = message.get_axis_idx(self.settings.time_dim)
        sample_shape = (
            message.data.shape[:time_ax_idx] + message.data.shape[time_ax_idx + 1 :]
        )
        return hash((message.key,) + sample_shape)

    def _reset_state(self, message: AxisArray) -> None:
        if self._state.cur_frequency is None:
            self._state.cur_frequency = self.settings.frequency
        if self._state.cur_amplitude is None:
            self._state.cur_amplitude = self.settings.amplitude
        time_ax_idx = message.get_axis_idx(self.settings.time_dim)
        self._state.cur_shape = (
            message.data.shape[:time_ax_idx] + message.data.shape[time_ax_idx + 1 :]
        )
        rng = np.random.default_rng(self.settings.mixing_seed)
        self._state.mixing = rng.random((1, message.shape2d(self.settings.time_dim)[1]))
        self._state.mixing = (self._state.mixing * 2.0) - 1.0

    async def _aprocess(self, message: AxisArray) -> AxisArray:
        if self._state.cur_frequency is None:
            return message
        out_msg = replace(message, data=message.data.copy())
        t = out_msg.ax(self.settings.time_dim).values[..., np.newaxis]
        signal = np.sin(2 * np.pi * self._state.cur_frequency * t)
        mixed_signal = signal * self._state.mixing * self._state.cur_amplitude
        with out_msg.view2d(self.settings.time_dim) as view:
            view[...] = view + mixed_signal.astype(view.dtype)
        return out_msg


class SignalInjector(
    BaseTransformerUnit[
        SignalInjectorSettings, AxisArray, AxisArray, SignalInjectorTransformer
    ]
):
    SETTINGS = SignalInjectorSettings
    INPUT_FREQUENCY = ez.InputStream(float | None)
    INPUT_AMPLITUDE = ez.InputStream(float)

    @ez.subscriber(INPUT_FREQUENCY)
    async def on_frequency(self, msg: float | None) -> None:
        self.processor.state.cur_frequency = msg

    @ez.subscriber(INPUT_AMPLITUDE)
    async def on_amplitude(self, msg: float) -> None:
        self.processor.state.cur_amplitude = msg
