import typing

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .filterbank import (
    FilterbankMode,
    FilterbankSettings,
    FilterbankTransformer,
    MinPhaseMode,
)
from .kaiser import KaiserFilterSettings, kaiser_design_fun


class FilterbankDesignSettings(ez.Settings):
    filters: typing.Iterable[KaiserFilterSettings]

    mode: FilterbankMode = FilterbankMode.CONV
    """
    "conv", "fft", or "auto". If "auto", the mode is determined by the size of the input data.
      fft mode is more efficient for long kernels. However, fft mode uses non-overlapping windows and will
      incur a delay equal to the window length, which is larger than the largest kernel.
      conv mode is less efficient but will return data for every incoming chunk regardless of how small it is
      and thus can provide shorter latency updates.
    """

    min_phase: MinPhaseMode = MinPhaseMode.NONE
    """
    If not None, convert the kernels to minimum-phase equivalents. Valid options are
      'hilbert', 'homomorphic', and 'homomorphic-full'. Complex filters not supported.
      See `scipy.signal.minimum_phase` for details.
    """

    axis: str = "time"
    """The name of the axis to operate on. This should usually be "time"."""

    new_axis: str = "kernel"
    """The name of the new axis corresponding to the kernel index."""


@processor_state
class FilterbankDesignState:
    filterbank: FilterbankTransformer | None = None
    needs_redesign: bool = False


class FilterbankDesignTransformer(
    BaseStatefulTransformer[FilterbankDesignSettings, AxisArray, AxisArray, FilterbankDesignState],
):
    """
    Transformer that designs and applies a filterbank based on Kaiser windowed FIR filters.
    """

    @classmethod
    def get_message_type(cls, dir: str) -> type[AxisArray]:
        if dir in ("in", "out"):
            return AxisArray
        else:
            raise ValueError(f"Invalid direction: {dir}. Must be 'in' or 'out'.")

    def update_settings(self, new_settings: typing.Optional[FilterbankDesignSettings] = None, **kwargs) -> None:
        """
        Update settings and mark that filter coefficients need to be recalculated.

        Args:
            new_settings: Complete new settings object to replace current settings
            **kwargs: Individual settings to update
        """
        # Update settings
        if new_settings is not None:
            self.settings = new_settings
        else:
            self.settings = replace(self.settings, **kwargs)

        # Set flag to trigger recalculation on next message
        if self.state.filterbank is not None:
            self.state.needs_redesign = True

    def _calculate_kernels(self, fs: float) -> list[npt.NDArray]:
        kernels = []
        for filter in self.settings.filters:
            output = kaiser_design_fun(
                fs,
                cutoff=filter.cutoff,
                ripple=filter.ripple,
                width=filter.width,
                pass_zero=filter.pass_zero,
                wn_hz=filter.wn_hz,
            )

            kernels.append(np.array([1.0]) if output is None else output[0])
        return kernels

    def __call__(self, message: AxisArray) -> AxisArray:
        if self.state.filterbank is not None and self.state.needs_redesign:
            self._reset_state(message)
            self.state.needs_redesign = False
        return super().__call__(message)

    def _hash_message(self, message: AxisArray) -> int:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        gain = message.axes[axis].gain if hasattr(message.axes[axis], "gain") else 1
        axis_idx = message.get_axis_idx(axis)
        samp_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        return hash((message.key, samp_shape, gain))

    def _reset_state(self, message: AxisArray) -> None:
        axis_obj = message.axes[self.settings.axis]
        assert isinstance(axis_obj, AxisArray.LinearAxis)
        fs = 1 / axis_obj.gain
        kernels = self._calculate_kernels(fs)
        new_settings = FilterbankSettings(
            kernels=kernels,
            mode=self.settings.mode,
            min_phase=self.settings.min_phase,
            axis=self.settings.axis,
            new_axis=self.settings.new_axis,
        )
        self.state.filterbank = FilterbankTransformer(settings=new_settings)

    def _process(self, message: AxisArray) -> AxisArray:
        return self.state.filterbank(message)
