"""
Compute differences along an axis.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from array_api_compat import get_namespace
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.messages.util import replace


class DiffSettings(ez.Settings):
    axis: str | None = None
    scale_by_fs: bool = False


@processor_state
class DiffState:
    last_dat: npt.NDArray | None = None
    last_time: float | None = None


class DiffTransformer(BaseStatefulTransformer[DiffSettings, AxisArray, AxisArray, DiffState]):
    def _hash_message(self, message: AxisArray) -> int:
        ax_idx = message.get_axis_idx(self.settings.axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        return hash((sample_shape, message.key))

    def _reset_state(self, message) -> None:
        ax_idx = message.get_axis_idx(self.settings.axis)
        self.state.last_dat = slice_along_axis(message.data, slice(0, 1), axis=ax_idx)
        if self.settings.scale_by_fs:
            ax_info = message.get_axis(self.settings.axis)
            if hasattr(ax_info, "data"):
                if len(ax_info.data) > 1:
                    self.state.last_time = 2 * ax_info.data[0] - ax_info.data[1]
                else:
                    self.state.last_time = ax_info.data[0] - 0.001

    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        axis = self.settings.axis or message.dims[0]
        ax_idx = message.get_axis_idx(axis)

        diffs = xp.diff(
            xp.concat((self.state.last_dat, message.data), axis=ax_idx),
            axis=ax_idx,
        )
        # Prepare last_dat for next iteration
        self.state.last_dat = slice_along_axis(message.data, slice(-1, None), axis=ax_idx)
        # Scale by fs if requested. This converts the diff to a derivative. e.g., diff of position becomes velocity.
        if self.settings.scale_by_fs:
            ax_info = message.get_axis(axis)
            if hasattr(ax_info, "data"):
                # ax_info.data is typically numpy for metadata, so use np.diff here
                dt = np.diff(np.concatenate(([self.state.last_time], ax_info.data)))
                # Expand dt dims to match diffs
                exp_sl = (None,) * ax_idx + (Ellipsis,) + (None,) * (message.data.ndim - ax_idx - 1)
                diffs /= xp.asarray(dt[exp_sl])
                self.state.last_time = ax_info.data[-1]  # For next iteration
            else:
                diffs /= ax_info.gain

        return replace(message, data=diffs)


class DiffUnit(BaseTransformerUnit[DiffSettings, AxisArray, AxisArray, DiffTransformer]):
    SETTINGS = DiffSettings


def diff(axis: str = "time", scale_by_fs: bool = False) -> DiffTransformer:
    return DiffTransformer(DiffSettings(axis=axis, scale_by_fs=scale_by_fs))
