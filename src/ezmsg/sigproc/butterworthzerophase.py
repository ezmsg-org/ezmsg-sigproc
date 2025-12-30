import functools
import typing

import ezmsg.core as ez
import numpy as np
import scipy.signal
from ezmsg.baseproc import SettingsType
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ezmsg.sigproc.butterworthfilter import ButterworthFilterSettings, butter_design_fun
from ezmsg.sigproc.filter import (
    BACoeffs,
    BaseFilterByDesignTransformerUnit,
    FilterByDesignTransformer,
    SOSCoeffs,
)


class ButterworthZeroPhaseSettings(ButterworthFilterSettings):
    """Settings for :obj:`ButterworthZeroPhase`."""

    # axis, coef_type, order, cuton, cutoff, wn_hz are inherited from ButterworthFilterSettings
    padtype: str | None = None
    """
    Padding type to use in `scipy.signal.filtfilt`.
    Must be one of {'odd', 'even', 'constant', None}.
    Default is None for no padding.
    """

    padlen: int | None = 0
    """
    Length of the padding to use in `scipy.signal.filtfilt`.
    If None, SciPy's default padding is used.
    """


class ButterworthZeroPhaseTransformer(FilterByDesignTransformer[ButterworthZeroPhaseSettings, BACoeffs | SOSCoeffs]):
    """Zero-phase (filtfilt) Butterworth using your design function."""

    def get_design_function(
        self,
    ) -> typing.Callable[[float], BACoeffs | SOSCoeffs | None]:
        return functools.partial(
            butter_design_fun,
            order=self.settings.order,
            cuton=self.settings.cuton,
            cutoff=self.settings.cutoff,
            coef_type=self.settings.coef_type,
            wn_hz=self.settings.wn_hz,
        )

    def update_settings(self, new_settings: typing.Optional[SettingsType] = None, **kwargs) -> None:
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
        self._coefs_cache = None
        self._fs_cache = None
        self.state.needs_redesign = True

    def _reset_state(self, message: AxisArray) -> None:
        self._coefs_cache = None
        self._fs_cache = None
        self.state.needs_redesign = True

    def _process(self, message: AxisArray) -> AxisArray:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        ax_idx = message.get_axis_idx(axis)
        fs = 1 / message.axes[axis].gain

        if (
            self._coefs_cache is None
            or self.state.needs_redesign
            or (self._fs_cache is None or not np.isclose(self._fs_cache, fs))
        ):
            self._coefs_cache = self.get_design_function()(fs)
            self._fs_cache = fs
            self.state.needs_redesign = False

        if self._coefs_cache is None or self.settings.order <= 0 or message.data.size <= 0:
            return message

        x = message.data
        if self.settings.coef_type == "sos":
            y = scipy.signal.sosfiltfilt(
                self._coefs_cache,
                x,
                axis=ax_idx,
                padtype=self.settings.padtype,
                padlen=self.settings.padlen,
            )
        elif self.settings.coef_type == "ba":
            b, a = self._coefs_cache
            y = scipy.signal.filtfilt(
                b,
                a,
                x,
                axis=ax_idx,
                padtype=self.settings.padtype,
                padlen=self.settings.padlen,
            )
        else:
            ez.logger.error("coef_type must be 'sos' or 'ba'.")
            raise ValueError("coef_type must be 'sos' or 'ba'.")

        return replace(message, data=y)


class ButterworthZeroPhase(
    BaseFilterByDesignTransformerUnit[ButterworthZeroPhaseSettings, ButterworthZeroPhaseTransformer]
):
    SETTINGS = ButterworthZeroPhaseSettings
