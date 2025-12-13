import functools
import typing

import numpy as np
import numpy.typing as npt
import scipy.signal

from .filter import (
    BACoeffs,
    BaseFilterByDesignTransformerUnit,
    FilterBaseSettings,
    FilterByDesignTransformer,
)


class FIRFilterSettings(FilterBaseSettings):
    """Settings for :obj:`FIRFilter`.  See scipy.signal.firwin for more details"""

    # axis and coef_type are inherited from FilterBaseSettings

    order: int = 0
    """
    Filter order/number of taps
    """

    cutoff: float | npt.ArrayLike | None = None
    """
    Cutoff frequency of filter (expressed in the same units as fs) OR an array of cutoff frequencies
    (that is, band edges). In the former case, as a float, the cutoff frequency should correspond with
    the half-amplitude point, where the attenuation will be -6dB. In the latter case, the frequencies in
    cutoff should be positive and monotonically increasing between 0 and fs/2. The values 0 and fs/2 must
    not be included in cutoff.
    """

    width: float | None = None
    """
    If width is not None, then assume it is the approximate width of the transition region (expressed in
    the same units as fs) for use in Kaiser FIR filter design. In this case, the window argument is ignored.
    """

    window: str | None = "hamming"
    """
    Desired window to use.  See scipy.signal.get_window for a list of windows and required parameters.
    """

    pass_zero: bool | str = True
    """
    If True, the gain at the frequency 0 (i.e., the “DC gain”) is 1. If False, the DC gain is 0. Can also
    be a string argument for the desired filter type (equivalent to btype in IIR design functions).
    {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
    """

    scale: bool = True
    """
    Set to True to scale the coefficients so that the frequency response is exactly unity at a certain
    frequency. That frequency is either:
    * 0 (DC) if the first passband starts at 0 (i.e. pass_zero is True)
    * fs/2 (the Nyquist frequency) if the first passband ends at fs/2
        (i.e the filter is a single band highpass filter);
        center of first passband otherwise
    """

    wn_hz: bool = True
    """
    Set False if provided Wn are normalized from 0 to 1, where 1 is the Nyquist frequency
    """


def firwin_design_fun(
    fs: float,
    order: int = 0,
    cutoff: float | npt.ArrayLike | None = None,
    width: float | None = None,
    window: str | None = "hamming",
    pass_zero: bool | str = True,
    scale: bool = True,
    wn_hz: bool = True,
) -> BACoeffs | None:
    """
    Design an `order`th-order FIR filter and return the filter coefficients.
    See :obj:`FIRFilterSettings` for argument description.

    Returns:
        The filter taps as designed by firwin
    """
    if order > 0:
        taps = scipy.signal.firwin(
            numtaps=order,
            cutoff=cutoff,
            width=width,
            window=window,
            pass_zero=pass_zero,
            scale=scale,
            fs=fs if wn_hz else None,
        )
        return (taps, np.array([1.0]))
    return None


class FIRFilterTransformer(FilterByDesignTransformer[FIRFilterSettings, BACoeffs]):
    def get_design_function(
        self,
    ) -> typing.Callable[[float], BACoeffs | None]:
        return functools.partial(
            firwin_design_fun,
            order=self.settings.order,
            cutoff=self.settings.cutoff,
            width=self.settings.width,
            window=self.settings.window,
            pass_zero=self.settings.pass_zero,
            scale=self.settings.scale,
            wn_hz=self.settings.wn_hz,
        )


class FIRFilter(BaseFilterByDesignTransformerUnit[FIRFilterSettings, FIRFilterTransformer]):
    SETTINGS = FIRFilterSettings
