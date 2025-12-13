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


class KaiserFilterSettings(FilterBaseSettings):
    """Settings for :obj:`KaiserFilter`"""

    # axis and coef_type are inherited from FilterBaseSettings

    cutoff: float | npt.ArrayLike | None = None
    """
    Cutoff frequency of filter (expressed in the same units as fs) OR an array of cutoff frequencies
    (that is, band edges). In the former case, as a float, the cutoff frequency should correspond with
    the half-amplitude point, where the attenuation will be -6dB. In the latter case, the frequencies in
    cutoff should be positive and monotonically increasing between 0 and fs/2. The values 0 and fs/2 must
    not be included in cutoff.
    """

    ripple: float | None = None
    """
    Upper bound for the deviation (in dB) of the magnitude of the filter's frequency response from that of
    the desired filter (not including frequencies in any transition intervals).
    See scipy.signal.kaiserord for more information.
    """

    width: float | None = None
    """
    If width is not None, then assume it is the approximate width of the transition region (expressed in
    the same units as fs) for use in Kaiser FIR filter design.
    See scipy.signal.kaiserord for more information.
    """

    pass_zero: bool | str = True
    """
    If True, the gain at the frequency 0 (i.e., the “DC gain”) is 1. If False, the DC gain is 0. Can also
    be a string argument for the desired filter type (equivalent to btype in IIR design functions).
    {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
    """

    wn_hz: bool = True
    """
    Set False if cutoff and width are normalized from 0 to 1, where 1 is the Nyquist frequency
    """


def kaiser_design_fun(
    fs: float,
    cutoff: float | npt.ArrayLike | None = None,
    ripple: float | None = None,
    width: float | None = None,
    pass_zero: bool | str = True,
    wn_hz: bool = True,
) -> BACoeffs | None:
    """
    Design an `order`th-order FIR Kaiser filter and return the filter coefficients.
    See :obj:`FIRFilterSettings` for argument description.

    Returns:
        The filter taps as designed by firwin
    """
    if ripple is None or width is None or cutoff is None:
        return None

    width = width / (0.5 * fs) if wn_hz else width
    n_taps, beta = scipy.signal.kaiserord(ripple, width)
    if n_taps % 2 == 0:
        n_taps += 1
    taps = scipy.signal.firwin(
        numtaps=n_taps,
        cutoff=cutoff,
        window=("kaiser", beta),  # type: ignore
        pass_zero=pass_zero,  # type: ignore
        scale=False,
        fs=fs if wn_hz else None,
    )

    return (taps, np.array([1.0]))


class KaiserFilterTransformer(FilterByDesignTransformer[KaiserFilterSettings, BACoeffs]):
    def get_design_function(
        self,
    ) -> typing.Callable[[float], BACoeffs | None]:
        return functools.partial(
            kaiser_design_fun,
            cutoff=self.settings.cutoff,
            ripple=self.settings.ripple,
            width=self.settings.width,
            pass_zero=self.settings.pass_zero,
            wn_hz=self.settings.wn_hz,
        )


class KaiserFilter(BaseFilterByDesignTransformerUnit[KaiserFilterSettings, KaiserFilterTransformer]):
    SETTINGS = KaiserFilterSettings
