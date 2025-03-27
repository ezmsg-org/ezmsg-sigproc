import functools
import typing

import scipy.signal
from scipy.signal import normalize

from .filter import (
    FilterBaseSettings,
    FilterByDesignTransformer,
    BACoeffs,
    SOSCoeffs,
    BaseFilterByDesignTransformerUnit,
)


class ChebyshevFilterSettings(FilterBaseSettings):
    """Settings for :obj:`ChebyshevFilter`."""

    # axis and coef_type are inherited from FilterBaseSettings

    order: int = 0
    """
    Filter order
    """

    ripple_tol: float | None = None
    """
    The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
    """

    Wn: float | tuple[float, float] | None = None
    """
    A scalar or length-2 sequence giving the critical frequencies.
    For Type I filters, this is the point in the transition band at which the gain first drops below -rp.
    For digital filters, Wn are in the same units as fs unless wn_hz is False.
    For analog filters, Wn is an angular frequency (e.g., rad/s).
    """

    btype: str = "lowpass"
    """
    {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
    """

    analog: bool = False
    """
    When True, return an analog filter, otherwise a digital filter is returned.
    """

    cheby_type: str = "cheby1"
    """
    Which type of Chebyshev filter to design. Either "cheby1" or "cheby2".
    """

    wn_hz: bool = True
    """
    Set False if provided Wn are normalized from 0 to 1, where 1 is the Nyquist frequency
    """


def cheby_design_fun(
    fs: float,
    order: int = 0,
    ripple_tol: float | None = None,
    Wn: float | tuple[float, float] | None = None,
    btype: str = "lowpass",
    analog: bool = False,
    coef_type: str = "ba",
    cheby_type: str = "cheby1",
    wn_hz: bool = True,
) -> BACoeffs | SOSCoeffs | None:
    """
    Chebyshev type I and type II digital and analog filter design.
    Design an `order`th-order digital or analog Chebyshev type I or type II filter and return the filter coefficients.
    See :obj:`ChebyFilterSettings` for argument description.

    Returns:
        The filter coefficients as a tuple of (b, a) for coef_type "ba", or as a single ndarray for "sos",
        or (z, p, k) for "zpk".
    """
    coefs = None
    if order > 0:
        if cheby_type == "cheby1":
            coefs = scipy.signal.cheby1(
                order,
                ripple_tol,
                Wn,
                btype=btype,
                analog=analog,
                output=coef_type,
                fs=fs if wn_hz else None,
            )
        elif cheby_type == "cheby2":
            coefs = scipy.signal.cheby2(
                order,
                ripple_tol,
                Wn,
                btype=btype,
                analog=analog,
                output=coef_type,
                fs=fs,
            )
    if coefs is not None and coef_type == "ba":
        coefs = normalize(*coefs)
    return coefs


class ChebyshevFilterTransformer(
    FilterByDesignTransformer[ChebyshevFilterSettings, BACoeffs | SOSCoeffs]
):
    def get_design_function(
        self,
    ) -> typing.Callable[[float], BACoeffs | SOSCoeffs | None]:
        return functools.partial(
            cheby_design_fun,
            order=self.settings.order,
            ripple_tol=self.settings.ripple_tol,
            Wn=self.settings.Wn,
            btype=self.settings.btype,
            analog=self.settings.analog,
            coef_type=self.settings.coef_type,
            cheby_type=self.settings.cheby_type,
            wn_hz=self.settings.wn_hz,
        )


class ChebyshevFilter(
    BaseFilterByDesignTransformerUnit[
        ChebyshevFilterSettings, ChebyshevFilterTransformer
    ]
):
    SETTINGS = ChebyshevFilterSettings
