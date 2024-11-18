import functools
import typing

import scipy.signal

from .filter import (
    FilterBaseSettings,
    FilterCoefsMultiType,
    FilterBase,
)


class ChebyshevFilterSettings(FilterBaseSettings):
    """Settings for :obj:`ButterworthFilter`."""

    order: int = 0
    """
    Filter order
    """

    ripple_tol: typing.Optional[float] = None
    """
    The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
    """

    Wn: typing.Optional[typing.Union[float, typing.Tuple[float, float]]] = None
    """
    A scalar or length-2 sequence giving the critical frequencies.
    For Type I filters, this is the point in the transition band at which the gain first drops below -rp.
    For digital filters, Wn are in the same units as fs.
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


def cheby_design_fun(
    fs: float,
    order: int = 0,
    ripple_tol: typing.Optional[float] = None,
    Wn: typing.Optional[typing.Union[float, typing.Tuple[float, float]]] = None,
    btype: str = "lowpass",
    analog: bool = False,
    coef_type: str = "ba",
    cheby_type: str = "cheby1",
) -> typing.Optional[FilterCoefsMultiType]:
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
                fs=fs,
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
    return coefs


class ChebyshevFilter(FilterBase):
    SETTINGS = ChebyshevFilterSettings

    def design_filter(
        self,
    ) -> typing.Callable[[float], typing.Optional[FilterCoefsMultiType]]:
        return functools.partial(
            cheby_design_fun,
            order=self.SETTINGS.order,
            ripple_tol=self.SETTINGS.ripple_tol,
            Wn=self.SETTINGS.Wn,
            btype=self.SETTINGS.btype,
            analog=self.SETTINGS.analog,
            coef_type=self.SETTINGS.coef_type,
            cheby_type=self.SETTINGS.cheby_type,
        )