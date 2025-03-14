import functools
import typing

import scipy.signal
from scipy.signal import normalize

from .filter import (
    FilterBaseSettings,
    BACoeffs,
    SOSCoeffs,
    FilterByDesignTransformer,
    BaseFilterByDesignTransformerUnit,
)


class ButterworthFilterSettings(FilterBaseSettings):
    """Settings for :obj:`ButterworthFilter`."""

    # axis and coef_type are inherited from FilterBaseSettings

    order: int = 0
    """
    Filter order
    """

    cuton: float | None = None
    """
    Cuton frequency (Hz). If `cutoff` is not specified then this is the highpass corner. Otherwise,
    if this is lower than `cutoff` then this is the beginning of the bandpass
    or if this is greater than `cutoff` then this is the end of the bandstop. 
    """

    cutoff: float | None = None
    """
    Cutoff frequency (Hz). If `cuton` is not specified then this is the lowpass corner. Otherwise,
    if this is greater than `cuton` then this is the end of the bandpass,
    or if this is less than `cuton` then this is the beginning of the bandstop. 
    """

    wn_hz: bool = True
    """
    Set False if provided Wn are normalized from 0 to 1, where 1 is the Nyquist frequency
    """

    def filter_specs(
        self,
    ) -> tuple[str, float | tuple[float, float]] | None:
        """
        Determine the filter type given the corner frequencies.

        Returns:
            A tuple with the first element being a string indicating the filter type
            (one of "lowpass", "highpass", "bandpass", "bandstop")
            and the second element being the corner frequency or frequencies.

        """
        if self.cuton is None and self.cutoff is None:
            return None
        elif self.cuton is None and self.cutoff is not None:
            return "lowpass", self.cutoff
        elif self.cuton is not None and self.cutoff is None:
            return "highpass", self.cuton
        elif self.cuton is not None and self.cutoff is not None:
            if self.cuton <= self.cutoff:
                return "bandpass", (self.cuton, self.cutoff)
            else:
                return "bandstop", (self.cutoff, self.cuton)


def butter_design_fun(
    fs: float,
    order: int = 0,
    cuton: float | None = None,
    cutoff: float | None = None,
    coef_type: str = "ba",
    wn_hz: bool = True,
) -> BACoeffs | SOSCoeffs | None:
    """
    See :obj:`ButterworthFilterSettings.filter_specs` for an explanation of specifying different
    filter types (lowpass, highpass, bandpass, bandstop) from the parameters.
    You are likely to want to use this function with :obj:`filter_by_design`, which only passes `fs` to the design
    function (this), meaning that you should wrap this function with a lambda or prepare with functools.partial.

    Args:
        fs: The sampling frequency of the data in Hz.
        order: Filter order.
        cuton: Corner frequency of the filter in Hz.
        cutoff: Corner frequency of the filter in Hz.
        coef_type: "ba", "sos", or "zpk"
        wn_hz: Set False if provided Wn are normalized from 0 to 1, where 1 is the Nyquist frequency

    Returns:
        The filter coefficients as a tuple of (b, a) for coef_type "ba", or as a single ndarray for "sos",
        or (z, p, k) for "zpk".

    """
    coefs = None
    if order > 0:
        btype, cutoffs = ButterworthFilterSettings(
            order=order, cuton=cuton, cutoff=cutoff
        ).filter_specs()
        coefs = scipy.signal.butter(
            order,
            Wn=cutoffs,
            btype=btype,
            fs=fs if wn_hz else None,
            output=coef_type,
        )
    if coefs is not None and coef_type == "ba":
        coefs = normalize(*coefs)
    return coefs


class ButterworthFilterTransformer(
    FilterByDesignTransformer[ButterworthFilterSettings, BACoeffs | SOSCoeffs]
):
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


class ButterworthFilter(
    BaseFilterByDesignTransformerUnit[
        ButterworthFilterSettings, ButterworthFilterTransformer
    ]
):
    SETTINGS = ButterworthFilterSettings


def butter(
    axis: str | None,
    order: int = 0,
    cuton: float | None = None,
    cutoff: float | None = None,
    coef_type: str = "ba",
    wn_hz: bool = True,
) -> ButterworthFilterTransformer:
    """
    Convenience generator wrapping filter_gen_by_design for Butterworth filters.
    Apply Butterworth filter to streaming data. Uses :obj:`scipy.signal.butter` to design the filter.
    See :obj:`ButterworthFilterSettings.filter_specs` for an explanation of specifying different
    filter types (lowpass, highpass, bandpass, bandstop) from the parameters.

    Returns:
        :obj:`ButterworthFilterTransformer`
    """
    return ButterworthFilterTransformer(
        ButterworthFilterSettings(
            axis=axis,
            order=order,
            cuton=cuton,
            cutoff=cutoff,
            coef_type=coef_type,
            wn_hz=wn_hz,
        )
    )
