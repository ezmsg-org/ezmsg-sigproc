import functools
import typing

import ezmsg.core as ez
import numpy as np
import scipy.signal

from ezmsg.sigproc.filter import (
    BACoeffs,
    BaseFilterByDesignTransformerUnit,
    FilterBaseSettings,
    FilterByDesignTransformer,
)


class ParksMcClellanFIRSettings(FilterBaseSettings):
    """Settings for :obj:`ParksMcClellanFIR`."""

    # axis inherited from FilterBaseSettings

    coef_type: str = "ba"
    """
    Coefficient type. Must be 'ba' for FIR.
    """

    order: int = 0
    """
    Filter order (taps = order + 1).
    PMC FIR filters require even order (odd taps).
    If odd order (even taps), order will be incremented by 1.
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

    transition: float = 10.0
    """
    Transition bandwidth (Hz) applied to each passband edge.
    For low/high: single transition. For bands: both edges.
    """

    weight_pass: float = 1.0
    """
    Weight for the passband.
    Used for both high and low passbands in bandstop filters.
    """

    weight_stop_lo: float = 1.0
    """
    Weight for the lower stopband.
    Not used for bandstop filters.
    """

    weight_stop_hi: float = 1.0
    """
    Weight for the upper stopband.
    Used as the central-stop weight for bandstop filters.
    """

    def filter_specs(
        self,
    ) -> tuple[str, tuple[float, float] | float] | None:
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


def parks_mcclellan_design_fun(
    fs: float,
    order: int = 0,
    cuton: float | None = None,
    cutoff: float | None = None,
    transition: float = 10.0,
    weight_pass: float = 1.0,
    weight_stop_lo: float = 1.0,
    weight_stop_hi: float = 1.0,
) -> BACoeffs | None:
    """
    See :obj:`ParksMcClellanFIRSettings.filter_specs` for an explanation of specifying different
    filter types (lowpass, highpass, bandpass, bandstop) from the parameters.

    Designs a Parks-McClellan FIR filter via the Remez exchange algorithm using the given specifications.
    PMC filters are equiripple and linear phase.

    You are likely to want to use this function with :obj:`filter_by_design`, which only passes `fs` to the design
    function (this), meaning that you should wrap this function with a lambda or prepare with functools.partial.

    Args:
        fs: The sampling frequency of the data in Hz.
        order: Filter order.
        cuton: Corner frequency of the filter in Hz.
        cutoff: Corner frequency of the filter in Hz.
        transition: Transition bandwidth (Hz) applied to each passband edge.
        weight_pass: Weight for the passband.
        weight_stop_lo: Weight for the lower stopband.
        weight_stop_hi: Weight for the upper stopband.

    Returns:
        The filter coefficients as a tuple of (b, a).
    """
    if order <= 0:
        return None
    if order % 2 == 1:
        order += 1

    specs = ParksMcClellanFIRSettings(cuton=cuton, cutoff=cutoff).filter_specs()
    if specs is None:
        # Under-specified: no filter
        return None

    btype, corners = specs
    nyq = fs / 2.0
    tw = max(transition, 0.0)

    def clip_hz(x: float) -> float:
        return float(min(max(x, 0.0), nyq))

    if btype == "lowpass":
        b = [0.0, clip_hz(corners), clip_hz(corners + tw), nyq]
        d = [1.0, 0.0]
        w = [max(weight_pass, 0.0), max(weight_stop_hi, 0.0)]

    elif btype == "highpass":
        b = [0.0, clip_hz(corners - tw), clip_hz(corners), nyq]
        d = [0.0, 1.0]
        w = [max(weight_stop_lo, 0.0), max(weight_pass, 0.0)]

    elif btype == "bandpass":
        b = [
            0.0,
            clip_hz(corners[0] - tw),
            clip_hz(corners[0]),
            clip_hz(corners[1]),
            clip_hz(corners[1] + tw),
            nyq,
        ]
        d = [0.0, 1.0, 0.0]
        w = [max(weight_stop_lo, 0.0), max(weight_pass, 0.0), max(weight_stop_hi, 0.0)]

    else:
        b = [
            0.0,
            clip_hz(corners[0]),
            clip_hz(corners[0] + tw),
            clip_hz(corners[1] - tw),
            clip_hz(corners[1]),
            nyq,
        ]
        d = [1.0, 0.0, 1.0]
        # For bandstop we can reuse stop_hi as central-stop weight; stop_lo is the DC-side passband stop weight
        w = [max(weight_pass, 0.0), max(weight_stop_hi, 0.0), max(weight_pass, 0.0)]

    # Ensure bands strictly increase and have nonzero width per segment
    # Adjust tiny overlaps due to clipping
    for i in range(1, len(b)):
        if b[i] <= b[i - 1]:
            b[i] = min(b[i - 1] + 1e-6, nyq)

    b = scipy.signal.remez(numtaps=order + 1, bands=b, desired=d, weight=w, fs=fs)
    return (b, np.array([1.0]))


class ParksMcClellanFIRTransformer(FilterByDesignTransformer[ParksMcClellanFIRSettings, BACoeffs]):
    def get_design_function(self) -> typing.Callable[[float], BACoeffs | None]:
        if self.settings.coef_type != "ba":
            ez.logger.error("ParksMcClellanFIR only supports coef_type='ba'.")
            raise ValueError("ParksMcClellanFIR only supports coef_type='ba'.")
        return functools.partial(
            parks_mcclellan_design_fun,
            order=self.settings.order,
            cuton=self.settings.cuton,
            cutoff=self.settings.cutoff,
            transition=self.settings.transition,
            weight_pass=self.settings.weight_pass,
            weight_stop_lo=self.settings.weight_stop_lo,
            weight_stop_hi=self.settings.weight_stop_hi,
        )


class ParksMcClellanFIR(BaseFilterByDesignTransformerUnit[ParksMcClellanFIRSettings, ParksMcClellanFIRTransformer]):
    SETTINGS = ParksMcClellanFIRSettings
