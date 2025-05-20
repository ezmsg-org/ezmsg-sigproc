import functools
import typing
import numpy as np
import scipy.signal
from scipy.signal import normalize

from .filter import (
    FilterBaseSettings,
    FilterByDesignTransformer,
    BACoeffs,
    SOSCoeffs,
    BaseFilterByDesignTransformerUnit,
)


class CombFilterSettings(FilterBaseSettings):
    """Settings for :obj:`CombFilter`."""

    # axis and coef_type are inherited from FilterBaseSettings

    fundamental_freq: float = 60.0
    """
    Fundamental frequency in Hz
    """

    num_harmonics: int = 3
    """
    Number of harmonics to include (including fundamental)
    """

    q_factor: float = 35.0
    """
    Quality factor (Q) for each peak/notch
    """

    filter_type: str = "notch"
    """
    Type of comb filter: 'notch' removes harmonics, 'peak' passes harmonics at the expense of others.
    """

    quality_scaling: str = "constant"
    """
    'constant': same quality for all harmonics results in wider bands at higher frequencies,
    'proportional': quality proportional to frequency results in constant bandwidths.
    """


def comb_design_fun(
    fs: float,
    fundamental_freq: float = 60.0,
    num_harmonics: int = 3,
    q_factor: float = 35.0,
    filter_type: str = "notch",
    coef_type: str = "sos",
    quality_scaling: str = "constant",
) -> BACoeffs | SOSCoeffs | None:
    """
    Design a comb filter as cascaded second-order sections targeting a fundamental frequency and its harmonics.

    Returns:
        The filter coefficients as SOS (recommended) or (b, a) for finite precision stability.
    """
    if coef_type != "sos" and coef_type != "ba":
        raise ValueError("Comb filter only supports 'sos' or 'ba' coefficient types")

    # Generate all SOS sections
    all_sos = []

    for i in range(1, num_harmonics + 1):
        freq = fundamental_freq * i

        # Skip if frequency exceeds Nyquist
        if freq >= fs / 2:
            continue

        # Adjust Q factor based on scaling method
        current_q = q_factor
        if quality_scaling == "proportional":
            current_q = q_factor * i

        if filter_type == "notch":
            sos = scipy.signal.iirnotch(w0=freq, Q=current_q, fs=fs)
        else:  # peak filter
            sos = scipy.signal.iirpeak(w0=freq, Q=current_q, fs=fs)
        # Though .iirnotch and .iirpeak return b, a pairs, these are second order so
        #  we can use them directly as SOS sections.
        #  Check:
        # assert np.allclose(scipy.signal.tf2sos(sos[0], sos[1])[0], np.hstack(sos))

        all_sos.append(np.hstack(sos))

    if not all_sos:
        return None

    # Combine all SOS sections
    combined_sos = np.vstack(all_sos)

    if coef_type == "ba":
        # Convert to transfer function form
        b, a = scipy.signal.sos2tf(combined_sos)
        return normalize(b, a)

    return combined_sos


class CombFilterTransformer(
    FilterByDesignTransformer[CombFilterSettings, BACoeffs | SOSCoeffs]
):
    def get_design_function(
        self,
    ) -> typing.Callable[[float], BACoeffs | SOSCoeffs | None]:
        return functools.partial(
            comb_design_fun,
            fundamental_freq=self.settings.fundamental_freq,
            num_harmonics=self.settings.num_harmonics,
            q_factor=self.settings.q_factor,
            filter_type=self.settings.filter_type,
            coef_type=self.settings.coef_type,
            quality_scaling=self.settings.quality_scaling,
        )


class CombFilterUnit(
    BaseFilterByDesignTransformerUnit[CombFilterSettings, CombFilterTransformer]
):
    SETTINGS = CombFilterSettings


def comb(
    axis: str | None,
    fundamental_freq: float = 50.0,
    num_harmonics: int = 3,
    q_factor: float = 35.0,
    filter_type: str = "notch",
    coef_type: str = "sos",
    quality_scaling: str = "constant",
) -> CombFilterTransformer:
    """
    Create a comb filter for enhancing or removing a fundamental frequency and its harmonics.

    Args:
        axis: Axis to filter along
        fundamental_freq: Base frequency in Hz
        num_harmonics: Number of harmonic peaks/notches (including fundamental)
        q_factor: Quality factor for peak/notch width
        filter_type: 'notch' to remove or 'peak' to enhance harmonics
        coef_type: Coefficient type ('sos' recommended for stability)
        quality_scaling: How to handle bandwidths across harmonics

    Returns:
        CombFilterTransformer
    """
    return CombFilterTransformer(
        CombFilterSettings(
            axis=axis,
            fundamental_freq=fundamental_freq,
            num_harmonics=num_harmonics,
            q_factor=q_factor,
            filter_type=filter_type,
            coef_type=coef_type,
            quality_scaling=quality_scaling,
        )
    )
