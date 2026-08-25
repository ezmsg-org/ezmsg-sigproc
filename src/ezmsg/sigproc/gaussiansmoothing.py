"""Gaussian kernel smoothing filter."""

import warnings
from typing import Callable

import numpy as np

from .filter import (
    BACoeffs,
    BaseFilterByDesignTransformerUnit,
    FilterBaseSettings,
    FilterByDesignTransformer,
)


class GaussianSmoothingSettings(FilterBaseSettings):
    sigma: float | None = 0.01
    """
    sigma : float
        Standard deviation of the Gaussian kernel, in **seconds**. Converted
        to samples using the sampling rate of the first message.
        The -3 dB corner frequency is sqrt(ln 2) / (2 * pi * sigma); the
        default of 0.01 s is equivalent to a ~13.2 Hz low-pass.
    """

    width: int | None = 4
    """
    width : int
        Number of standard deviations covered by the kernel window if kernel_size is not provided.
    """

    kernel_size: int | None = None
    """
    kernel_size : int | None
        Length of the kernel in samples. If provided, overrides automatic calculation.
        In causal mode this is the number of *causal* taps, i.e. the kernel spans
        ``kernel_size`` samples into the past rather than ``kernel_size // 2``.
    """

    causal: bool = False
    """
    causal : bool
        If False (default), the kernel is a symmetric Gaussian of
        ``2 * width * sigma + 1`` taps. Filtering is applied causally (``lfilter``),
        so the acausal half of the kernel manifests purely as group delay of
        ``(kernel_size - 1) / 2 == width * sigma`` samples -- 4 * sigma at the
        default ``width=4``.

        If True, the kernel is the causal half of that Gaussian (peak at lag 0,
        tail extending only into the past), renormalized to unit sum. Its group
        delay is the centroid of a half-Gaussian, ``sigma * sqrt(2 / pi)``
        (~0.8 * sigma), i.e. roughly a factor of 5 less lag than the symmetric
        kernel at the same sigma.

        The two modes are **not** interchangeable at equal sigma: halving the
        kernel also halves the effective averaging window, so the causal kernel
        smooths less and its stopband rolls off less steeply (-12 dB/octave
        versus the symmetric kernel's much sharper Gaussian rolloff) for a given
        sigma. Compare them at matched white-noise variance reduction
        (``sum(b ** 2)``) rather than at matched sigma; on that footing the
        causal kernel reaches the same noise gain at roughly a third of the lag.
        For example, at 100 Hz a symmetric sigma of 20 ms gives a noise gain of
        0.141 for 80 ms of delay, while a causal sigma of 38 ms gives the same
        0.141 for 27 ms.
    """


def gaussian_smoothing_filter_design(
    sigma: float = 1.0,
    width: int = 4,
    kernel_size: int | None = None,
    causal: bool = False,
) -> BACoeffs | None:
    """Design a normalized Gaussian FIR kernel. ``sigma`` is in **samples**;
    callers with a time-domain sigma must scale by the sampling rate first.

    If ``causal`` is True, only the causal half of the Gaussian is kept -- the
    peak sits at lag 0 and the tail extends into the past -- and ``kernel_size``
    counts causal taps. See :class:`GaussianSmoothingSettings` for the group
    delay of each mode.
    """
    # Parameter checks
    if sigma <= 0:
        raise ValueError(f"sigma must be positive. Received: {sigma}")

    if width <= 0:
        raise ValueError(f"width must be positive. Received: {width}")

    # A symmetric kernel spans ``width`` sigmas either side of the peak; a causal
    # kernel spans them on the past side only.
    expected_kernel_size = int(width * sigma + 1) if causal else int(2 * width * sigma + 1)

    if kernel_size is not None:
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1. Received: {kernel_size}")
    else:
        kernel_size = expected_kernel_size

    # Warn if kernel_size is smaller than recommended but don't fail
    if kernel_size < expected_kernel_size:
        ## TODO: Either add a warning or determine appropriate kernel size and raise an error
        warnings.warn(
            f"Provided kernel_size {kernel_size} is smaller than recommended "
            f"size {expected_kernel_size} for sigma={sigma}, width={width} and "
            f"causal={causal}. The kernel may be truncated."
        )

    if kernel_size == 1:
        warnings.warn(
            f"kernel_size=1 (sigma={sigma} samples, width={width}) yields an "
            "identity (single-tap) kernel: no smoothing will be performed."
        )

    from scipy.signal.windows import gaussian

    if causal:
        # Take the peak and everything to its right from a symmetric kernel of
        # 2 * kernel_size - 1 taps, then reverse-normalize: lfilter convolves
        # b[0] with the newest sample, so index 0 is the peak (lag 0) and
        # increasing index reaches further into the past.
        b = gaussian(2 * kernel_size - 1, std=sigma)[kernel_size - 1 :]
    else:
        b = gaussian(kernel_size, std=sigma)
    b = b / np.sum(b)  # Ensure normalization
    a = np.array([1.0])

    return b, a


class GaussianSmoothingFilterTransformer(FilterByDesignTransformer[GaussianSmoothingSettings, BACoeffs]):
    def get_design_function(
        self,
    ) -> Callable[[float], BACoeffs | None]:
        def design_wrapper(fs: float) -> BACoeffs | None:
            if (
                self.settings.sigma is None
                or self.settings.sigma <= 0
                or self.settings.width is None
                or self.settings.width <= 0
                or (self.settings.kernel_size is not None and self.settings.kernel_size <= 1)
            ):
                return None
            return gaussian_smoothing_filter_design(
                sigma=self.settings.sigma * fs,  # settings.sigma is in seconds
                width=self.settings.width,
                kernel_size=self.settings.kernel_size,
                causal=self.settings.causal,
            )

        return design_wrapper


class GaussianSmoothingFilter(
    BaseFilterByDesignTransformerUnit[GaussianSmoothingSettings, GaussianSmoothingFilterTransformer]
):
    SETTINGS = GaussianSmoothingSettings
