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
    """


def gaussian_smoothing_filter_design(
    sigma: float = 1.0,
    width: int = 4,
    kernel_size: int | None = None,
) -> BACoeffs | None:
    """Design a normalized Gaussian FIR kernel. ``sigma`` is in **samples**;
    callers with a time-domain sigma must scale by the sampling rate first."""
    # Parameter checks
    if sigma <= 0:
        raise ValueError(f"sigma must be positive. Received: {sigma}")

    if width <= 0:
        raise ValueError(f"width must be positive. Received: {width}")

    if kernel_size is not None:
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1. Received: {kernel_size}")
    else:
        kernel_size = int(2 * width * sigma + 1)

    # Warn if kernel_size is smaller than recommended but don't fail
    expected_kernel_size = int(2 * width * sigma + 1)
    if kernel_size < expected_kernel_size:
        ## TODO: Either add a warning or determine appropriate kernel size and raise an error
        warnings.warn(
            f"Provided kernel_size {kernel_size} is smaller than recommended "
            f"size {expected_kernel_size} for sigma={sigma} and width={width}. "
            "The kernel may be truncated."
        )

    if kernel_size == 1:
        warnings.warn(
            f"kernel_size=1 (sigma={sigma} samples, width={width}) yields an "
            "identity (single-tap) kernel: no smoothing will be performed."
        )

    from scipy.signal.windows import gaussian

    b = gaussian(kernel_size, std=sigma)
    b /= np.sum(b)  # Ensure normalization
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
            )

        return design_wrapper


class GaussianSmoothingFilter(
    BaseFilterByDesignTransformerUnit[GaussianSmoothingSettings, GaussianSmoothingFilterTransformer]
):
    SETTINGS = GaussianSmoothingSettings
