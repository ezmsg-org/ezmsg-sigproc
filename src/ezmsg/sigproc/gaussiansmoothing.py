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
    sigma: float | None = 1.0
    """
    sigma : float
        Standard deviation of the Gaussian kernel.
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

    from scipy.signal.windows import gaussian

    b = gaussian(kernel_size, std=sigma)
    b /= np.sum(b)  # Ensure normalization
    a = np.array([1.0])

    return b, a


class GaussianSmoothingFilterTransformer(FilterByDesignTransformer[GaussianSmoothingSettings, BACoeffs]):
    def get_design_function(
        self,
    ) -> Callable[[float], BACoeffs]:
        # Create a wrapper function that ignores fs parameter since gaussian smoothing doesn't need it
        def design_wrapper(fs: float) -> BACoeffs:
            return gaussian_smoothing_filter_design(
                sigma=self.settings.sigma,
                width=self.settings.width,
                kernel_size=self.settings.kernel_size,
            )

        return design_wrapper


class GaussianSmoothingFilter(
    BaseFilterByDesignTransformerUnit[GaussianSmoothingSettings, GaussianSmoothingFilterTransformer]
):
    SETTINGS = GaussianSmoothingSettings
