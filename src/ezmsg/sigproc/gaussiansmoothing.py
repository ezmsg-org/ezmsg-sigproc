from typing import Callable
import warnings

import numpy as np

from .filter import (
    FilterBaseSettings,
    BACoeffs,
    FilterByDesignTransformer,
    BaseFilterByDesignTransformerUnit
)

class GaussianSmoothingSettings(FilterBaseSettings):

    sigma: float | None = 1.0
    """
    sigma : float
        Standard deviation of the Gaussian kernel.
    """

    # dimensions: int | None = 1
    """
    Kernel dimension. Either 1D, 2D, or 3D.
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
    #dims: int = 1,
    width: int = 4,
    kernel_size: int | None = None,
    coef_type: str = "ba"
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

    kernel_sequence = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1) # default spacing of 1.0 -> change to np.linspace to adjust spacing <1.0
    
    b = np.exp(-1/2 * (kernel_sequence**2) / (sigma**2))
    b /= np.sum(b)

    a = np.array([1.0]) # default for gaussian window

    if coef_type == "ba":
        return (b, a)  # Return as tuple, not BACoeffs(b=b, a=a)
    return (np.array([1.0]), np.array([1.0]))  # Return as tuple for non-ba case

class GaussianSmoothingFilterTransformer(
    FilterByDesignTransformer[GaussianSmoothingSettings, BACoeffs]
):
    def get_design_function(
            self,
    ) -> Callable[[float], BACoeffs]:
        # Create a wrapper function that ignores fs parameter since gaussian smoothing doesn't need it
        def design_wrapper(fs: float) -> BACoeffs:
            return gaussian_smoothing_filter_design(
                sigma=self.settings.sigma,
                width=self.settings.width,
                kernel_size=self.settings.kernel_size,
                coef_type=self.settings.coef_type
            )
        return design_wrapper
    

class GaussianSmoothingFilter(
    BaseFilterByDesignTransformerUnit[
        GaussianSmoothingSettings, GaussianSmoothingFilterTransformer
        ]
):
    SETTINGS = GaussianSmoothingSettings

def gaussian_smoothing_filter(
    axis:str | None,
    sigma: float = 1.0,
    #dims: int = 1,
    width: int = 4,
    kernel_size: int | None = None,
    coef_type: str = "ba",
) -> GaussianSmoothingFilterTransformer:
    
    return GaussianSmoothingFilterTransformer(
        GaussianSmoothingSettings(
            axis=axis,
            sigma=sigma,
            #dims=dims,
            width=width,
            kernel_size=kernel_size,
            coef_type=coef_type
        )
    )