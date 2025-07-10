import functools
from typing import Callable

import numpy as np

from .filter import (
    FilterSettings,
    BACoeffs,
    FilterByDesignTransformer,
    BaseFilterByDesignTransformerUnit
)

class GaussianSmoothingSettings(FilterSettings):

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
    
    if kernel_size is None:
        kernel_size = int(2 * width * sigma + 1)
    kernel_sequence = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1) # default spacing of 1.0 -> change to np.linspace to adjust spacing <1.0
    
    b = np.exp(-1/2 * (kernel_sequence**2) / (sigma**2))
    b /= np.sum(b)

    a = [1.0] # default for gaussian window

    if coef_type == "ba":
        return BACoeffs(b=b, a=a)
    return BACoeffs(b=1.0, a=1.0)

class GaussianSmoothingFilterTransformer(
    FilterByDesignTransformer[GaussianSmoothingSettings, BACoeffs]
):
    def get_design_function(
            self,
    ) -> Callable[[int], BACoeffs]:
        return functools.partial(
            gaussian_smoothing_filter_design,
            sigma = self.settings.sigma,
            width = self.settings.width,
            coef_type = self.settings.coef_type
        )
    

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