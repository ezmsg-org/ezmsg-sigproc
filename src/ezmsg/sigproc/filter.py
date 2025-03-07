from abc import abstractmethod, ABC
from dataclasses import dataclass, field
import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
import numpy as np
import numpy.typing as npt
import scipy.signal

from ezmsg.sigproc.base import (
    processor_state,
    BaseStatefulTransformer,
    BaseTransformerUnit,
    SettingsType,
)


@dataclass
class FilterCoefficients:
    b: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    a: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))


# Type aliases
BACoeffs = tuple[npt.NDArray, npt.NDArray]
SOSCoeffs = npt.NDArray
FilterCoefsType = typing.TypeVar("FilterCoefsType", BACoeffs, SOSCoeffs)


def _normalize_coefs(
    coefs: FilterCoefficients | tuple[npt.NDArray, npt.NDArray] | npt.NDArray,
) -> tuple[str, tuple[npt.NDArray, ...]]:
    coef_type = "ba"
    if coefs is not None:
        # scipy.signal functions called with first arg `*coefs`.
        # Make sure we have a tuple of coefficients.
        if isinstance(coefs, npt.NDArray):
            coef_type = "sos"
            coefs = (coefs,)  # sos funcs just want a single ndarray.
        elif isinstance(coefs, FilterCoefficients):
            coefs = (FilterCoefficients.b, FilterCoefficients.a)
    return coef_type, coefs


class FilterBaseSettings(ez.Settings):
    axis: str | None = None
    """The name of the axis to operate on."""

    coef_type: str = "ba"
    """The type of filter coefficients. One of "ba" or "sos"."""


class FilterSettings(FilterBaseSettings):
    coefs: FilterCoefficients | None = None
    """The pre-calculated filter coefficients."""

    # Note: coef_type = "ba" is assumed for this class.


@processor_state
class FilterState:
    zi: npt.NDArray | None = None


class FilterTransformer(
    BaseStatefulTransformer[FilterSettings, AxisArray, AxisArray, FilterState]
):
    """
    Filter data using the provided coefficients.
    """

    def __call__(self, message: AxisArray) -> AxisArray:
        if self.settings.coefs is None:
            return message
        return super().__call__(message)

    def _hash_message(self, message: AxisArray) -> int:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        axis_idx = message.get_axis_idx(axis)
        samp_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        return hash((message.key, samp_shape))

    def _reset_state(self, message: AxisArray) -> None:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        axis_idx = message.get_axis_idx(axis)
        n_tail = message.data.ndim - axis_idx - 1
        coefs = (
            (self.settings.coefs,)
            if self.settings.coefs is not None
            and not isinstance(self.settings.coefs, tuple)
            else self.settings.coefs
        )
        zi_func = {"ba": scipy.signal.lfilter_zi, "sos": scipy.signal.sosfilt_zi}[
            self.settings.coef_type
        ]
        zi = zi_func(*coefs)
        zi_expand = (None,) * axis_idx + (slice(None),) + (None,) * n_tail
        n_tile = (
            message.data.shape[:axis_idx] + (1,) + message.data.shape[axis_idx + 1 :]
        )

        if self.settings.coef_type == "sos":
            zi_expand = (slice(None),) + zi_expand
            n_tile = (1,) + n_tile

        self.state.zi = np.tile(zi[zi_expand], n_tile)

    def _process(self, message: AxisArray) -> AxisArray:
        if message.data.size > 0:
            axis = message.dims[0] if self.settings.axis is None else self.settings.axis
            axis_idx = message.get_axis_idx(axis)
            coefs = (
                (self.settings.coefs,)
                if self.settings.coefs is not None
                and not isinstance(self.settings.coefs, tuple)
                else self.settings.coefs
            )
            filt_func = {"ba": scipy.signal.lfilter, "sos": scipy.signal.sosfilt}[
                self.settings.coef_type
            ]
            dat_out, self.state.zi = filt_func(
                *coefs, message.data, axis=axis_idx, zi=self.state.zi
            )
        else:
            dat_out = message.data

        return replace(message, data=dat_out)


class Filter(
    BaseTransformerUnit[FilterSettings, AxisArray, AxisArray, FilterTransformer]
):
    SETTINGS = FilterSettings


def filtergen(
    axis: str, coefs: npt.NDArray | tuple[npt.NDArray] | None, coef_type: str
) -> FilterTransformer:
    """
    Filter data using the provided coefficients.

    Returns:
        :obj:`FilterTransformer`.
    """
    return FilterTransformer(
        FilterSettings(axis=axis, coefs=coefs, coef_type=coef_type)
    )


@processor_state
class FilterByDesignState:
    filter: FilterTransformer | None = None


class FilterByDesignTransformer(
    BaseStatefulTransformer[SettingsType, AxisArray, AxisArray, FilterByDesignState],
    ABC,
    typing.Generic[SettingsType, FilterCoefsType],
):
    """Abstract base class for filter design transformers."""

    @classmethod
    def get_message_type(cls, _) -> typing.Type[AxisArray]:
        return AxisArray

    @abstractmethod
    def get_design_function(self) -> typing.Callable[[float], FilterCoefsType | None]:
        """Return a function that takes sampling frequency and returns filter coefficients."""
        ...

    def __call__(self, message: AxisArray) -> AxisArray:
        # Offer a shortcut when there is no design function or order is 0.
        if hasattr(self.settings, "order") and not self.settings.order:
            return message
        design_fun = self.get_design_function()
        if design_fun is None:
            return message
        return super().__call__(message)

    def _hash_message(self, message: AxisArray) -> int:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        gain = message.axes[axis].gain if hasattr(message.axes[axis], "gain") else 1
        axis_idx = message.get_axis_idx(axis)
        samp_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        return hash((message.key, samp_shape, gain))

    def _reset_state(self, message: AxisArray) -> None:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        design_fun = self.get_design_function()
        coefs = design_fun(1 / message.axes[axis].gain)
        self.state.filter = filtergen(axis, coefs, self.settings.coef_type)

    def _process(self, message: AxisArray) -> AxisArray:
        return self.state.filter(message)
