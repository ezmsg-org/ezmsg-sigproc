"""Digitize floating point signal data into signed integer samples."""

import enum

import ezmsg.core as ez
import numpy as np
from array_api_compat import get_namespace
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class DigitizeDType(str, enum.Enum):
    """Supported output dtypes for :obj:`DigitizeTransformer`."""

    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"

    @classmethod
    def options(cls) -> list[str]:
        return [dtype.value for dtype in cls]


_DTYPE_INFO = {
    DigitizeDType.INT16: np.iinfo(np.int16),
    DigitizeDType.INT32: np.iinfo(np.int32),
    DigitizeDType.INT64: np.iinfo(np.int64),
}

_INT64_CAST_MAX = np.nextafter(np.float64(np.iinfo(np.int64).max), 0.0)


class DigitizeSettings(ez.Settings):
    min_val: float
    """Input value that maps to the minimum representable value of ``dtype``."""

    max_val: float
    """Input value that maps to the maximum representable value of ``dtype``."""

    dtype: str | DigitizeDType = DigitizeDType.INT16
    """Signed integer output dtype. Supported values are ``"int16"``, ``"int32"``, and ``"int64"``."""


def _resolve_dtype(dtype: str | DigitizeDType) -> DigitizeDType:
    if isinstance(dtype, DigitizeDType):
        return dtype

    dtype_name = dtype.lower()
    try:
        return DigitizeDType(dtype_name)
    except ValueError as exc:
        raise ValueError(f"Unrecognized digitize dtype {dtype!r}. Must be one of {DigitizeDType.options()}") from exc


class DigitizeTransformer(BaseTransformer[DigitizeSettings, AxisArray, AxisArray]):
    def _process(
        self,
        message: AxisArray,
    ) -> AxisArray:
        if self.settings.max_val <= self.settings.min_val:
            raise ValueError("DigitizeSettings.max_val must be greater than min_val")

        dtype = _resolve_dtype(self.settings.dtype)
        dtype_info = _DTYPE_INFO[dtype]
        xp = get_namespace(message.data)
        target_dtype = getattr(xp, dtype.value)

        input_range = self.settings.max_val - self.settings.min_val
        output_min = dtype_info.min
        output_span = dtype_info.max - dtype_info.min

        data = xp.clip(message.data, self.settings.min_val, self.settings.max_val)
        data = (data - self.settings.min_val) / input_range
        data = xp.round(data * output_span + output_min)

        if dtype is DigitizeDType.INT64:
            high_mask = data >= _INT64_CAST_MAX
            data = xp.clip(data, dtype_info.min, _INT64_CAST_MAX).astype(target_dtype)
            data = xp.where(high_mask, xp.asarray(dtype_info.max, dtype=target_dtype), data)
        else:
            data = data.astype(target_dtype)

        return replace(message, data=data)


class Digitize(BaseTransformerUnit[DigitizeSettings, AxisArray, AxisArray, DigitizeTransformer]):
    SETTINGS = DigitizeSettings


def digitize(
    min_val: float,
    max_val: float,
    dtype: str | DigitizeDType = DigitizeDType.INT16,
) -> DigitizeTransformer:
    """
    Digitize floating point signal data into a signed integer dtype.

    Args:
        min_val: Input value that maps to the minimum representable value of ``dtype``.
        max_val: Input value that maps to the maximum representable value of ``dtype``.
        dtype: Signed integer output dtype: ``"int16"``, ``"int32"``, or ``"int64"``.

    Returns:
        :obj:`DigitizeTransformer`.
    """
    return DigitizeTransformer(DigitizeSettings(min_val=min_val, max_val=max_val, dtype=dtype))
