import enum

import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, replace

from .base import BaseTransformer, BaseTransformerUnit


class QuantizeSettings(ez.Settings):
    """
    Settings for the Quantizer.
    """

    max_val: float
    """
    Clip the data to this maximum value before quantization and map the [min_val max_val] range to the quantized range.
    """

    min_val: float = 0.0
    """
    Clip the data to this minimum value before quantization and map the [min_val max_val] range to the quantized range.
    Default: 0
    """

    bits: int = 8
    """
    Number of bits for quantization.
    Note: The data type will be integer of the next power of 2 greater than or equal to this value.
    Default: 8
    """



class QuantizeTransformer(BaseTransformer[QuantizeSettings, AxisArray, AxisArray]):
    def _process(
        self,
        message: AxisArray,
    ) -> AxisArray:
        # Determine appropriate integer type based on bits
        if self.settings.bits <= 1:
            dtype = bool
        elif self.settings.bits <= 8:
            dtype = np.uint8
        elif self.settings.bits <= 16:
            dtype = np.uint16
        elif self.settings.bits <= 32:
            dtype = np.uint32
        else:
            dtype = np.uint64

        data = message.data.clip(self.settings.min_val, self.settings.max_val)
        data = (data - self.settings.min_val) / (self.settings.max_val - self.settings.min_val)

        # Scale to the quantized range [0, 2^bits - 1]
        scale_factor = 2**self.settings.bits - 1
        data = np.round(data * scale_factor).astype(dtype)

        # Create a new AxisArray with the quantized data
        return replace(message, data=data)


class QuantizerUnit(
    BaseTransformerUnit[QuantizeSettings, AxisArray, AxisArray, QuantizeTransformer]
):
    SETTINGS = QuantizeSettings
