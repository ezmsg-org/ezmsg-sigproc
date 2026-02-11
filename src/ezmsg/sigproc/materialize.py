"""
Materialize (evaluate) lazy array data.

MLX arrays are lazily evaluated — computations are queued but not executed
until the result is needed. This module provides an explicit evaluation point
so that downstream processors receive fully-evaluated data.
"""

from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray


class MaterializeTransformer(BaseTransformer[None, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        try:
            import mlx.core as mx

            if isinstance(message.data, mx.array):
                mx.eval(message.data)
        except ImportError:
            pass
        return message


class Materialize(BaseTransformerUnit[None, AxisArray, AxisArray, MaterializeTransformer]): ...
