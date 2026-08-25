"""
Apply the Anscombe variance-stabilizing transform to the data, ``2 * sqrt(c + 3/8)``,
or invert it.

This is a variance-stabilizing transform for Poisson-distributed data such as
spike or photon counts: the output has approximately unit variance regardless
of the underlying rate, which lets downstream steps that assume homoscedastic
Gaussian noise be applied to count data.

Inputs below ``-3/8`` produce NaN, so this expects non-negative counts.

.. note::
    This module supports the :doc:`Array API standard </guides/explanations/array_api>`,
    enabling use with NumPy, CuPy, PyTorch, and other compatible array libraries.
"""

import math

import ezmsg.core as ez
from array_api_compat import get_namespace
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ..spectral import OptionsEnum

_OFFSET = 3.0 / 8.0
"""The constant in Anscombe's transform, chosen to make the residual bias O(1/c)."""

_D_MIN = 2.0 * math.sqrt(_OFFSET)
"""The forward transform's value at zero counts; the exact inverse is undefined below it."""

_SQRT_1P5 = math.sqrt(1.5)


class AnscombeTransformer(BaseTransformer[None, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        return replace(message, data=2.0 * xp.sqrt(message.data + _OFFSET))


class Anscombe(BaseTransformerUnit[None, AxisArray, AxisArray, AnscombeTransformer]): ...  # SETTINGS = None


class InverseMethod(OptionsEnum):
    """How to map stabilized values back to counts.

    Every inverse here maps a *denoised* stabilized value back to a rate -- that is,
    they invert ``rate -> E[anscombe(counts)]``. Applying one to still-noisy data
    returns noisy counts, but the mean of that output is not the mean of the input.
    """

    EXACT = "exact"
    """Closed-form approximation of the exact unbiased inverse (Makitalo & Foi, 2011).
    Unbiased down to very low rates, at the cost of a few extra elementwise ops."""

    ASYMPTOTIC = "asymptotic"
    """``(y/2)**2 - 1/8``. Unbiased as the rate grows, noticeably biased below ~5 counts."""

    ALGEBRAIC = "algebraic"
    """``(y/2)**2 - 3/8``. The strict functional inverse of the forward transform, so it
    round-trips exactly, but it underestimates the mean of noisy data at low rates."""


class InverseAnscombeSettings(ez.Settings):
    method: str | InverseMethod = InverseMethod.EXACT
    """Which inverse to apply. See :obj:`InverseMethod`. Default is EXACT."""


class InverseAnscombeTransformer(BaseTransformer[InverseAnscombeSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        # Accepts the enum or its string value; raises ValueError on anything else.
        method = InverseMethod(self.settings.method)
        data = message.data

        if method is not InverseMethod.EXACT:
            offset = _OFFSET if method is InverseMethod.ALGEBRAIC else 0.125
            return replace(message, data=data * data / 4.0 - offset)

        # The D**-3 term diverges as the input approaches zero, so clamp at the value
        # the forward transform produces for zero counts. The expression evaluates to
        # exactly 0 there, so this floors the output at zero counts rather than
        # introducing a discontinuity.
        xp = get_namespace(data)
        d = xp.clip(data, _D_MIN, None)
        r = 1.0 / d
        inv = d * d / 4.0 + 0.25 * _SQRT_1P5 * r - 1.375 * r * r + 0.625 * _SQRT_1P5 * r * r * r - 0.125
        return replace(message, data=inv)


class InverseAnscombe(BaseTransformerUnit[InverseAnscombeSettings, AxisArray, AxisArray, InverseAnscombeTransformer]):
    SETTINGS = InverseAnscombeSettings
