from dataclasses import replace
import typing

import numpy as np
import scipy.special
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer

from .spectral import OptionsEnum
from .base import GenAxisArray


class ActivationFunction(OptionsEnum):
    """Activation (transformation) function."""

    NONE = "none"
    """None."""

    SIGMOID = "sigmoid"
    """:obj:`scipy.special.expit`"""

    EXPIT = "expit"
    """:obj:`scipy.special.expit`"""

    LOGIT = "logit"
    """:obj:`scipy.special.logit`"""

    LOGEXPIT = "log_expit"
    """:obj:`scipy.special.log_expit`"""


ACTIVATIONS = {
    ActivationFunction.NONE: lambda x: x,
    ActivationFunction.SIGMOID: scipy.special.expit,
    ActivationFunction.EXPIT: scipy.special.expit,
    ActivationFunction.LOGIT: scipy.special.logit,
    ActivationFunction.LOGEXPIT: scipy.special.log_expit,
}


@consumer
def activation(
    function: typing.Union[str, ActivationFunction],
) -> typing.Generator[AxisArray, AxisArray, None]:
    if type(function) is ActivationFunction:
        func = ACTIVATIONS[function]
    else:
        # str type. There's probably an easier way to support either enum or str argument. Oh well this works.
        function: str = function.lower()
        if function not in ActivationFunction.options():
            raise ValueError(
                f"Unrecognized activation function {function}. Must be one of {ACTIVATIONS.keys()}"
            )
        function = list(ACTIVATIONS.keys())[
            ActivationFunction.options().index(function)
        ]
        func = ACTIVATIONS[function]

    msg_out = AxisArray(np.array([]), dims=[""])

    while True:
        msg_in: AxisArray = yield msg_out
        msg_out = replace(msg_in, data=func(msg_in.data))


class ActivationSettings(ez.Settings):
    function: str = ActivationFunction.NONE


class Activation(GenAxisArray):
    SETTINGS = ActivationSettings

    def construct_generator(self):
        self.STATE.gen = activation(function=self.SETTINGS.function)
