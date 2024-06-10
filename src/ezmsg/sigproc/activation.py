import copy
import enum
import typing

import numpy as np
import scipy.special
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer, GenAxisArray


class OptionsEnum(enum.Enum):
    @classmethod
    def options(cls):
        return list(map(lambda c: c.value, cls))


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
            raise ValueError(f"Unrecognized activation function {function}. Must be one of {ACTIVATIONS.keys()}")
        function = list(ACTIVATIONS.keys())[ActivationFunction.options().index(function)]
        func = ACTIVATIONS[function]

    msg_in = AxisArray(np.array([]), dims=[""])
    msg_out = AxisArray(np.array([]), dims=[""])

    while True:
        msg_in = yield msg_out

        msg_out = copy.copy(msg_in)
        msg_out.data = func(msg_out.data)


class ActivationSettings(ez.Settings):
    function: str = ActivationFunction.NONE


class Activation(GenAxisArray):
    SETTINGS: ActivationSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = activation(
            function=self.SETTINGS.function
        )

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> typing.AsyncGenerator:
        ret = self.STATE.gen.send(message)
        if ret is not None:
            yield self.OUTPUT_SIGNAL, ret
