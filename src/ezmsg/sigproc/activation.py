import scipy.special
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .spectral import OptionsEnum
from .base import BaseTransformer, BaseTransformerUnit


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


class ActivationSettings(ez.Settings):
    function: str | ActivationFunction = ActivationFunction.NONE
    """An enum value from ActivationFunction or a string representing the activation function.
         Possible values are: SIGMOID, EXPIT, LOGIT, LOGEXPIT, "sigmoid", "expit", "logit", "log_expit".
         SIGMOID and EXPIT are equivalent. See :obj:`scipy.special.expit` for more details."""


class ActivationTransformer(BaseTransformer[ActivationSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        if type(self.settings.function) is ActivationFunction:
            func = ACTIVATIONS[self.settings.function]
        else:
            # str type handling
            function = self.settings.function.lower()
            if function not in ActivationFunction.options():
                raise ValueError(
                    f"Unrecognized activation function {function}. Must be one of {ACTIVATIONS.keys()}"
                )
            function = list(ACTIVATIONS.keys())[
                ActivationFunction.options().index(function)
            ]
            func = ACTIVATIONS[function]

        return replace(message, data=func(message.data))


class Activation(
    BaseTransformerUnit[ActivationSettings, AxisArray, AxisArray, ActivationTransformer]
):
    SETTINGS = ActivationSettings


def activation(
    function: str | ActivationFunction,
) -> ActivationTransformer:
    """
    Transform the data with a simple activation function.

    Args:
        function: An enum value from ActivationFunction or a string representing the activation function.
         Possible values are: SIGMOID, EXPIT, LOGIT, LOGEXPIT, "sigmoid", "expit", "logit", "log_expit".
         SIGMOID and EXPIT are equivalent. See :obj:`scipy.special.expit` for more details.

    Returns: :obj:`ActivationTransformer`

    """
    return ActivationTransformer(ActivationSettings(function=function))
