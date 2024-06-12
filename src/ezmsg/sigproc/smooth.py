import copy
import typing

import numpy as np
import numpy.typing as npt
import scipy.signal
import ezmsg.core as ez
from ezmsg.util.generator import consumer, GenAxisArray
from ezmsg.util.messages.axisarray import AxisArray


@consumer
def expdecay(
    alpha: float,
    method="manual",
    axis="time"
) -> typing.Generator[AxisArray, AxisArray, None]:
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in closed range [0, 1]")

    if method == "manual":
        last_result: typing.Optional[npt.NDArray] = None
    elif method == "lfilt":
        coefs = [alpha], [1, alpha - 1]
        zi: typing.Optional[npt.NDArray] = None

    msg_out = AxisArray(np.array([]), dims=[""])
    while True:
        msg_in: AxisArray = yield msg_out

        targ_ax = msg_in.get_axis_idx(axis)
        data = np.moveaxis(msg_in.data, targ_ax, 0)

        if method == "manual":
            result = np.zeros_like(data)
            if last_result is None:
                result[0] = data[[0], :]
                # TODO: if sparse, result[0] = data[[0], :].data
                last_result = result[0]

            for step_ix, step_data in enumerate(data):
                result[step_ix] = alpha * step_data + (1 - alpha) * last_result
                last_result = result[step_ix]

        elif method == "lfilt":
            if zi is None:
                zi = scipy.signal.lfilter_zi(*coefs)
                zi = zi * data[:1]
            result, zi = scipy.signal.lfilter(*coefs, data, zi=zi, axis=0)

        msg_out = copy.copy(msg_in)
        msg_out.data = np.moveaxis(result, 0, -1)


class ExpDecaySettings(ez.Settings):
    alpha: float
    method = "manual"
    axis = "time"


class ExpDecay(GenAxisArray):
    SETTINGS = ExpDecaySettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = expdecay(
            alpha=self.SETTINGS.alpha,
            method=self.SETTINGS.method,
            axis=self.SETTINGS.axis
        )

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> typing.AsyncGenerator:
        yield self.OUTPUT_SIGNAL, self.STATE.gen.send(message)
