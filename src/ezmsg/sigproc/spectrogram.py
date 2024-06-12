import typing

import numpy as np

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer, GenAxisArray, compose
from ezmsg.util.messages.modify import modify_axis
from ezmsg.sigproc.window import windowing
from ezmsg.sigproc.spectrum import (
    spectrum,
    WindowFunction, SpectralTransform, SpectralOutput
)


@consumer
def spectrogram(
    window_dur: typing.Optional[float] = None,
    window_shift: typing.Optional[float] = None,
    window: WindowFunction = WindowFunction.HANNING,
    transform: SpectralTransform = SpectralTransform.REL_DB,
    output: SpectralOutput = SpectralOutput.POSITIVE
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Calculate a spectrogram on streaming data.

    Chains :obj:`ezmsg.sigproc.window.windowing` to apply a moving window on the data,
    :obj:`ezmsg.sigproc.spectrum.spectrum` to calculate spectra for each window,
    and finally :obj:`ezmsg.util.messages.modify.modify_axis` to convert the win axis back to time axis.

    Args:
        window_dur: See :obj:`ezmsg.sigproc.window.windowing`
        window_shift: See :obj:`ezmsg.sigproc.window.windowing`
        window: See :obj:`ezmsg.sigproc.spectrum.spectrum`
        transform: See :obj:`ezmsg.sigproc.spectrum.spectrum`
        output: See :obj:`ezmsg.sigproc.spectrum.spectrum`

    Returns:
        A primed generator object that expects `.send(axis_array)` of continuous data
        and yields an AxisArray of time-frequency power values.
    """

    pipeline = compose(
        windowing(axis="time", newaxis="win", window_dur=window_dur, window_shift=window_shift),
        spectrum(axis="time", window=window, transform=transform, output=output),
        modify_axis(name_map={"win": "time"})
    )

    # State variables
    axis_arr_out: typing.Optional[AxisArray] = None

    while True:
        axis_arr_in: AxisArray = yield axis_arr_out
        axis_arr_out = pipeline(axis_arr_in)


class SpectrogramSettings(ez.Settings):
    """
    Settings for :obj:`Spectrogram`.
    See :obj:`spectrogram` for a description of the parameters.
    """
    window_dur: typing.Optional[float] = None  # window duration in seconds
    window_shift: typing.Optional[float] = None  # window step in seconds. If None, window_shift == window_dur
    # See SpectrumSettings for details of following settings:
    window: WindowFunction = WindowFunction.HAMMING
    transform: SpectralTransform = SpectralTransform.REL_DB
    output: SpectralOutput = SpectralOutput.POSITIVE


class Spectrogram(GenAxisArray):
    """
    Unit for :obj:`spectrogram`.
    """
    SETTINGS: SpectrogramSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = spectrogram(
            window_dur=self.SETTINGS.window_dur,
            window_shift=self.SETTINGS.window_shift,
            window=self.SETTINGS.window,
            transform=self.SETTINGS.transform,
            output=self.SETTINGS.output
        )
    
    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, msg: AxisArray) -> typing.AsyncGenerator:
        out_msg = self.STATE.gen.send(msg)
        # There's a chance the return will be empty because windowing
        #  might not have received enough data.
        if out_msg.data.size > 0:
            yield self.OUTPUT_SIGNAL, out_msg
