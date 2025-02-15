from dataclasses import field
import os

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.spectrum import (
    SpectralTransform,
    SpectralOutput,
    WindowFunction,
    Spectrum,
    SpectrumSettings,
)
from ezmsg.sigproc.window import Window, WindowSettings
from ezmsg.sigproc.synth import EEGSynth, EEGSynthSettings
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messagecodec import message_log
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings
from tests.helpers.util import (
    get_test_fn,
)


class SpectrumSettingsTest(ez.Settings):
    synth_settings: EEGSynthSettings
    window_settings: WindowSettings
    spectrum_settings: SpectrumSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(
        default_factory=TerminateOnTotalSettings
    )


class SpectrumIntegrationTest(ez.Collection):
    SETTINGS = SpectrumSettingsTest

    SOURCE = EEGSynth()
    WIN = Window()
    SPEC = Spectrum()
    SINK = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.SOURCE.apply_settings(self.SETTINGS.synth_settings)
        self.WIN.apply_settings(self.SETTINGS.window_settings)
        self.SPEC.apply_settings(self.SETTINGS.spectrum_settings)
        self.SINK.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.SOURCE.OUTPUT_SIGNAL, self.WIN.INPUT_SIGNAL),
            (self.WIN.OUTPUT_SIGNAL, self.SPEC.INPUT_SIGNAL),
            (self.SPEC.OUTPUT_SIGNAL, self.SINK.INPUT_MESSAGE),
            (self.SINK.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
        )


def test_spectrum_system(
    test_name: str | None = None,
):
    fs = 500.0
    n_time = 100  # samples per block. dispatch_rate = fs / n_time
    target_dur = 2.0
    window_dur = 1.0
    window_shift = 0.2
    n_ch = 8
    target_messages = int((target_dur - window_dur) / window_shift + 1)
    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    settings = SpectrumSettingsTest(
        synth_settings=EEGSynthSettings(
            fs=fs,
            n_time=n_time,
            alpha_freq=10.5,
            n_ch=n_ch,
        ),
        window_settings=WindowSettings(
            axis="time",
            window_dur=window_dur,
            window_shift=window_shift,
        ),
        spectrum_settings=SpectrumSettings(
            axis="time",
            window=WindowFunction.HAMMING,
            transform=SpectralTransform.REL_DB,
            output=SpectralOutput.POSITIVE,
        ),
        log_settings=MessageLoggerSettings(
            output=test_filename,
        ),
        term_settings=TerminateOnTotalSettings(
            total=target_messages,
        ),
    )
    system = SpectrumIntegrationTest(settings)
    ez.run(SYSTEM=system)

    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)
    agg = AxisArray.concatenate(*messages, dim="time")
    # Spectral length is half window length because we output only POSITIVE frequencies.
    win_len = window_dur * fs
    assert agg.data.shape == (target_messages, win_len // 2 + 1 - (win_len % 2), n_ch)
