import os


import ezmsg.core as ez
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.sigproc.sampler import (
    Sampler,
    SamplerSettings,
    TriggerGenerator,
    TriggerGeneratorSettings,
    SampleTriggerMessage,
)
from ezmsg.sigproc.synth import Oscillator, OscillatorSettings
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings
from ezmsg.util.debuglog import DebugLog

from tests.helpers.util import get_test_fn


class SamplerSystemSettings(ez.Settings):
    # num_msgs: int
    osc_settings: OscillatorSettings
    trigger_settings: TriggerGeneratorSettings
    sampler_settings: SamplerSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings


class SamplerSystem(ez.Collection):
    SETTINGS = SamplerSystemSettings

    OSC = Oscillator()
    TRIGGER = TriggerGenerator()
    SAMPLER = Sampler()
    LOG = MessageLogger()
    DEBUG = DebugLog()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.OSC.apply_settings(self.SETTINGS.osc_settings)
        self.LOG.apply_settings(self.SETTINGS.log_settings)
        self.SAMPLER.apply_settings(self.SETTINGS.sampler_settings)
        self.TRIGGER.apply_settings(self.SETTINGS.trigger_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.OSC.OUTPUT_SIGNAL, self.SAMPLER.INPUT_SIGNAL),
            (self.SAMPLER.OUTPUT_SIGNAL, self.LOG.INPUT_MESSAGE),
            (self.LOG.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
            # Trigger branch
            (self.TRIGGER.OUTPUT_SIGNAL, self.SAMPLER.INPUT_TRIGGER),
            # Debug branches
            (self.TRIGGER.OUTPUT_SIGNAL, self.DEBUG.INPUT),
            (self.SAMPLER.OUTPUT_SIGNAL, self.DEBUG.INPUT),
        )


def test_sampler_system(test_name: str | None = None):
    freq = 40.0
    period = (0.5, 1.5)
    n_msgs = 4

    sample_dur = period[1] - period[0]
    publish_period = sample_dur * 2.0

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    settings = SamplerSystemSettings(
        osc_settings=OscillatorSettings(
            n_time=2,  # Number of samples to output per block
            fs=freq,  # Sampling rate of signal output in Hz
            dispatch_rate="realtime",
            freq=2.0,  # Oscillation frequency in Hz
            amp=1.0,  # Amplitude
            phase=0.0,  # Phase offset (in radians)
            sync=True,  # Adjust `freq` to sync with sampling rate
        ),
        trigger_settings=TriggerGeneratorSettings(
            period=period, prewait=0.5, publish_period=publish_period
        ),
        sampler_settings=SamplerSettings(buffer_dur=publish_period + 1.0),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateOnTotalSettings(total=n_msgs),
    )

    system = SamplerSystem(settings)

    ez.run(SYSTEM=system)
    messages: list[SampleTriggerMessage] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)
    ez.logger.info(f"Analyzing recording of {len(messages)} messages...")
    assert len(messages) == n_msgs
    assert all([_.sample.data.shape == (int(freq * sample_dur), 1) for _ in messages])
    # Test the sample window slice vs the trigger timestamps
    latencies = [
        _.sample.axes["time"].offset - (_.trigger.timestamp + _.trigger.period[0])
        for _ in messages
    ]
    assert all([0 <= _ < 1 / freq for _ in latencies])
    # Given that the input is a pure sinusoid, we could test that the signal has expected characteristics.
