from dataclasses import field
import os

import pytest
import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messagegate import MessageGate, MessageGateSettings
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messagecodec import message_log
from ezmsg.util.terminate import TerminateOnTimeout as TerminateTest
from ezmsg.util.terminate import TerminateOnTimeoutSettings as TerminateTestSettings
from ezmsg.util.debuglog import DebugLog

from ezmsg.sigproc.synth import Counter, CounterSettings
from ezmsg.sigproc.window import Window, WindowSettings

from tests.helpers.util import get_test_fn, calculate_expected_windows


class WindowSystemSettings(ez.Settings):
    num_msgs: int
    counter_settings: CounterSettings
    window_settings: WindowSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateTestSettings = field(default_factory=TerminateTestSettings)


class WindowSystem(ez.Collection):
    COUNTER = Counter()
    GATE = MessageGate()
    WIN = Window()
    LOG = MessageLogger()
    TERM = TerminateTest()

    DEBUG = DebugLog()

    SETTINGS = WindowSystemSettings

    def configure(self) -> None:
        self.COUNTER.apply_settings(self.SETTINGS.counter_settings)
        self.GATE.apply_settings(
            MessageGateSettings(
                start_open=True,
                default_open=False,
                default_after=self.SETTINGS.num_msgs,
            )
        )
        self.WIN.apply_settings(self.SETTINGS.window_settings)
        self.LOG.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.COUNTER.OUTPUT_SIGNAL, self.GATE.INPUT),
            # ( self.COUNTER.OUTPUT_SIGNAL, self.DEBUG.INPUT ),
            (self.GATE.OUTPUT, self.WIN.INPUT_SIGNAL),
            # ( self.GATE.OUTPUT, self.DEBUG.INPUT ),
            (self.WIN.OUTPUT_SIGNAL, self.LOG.INPUT_MESSAGE),
            # ( self.WIN.OUTPUT_SIGNAL, self.DEBUG.INPUT ),
            (self.LOG.OUTPUT_MESSAGE, self.TERM.INPUT),
            # ( self.LOG.OUTPUT_MESSAGE, self.DEBUG.INPUT ),
        )


# It takes >15 minutes to go through the full set of combinations tested for the generator.
# We need only test a subset to assert integration is correct.
@pytest.mark.parametrize(
    "msg_block_size, newaxis, win_dur, win_shift, zero_pad, fs",
    [
        (1, None, 0.2, None, "input", 10.0),
        (20, None, 0.2, None, "input", 10.0),
        (1, "step", 0.2, None, "input", 10.0),
        (10, "step", 0.2, 1.0, "shift", 500.0),
        (20, "step", 1.0, 1.0, "shift", 500.0),
        (10, "step", 1.0, 1.0, "none", 500.0),
        (20, None, None, None, "input", 10.0),
    ],
)
def test_window_system(
    msg_block_size: int,
    newaxis: str | None,
    win_dur: float,
    win_shift: float | None,
    zero_pad: str,
    fs: float,
    test_name: str | None = None,
):
    # Calculate expected dimensions.
    win_len = int((win_dur or 1.0) * fs)
    shift_len = int(win_shift * fs) if win_shift is not None else msg_block_size
    # num_msgs should be the greater value between (2 full windows + a shift) or 4.0 seconds
    data_len = max(2 * win_len + shift_len - 1, int(4.0 * fs))
    num_msgs = int(np.ceil(data_len / msg_block_size))

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    settings = WindowSystemSettings(
        num_msgs=num_msgs,
        counter_settings=CounterSettings(
            n_time=msg_block_size,
            fs=fs,
            dispatch_rate=float(num_msgs),  # Get through them in about 1 second.
        ),
        window_settings=WindowSettings(
            axis="time",
            newaxis=newaxis,
            window_dur=win_dur,
            window_shift=win_shift,
            zero_pad_until=zero_pad,
        ),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateTestSettings(time=1.0),  # sec
    )

    system = WindowSystem(settings)
    ez.run(SYSTEM=system)

    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)
    ez.logger.info(f"Analyzing recording of {len(messages)} messages...")

    # Within a test config, the metadata should not change across messages.
    for msg in messages:
        # In this test, fs should never change
        assert 1.0 / msg.axes["time"].gain == fs
        # In this test, we should have consistent dimensions
        assert msg.dims == ([newaxis, "time", "ch"] if newaxis else ["time", "ch"])
        # Window should always output the same shape data
        assert msg.shape[msg.get_axis_idx("ch")] == 1
        # Counter yields only one channel.
        assert msg.shape[msg.get_axis_idx("time")] == (
            msg_block_size if win_dur is None else win_len
        )

    ez.logger.info("Consistent metadata!")

    # Collect the outputs we want to test
    data: list[npt.NDArray] = [msg.data for msg in messages]
    if newaxis is None:
        offsets = np.array([_.axes["time"].offset for _ in messages])
    else:
        offsets = np.hstack(
            [
                _.axes[newaxis].offset
                + _.axes[newaxis].gain * np.arange(_.data.shape[0])
                for _ in messages
            ]
        )

    # If this test was performed in "one-to-one" mode, we should
    # have one window output per message pushed to Window
    if win_shift is None:
        assert len(data) == num_msgs

    # Turn the data into a ndarray.
    if newaxis is not None:
        data = np.concatenate(data, axis=messages[0].get_axis_idx(newaxis))
    else:
        data = np.stack(data, axis=messages[0].get_axis_idx("time"))

    # Calculate the expected results for comparison.
    sent_data = np.arange(num_msgs * msg_block_size)[None, :]
    expected, tvec = calculate_expected_windows(
        sent_data,
        fs,
        win_shift,
        zero_pad,
        "beginning",
        msg_block_size,
        shift_len,
        win_len,
        1,
        data_len,
        num_msgs,
        0,
    )

    # Compare results to expected
    if win_dur is None:
        assert np.array_equal(data, sent_data.reshape((num_msgs, msg_block_size, -1)))
    else:
        assert np.array_equal(data, expected)
        assert np.allclose(offsets, tvec)

    ez.logger.info("Test Complete.")


if __name__ == "__main__":
    test_window_system(5, 0.6, None, test_name="test_window_system")
