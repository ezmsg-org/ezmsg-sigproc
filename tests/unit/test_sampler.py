import copy

import numpy as np
from frozendict import frozendict

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.util.message import SampleTriggerMessage
from ezmsg.sigproc.sampler import (
    sampler,
)

from tests.helpers.util import assert_messages_equal


def test_sampler_gen():
    data_dur = 10.0
    chunk_period = 0.1
    fs = 500.0
    n_chans = 3

    # The sampler is a bit complicated as it requires 2 different inputs: signal and triggers
    # Prepare signal data
    n_data = int(data_dur * fs)
    data = np.arange(n_chans * n_data).reshape(n_chans, n_data)
    offsets = np.arange(n_data) / fs
    n_chunks = int(np.ceil(data_dur / chunk_period))
    n_per_chunk = int(np.ceil(n_data / n_chunks))
    signal_msgs = [
        AxisArray(
            data=data[:, ix * n_per_chunk : (ix + 1) * n_per_chunk],
            dims=["ch", "time"],
            axes=frozendict(
                {
                    "time": AxisArray.TimeAxis(fs=fs, offset=offsets[ix * n_per_chunk]),
                    "ch": AxisArray.CoordinateAxis(
                        data=np.arange(n_chans).astype(str), dims=["ch"]
                    ),
                }
            ),
            key="test_sampler_gen",
        )
        for ix in range(n_chunks)
    ]
    backup_signal = [copy.deepcopy(_) for _ in signal_msgs]

    # Prepare triggers
    n_trigs = 7
    trig_ts = np.linspace(0.1, data_dur - 1.0, n_trigs) + np.random.randn(n_trigs) / fs
    period = (-0.01, 0.74)
    trigger_msgs = [
        SampleTriggerMessage(
            timestamp=_ts, period=period, value=["Start", "Stop"][_ix % 2]
        )
        for _ix, _ts in enumerate(trig_ts)
    ]
    backup_trigger = [copy.deepcopy(_) for _ in trigger_msgs]

    # Mix the messages and sort by time
    msg_ts = [_.axes["time"].offset for _ in signal_msgs] + [
        _.timestamp for _ in trigger_msgs
    ]
    mix_msgs = signal_msgs + trigger_msgs
    mix_msgs = [mix_msgs[_] for _ in np.argsort(msg_ts)]

    # Create the sample-generator
    period_dur = period[1] - period[0]
    buffer_dur = 2 * max(period_dur, period[1])
    gen = sampler(
        buffer_dur, axis="time", period=None, value=None, estimate_alignment=True
    )

    # Run the messages through the generator and collect samples.
    samples = []
    for msg in mix_msgs:
        samples.extend(gen.send(msg))

    assert_messages_equal(signal_msgs, backup_signal)
    assert_messages_equal(trigger_msgs, backup_trigger)

    assert len(samples) == n_trigs
    # Check sample data size
    assert all(
        [_.sample.data.shape == (n_chans, int(fs * period_dur)) for _ in samples]
    )
    # Compare the sample window slice against the trigger timestamps
    latencies = [
        _.sample.axes["time"].offset - (_.trigger.timestamp + _.trigger.period[0])
        for _ in samples
    ]
    assert all([0 <= _ < 1 / fs for _ in latencies])
    # Check the sample trigger value matches the trigger input.
    assert all(
        [_.trigger.value == ["Start", "Stop"][ix % 2] for ix, _ in enumerate(samples)]
    )
