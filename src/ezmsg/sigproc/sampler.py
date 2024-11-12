import asyncio  # Dev/test apparatus
from collections import deque
from dataclasses import dataclass, field
import time
import typing

import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
    replace,
)
from ezmsg.util.generator import consumer


@dataclass(unsafe_hash=True)
class SampleTriggerMessage:
    timestamp: float = field(default_factory=time.time)
    """Time of the trigger, in seconds. The Clock depends on the input but defaults to time.time"""

    period: typing.Optional[typing.Tuple[float, float]] = None
    """The period around the timestamp, in seconds"""

    value: typing.Any = None
    """A value or 'label' associated with the trigger."""


@dataclass
class SampleMessage:
    trigger: SampleTriggerMessage
    """The time, window, and value (if any) associated with the trigger."""

    sample: AxisArray
    """The data sampled around the trigger."""


@consumer
def sampler(
    buffer_dur: float,
    axis: typing.Optional[str] = None,
    period: typing.Optional[typing.Tuple[float, float]] = None,
    value: typing.Any = None,
    estimate_alignment: bool = True,
) -> typing.Generator[
    typing.List[SampleMessage], typing.Union[AxisArray, SampleTriggerMessage], None
]:
    """
    Sample data into a buffer, accept triggers, and return slices of sampled
    data around the trigger time.

    Args:
        buffer_dur: The duration of the buffer in seconds. The buffer must be long enough to store the oldest
            sample to be included in a window. e.g., a trigger lagged by 0.5 seconds with a period of (-1.0, +1.5) will
            need a buffer of 0.5 + (1.5 - -1.0) = 3.0 seconds. It is best to at least double your estimate if memory allows.
        axis: The axis along which to sample the data.
            None (default) will choose the first axis in the first input.
            Note: (for now) the axis must exist in the msg .axes and be of type AxisArray.LinearAxis
        period: The period in seconds during which to sample the data.
            Defaults to None. Only used if not None and the trigger message does not define its own period.
        value: The value to sample. Defaults to None.
        estimate_alignment: Whether to estimate the sample alignment. Defaults to True.
            If True, the trigger timestamp field is used to slice the buffer.
            If False, the trigger timestamp is ignored and the next signal's .offset is used.
            NOTE: For faster-than-realtime playback -- Signals and triggers must share the same (fast) clock for
            estimate_alignment to operate correctly.

    Returns:
        A generator that expects `.send` either an :obj:`AxisArray` containing streaming data messages,
        or a :obj:`SampleTriggerMessage` containing a trigger, and yields the list of :obj:`SampleMessage` s.
    """
    msg_out: list[SampleMessage] = []

    # State variables (most shared between trigger- and data-processing.
    triggers: deque[SampleTriggerMessage] = deque()
    buffer: typing.Optional[npt.NDArray] = None
    n_samples: int = 0
    offset: float = 0.0

    check_inputs = {
        "fs": None,  # Also a state variable
        "key": None,
        "shape": None,
    }

    while True:
        msg_in = yield msg_out
        msg_out = []

        if isinstance(msg_in, SampleTriggerMessage):
            # Input is a trigger message that we will use to sample the buffer.

            if buffer is None or check_inputs["fs"] is None:
                # We've yet to see any data; drop the trigger.
                continue

            _period = msg_in.period if msg_in.period is not None else period
            _value = msg_in.value if msg_in.value is not None else value

            if _period is None:
                ez.logger.warning("Sampling failed: period not specified")
                continue

            # Check that period is valid
            if _period[0] >= _period[1]:
                ez.logger.warning(
                    f"Sampling failed: invalid period requested ({_period})"
                )
                continue

            # Check that period is compatible with buffer duration.
            max_buf_len = int(np.round(buffer_dur * check_inputs["fs"]))
            req_buf_len = int(np.round((_period[1] - _period[0]) * check_inputs["fs"]))
            if req_buf_len >= max_buf_len:
                ez.logger.warning(f"Sampling failed: {period=} >= {buffer_dur=}")
                continue

            trigger_ts: float = msg_in.timestamp
            if not estimate_alignment:
                # Override the trigger timestamp with the next sample's likely timestamp.
                trigger_ts = offset + (n_samples + 1) / check_inputs["fs"]

            new_trig_msg = replace(
                msg_in, timestamp=trigger_ts, period=_period, value=_value
            )
            triggers.append(new_trig_msg)

        elif isinstance(msg_in, AxisArray):
            # Get properties from message
            axis = axis or msg_in.dims[0]
            axis_idx = msg_in.get_axis_idx(axis)
            axis_info = msg_in.get_axis(axis)
            fs = 1.0 / axis_info.gain
            sample_shape = (
                msg_in.data.shape[:axis_idx] + msg_in.data.shape[axis_idx + 1 :]
            )

            # TODO: We could accommodate change in dim order.
            # if axis_idx != check_inputs["axis_idx"]:
            #     msg_in = replace(
            #         msg_in,
            #         data=np.moveaxis(msg_in.data, axis_idx, check_inputs["axis_idx"]),
            #         dims=TODO...
            #     )
            #    axis_idx = check_inputs["axis_idx"]

            # If the properties have changed in a breaking way then reset buffer and triggers.
            b_reset = fs != check_inputs["fs"]
            b_reset = b_reset or sample_shape != check_inputs["shape"]
            # TODO: Skip next line if we do np.moveaxis above
            b_reset = b_reset or axis_idx != check_inputs["axis_idx"]
            b_reset = b_reset or msg_in.key != check_inputs["key"]
            if b_reset:
                check_inputs["fs"] = fs
                check_inputs["shape"] = sample_shape
                check_inputs["axis_idx"] = axis_idx
                check_inputs["key"] = msg_in.key
                n_samples = msg_in.data.shape[axis_idx]
                buffer = None
                if len(triggers) > 0:
                    ez.logger.warning("Data stream changed: Discarding all triggers")
                triggers.clear()

            # Save some info for trigger processing
            offset = axis_info.offset

            # Update buffer
            buffer = (
                msg_in.data
                if buffer is None
                else np.concatenate((buffer, msg_in.data), axis=axis_idx)
            )

            # Calculate timestamps associated with buffer.
            buffer_offset = np.arange(buffer.shape[axis_idx], dtype=float)
            buffer_offset -= buffer_offset[-msg_in.data.shape[axis_idx]]
            buffer_offset *= axis_info.gain
            buffer_offset += axis_info.offset

            # ... for each trigger, collect the message (if possible) and append to msg_out
            for trig in list(triggers):
                if trig.period is None:
                    # This trigger was malformed; drop it.
                    triggers.remove(trig)

                # If the previous iteration had insufficient data for the trigger timestamp + period,
                #  and buffer-management removed data required for the trigger, then we will never be able
                #  to accommodate this trigger. Discard it. An increase in buffer_dur is recommended.
                if (trig.timestamp + trig.period[0]) < buffer_offset[0]:
                    ez.logger.warning(
                        f"Sampling failed: Buffer span {buffer_offset[0]} is beyond the "
                        f"requested sample period start: {trig.timestamp + trig.period[0]}"
                    )
                    triggers.remove(trig)

                t_start = trig.timestamp + trig.period[0]
                if t_start >= buffer_offset[0]:
                    start = np.searchsorted(buffer_offset, t_start)
                    stop = start + int(np.round(fs * (trig.period[1] - trig.period[0])))
                    if buffer.shape[axis_idx] > stop:
                        # Trigger period fully enclosed in buffer.
                        msg_out.append(
                            SampleMessage(
                                trigger=trig,
                                sample=replace(
                                    msg_in,
                                    data=slice_along_axis(
                                        buffer, slice(start, stop), axis_idx
                                    ),
                                    axes={
                                        **msg_in.axes,
                                        axis: replace(
                                            axis_info, offset=buffer_offset[start]
                                        ),
                                    },
                                ),
                            )
                        )
                        triggers.remove(trig)

            buf_len = int(buffer_dur * fs)
            buffer = slice_along_axis(buffer, np.s_[-buf_len:], axis_idx)


class SamplerSettings(ez.Settings):
    """
    Settings for :obj:`Sampler`.
    See :obj:`sampler` for a description of the fields.
    """

    buffer_dur: float
    axis: typing.Optional[str] = None
    period: typing.Optional[typing.Tuple[float, float]] = None
    """Optional default period if unspecified in SampleTriggerMessage"""

    value: typing.Any = None
    """Optional default value if unspecified in SampleTriggerMessage"""

    estimate_alignment: bool = True
    """
    If true, use message timestamp fields and reported sampling rate to estimate sample-accurate alignment for samples.
    If false, sampling will be limited to incoming message rate -- "Block timing"
    NOTE: For faster-than-realtime playback --  Incoming timestamps must reflect
    "realtime" operation for estimate_alignment to operate correctly.
    """


class SamplerState(ez.State):
    cur_settings: SamplerSettings
    gen: typing.Generator[
        typing.Union[AxisArray, SampleTriggerMessage], typing.List[SampleMessage], None
    ]


class Sampler(ez.Unit):
    """An :obj:`Unit` for :obj:`sampler`."""

    SETTINGS = SamplerSettings
    STATE = SamplerState

    INPUT_TRIGGER = ez.InputStream(SampleTriggerMessage)
    INPUT_SETTINGS = ez.InputStream(SamplerSettings)
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SAMPLE = ez.OutputStream(SampleMessage)

    def construct_generator(self):
        self.STATE.gen = sampler(
            buffer_dur=self.STATE.cur_settings.buffer_dur,
            axis=self.STATE.cur_settings.axis,
            period=self.STATE.cur_settings.period,
            value=self.STATE.cur_settings.value,
            estimate_alignment=self.STATE.cur_settings.estimate_alignment,
        )

    async def initialize(self) -> None:
        self.STATE.cur_settings = self.SETTINGS
        self.construct_generator()

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: SamplerSettings) -> None:
        self.STATE.cur_settings = msg
        self.construct_generator()

    @ez.subscriber(INPUT_TRIGGER)
    async def on_trigger(self, msg: SampleTriggerMessage) -> None:
        _ = self.STATE.gen.send(msg)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SAMPLE)
    async def on_signal(self, msg: AxisArray) -> typing.AsyncGenerator:
        pub_samples = self.STATE.gen.send(msg)
        for sample in pub_samples:
            yield self.OUTPUT_SAMPLE, sample


class TriggerGeneratorSettings(ez.Settings):
    period: typing.Tuple[float, float]
    """The period around the trigger event."""

    prewait: float = 0.5
    """The time before the first trigger (sec)"""

    publish_period: float = 5.0
    """The period between triggers (sec)"""


class TriggerGenerator(ez.Unit):
    """
    A unit to generate triggers every `publish_period` interval.
    """

    SETTINGS = TriggerGeneratorSettings

    OUTPUT_TRIGGER = ez.OutputStream(SampleTriggerMessage)

    @ez.publisher(OUTPUT_TRIGGER)
    async def generate(self) -> typing.AsyncGenerator:
        await asyncio.sleep(self.SETTINGS.prewait)

        output = 0
        while True:
            out_msg = SampleTriggerMessage(period=self.SETTINGS.period, value=output)
            yield self.OUTPUT_TRIGGER, out_msg

            await asyncio.sleep(self.SETTINGS.publish_period)
            output += 1
