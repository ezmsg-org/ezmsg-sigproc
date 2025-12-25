import asyncio
import copy
import traceback
import typing
from collections import deque

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import (
    BaseConsumerUnit,
    BaseProducerUnit,
    BaseStatefulProducer,
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import (
    AxisArray,
)
from ezmsg.util.messages.util import replace

from .util.axisarray_buffer import HybridAxisArrayBuffer
from .util.buffer import UpdateStrategy
from .util.message import SampleMessage, SampleTriggerMessage
from .util.profile import profile_subpub


class SamplerSettings(ez.Settings):
    """
    Settings for :obj:`Sampler`.
    See :obj:`sampler` for a description of the fields.
    """

    buffer_dur: float
    """
     The duration of the buffer in seconds. The buffer must be long enough to store the oldest
        sample to be included in a window. e.g., a trigger lagged by 0.5 seconds with a period of (-1.0, +1.5) will
        need a buffer of 0.5 + (1.5 - -1.0) = 3.0 seconds. It is best to at least double your estimate if memory allows.
    """

    axis: str | None = None
    """
    The axis along which to sample the data.
        None (default) will choose the first axis in the first input.
        Note: (for now) the axis must exist in the msg .axes and be of type AxisArray.LinearAxis
    """

    period: tuple[float, float] | None = None
    """Optional default period (in seconds) if unspecified in SampleTriggerMessage."""

    value: typing.Any = None
    """Optional default value if unspecified in SampleTriggerMessage"""

    estimate_alignment: bool = True
    """
    If true, use message timestamp fields and reported sampling rate to estimate
     sample-accurate alignment for samples.
    If false, sampling will be limited to incoming message rate -- "Block timing"
    NOTE: For faster-than-realtime playback --  Incoming timestamps must reflect
    "realtime" operation for estimate_alignment to operate correctly.
    """

    buffer_update_strategy: UpdateStrategy = "immediate"
    """
    The buffer update strategy. See :obj:`ezmsg.sigproc.util.buffer.UpdateStrategy`.
    If you expect to push data much more frequently than triggers, then "on_demand"
    might be more efficient. For most other scenarios, "immediate" is best.
    """


@processor_state
class SamplerState:
    buffer: HybridAxisArrayBuffer | None = None
    triggers: deque[SampleTriggerMessage] | None = None


class SamplerTransformer(BaseStatefulTransformer[SamplerSettings, AxisArray, AxisArray, SamplerState]):
    def __call__(self, message: AxisArray | SampleTriggerMessage) -> list[SampleMessage]:
        # TODO: Currently we have a single entry point that accepts both
        #  data and trigger messages and we choose a code path based on
        #  the message type. However, in the future we will likely replace
        #  SampleTriggerMessage with an agumented form of AxisArray,
        #  leveraging its attrs field, which makes this a bit harder.
        #  We should probably force callers of this object to explicitly
        #  call `push_trigger` for trigger messages. This will also
        #  simplify typing somewhat because `push_trigger` should not
        #  return anything yet we currently have it returning an empty
        #  list just to be compatible with __call__.
        if isinstance(message, AxisArray):
            return super().__call__(message)
        else:
            return self.push_trigger(message)

    def _hash_message(self, message: AxisArray) -> int:
        # Compute hash based on message properties that require state reset
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        sample_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        return hash((sample_shape, message.key))

    def _reset_state(self, message: AxisArray) -> None:
        self._state.buffer = HybridAxisArrayBuffer(
            duration=self.settings.buffer_dur,
            axis=self.settings.axis or message.dims[0],
            update_strategy=self.settings.buffer_update_strategy,
            overflow_strategy="warn-overwrite",  # True circular buffer
        )
        if self._state.triggers is None:
            self._state.triggers = deque()
        self._state.triggers.clear()

    def _process(self, message: AxisArray) -> list[SampleMessage]:
        self._state.buffer.write(message)

        # How much data in the buffer?
        buff_t_range = (
            self._state.buffer.axis_first_value,
            self._state.buffer.axis_final_value,
        )

        # Process in reverse order so that we can remove triggers safely as we iterate.
        msgs_out: list[SampleMessage] = []
        for trig_ix in range(len(self._state.triggers) - 1, -1, -1):
            trig = self._state.triggers[trig_ix]
            if trig.period is None:
                ez.logger.warning("Sampling failed: trigger period not specified")
                del self._state.triggers[trig_ix]
                continue

            trig_range = trig.timestamp + np.array(trig.period)

            # If the previous iteration had insufficient data for the trigger timestamp + period,
            #  and buffer-management removed data required for the trigger, then we will never be able
            #  to accommodate this trigger. Discard it. An increase in buffer_dur is recommended.
            if trig_range[0] < buff_t_range[0]:
                ez.logger.warning(
                    f"Sampling failed: Buffer span {buff_t_range} begins beyond the "
                    f"requested sample period start: {trig_range[0]}"
                )
                del self._state.triggers[trig_ix]
                continue

            if trig_range[1] > buff_t_range[1]:
                # We don't *yet* have enough data to satisfy this trigger.
                continue

            # We know we have enough data in the buffer to satisfy this trigger.
            buff_idx = self._state.buffer.axis_searchsorted(trig_range, side="right")
            self._state.buffer.seek(buff_idx[0])  # FFWD to starting position.
            buff_axarr = self._state.buffer.peek(buff_idx[1] - buff_idx[0])
            self._state.buffer.seek(-buff_idx[0])  # Rewind it back.
            # Note: buffer will trim itself as needed based on buffer_dur.

            # Prepare output and drop trigger
            msgs_out.append(SampleMessage(trigger=copy.copy(trig), sample=buff_axarr))
            del self._state.triggers[trig_ix]

        msgs_out.reverse()  # in-place
        return msgs_out

    def push_trigger(self, message: SampleTriggerMessage) -> list[SampleMessage]:
        # Input is a trigger message that we will use to sample the buffer.

        if self._state.buffer is None:
            # We've yet to see any data; drop the trigger.
            return []

        _period = message.period if message.period is not None else self.settings.period
        _value = message.value if message.value is not None else self.settings.value

        if _period is None:
            ez.logger.warning("Sampling failed: period not specified")
            return []

        # Check that period is valid
        if _period[0] >= _period[1]:
            ez.logger.warning(f"Sampling failed: invalid period requested ({_period})")
            return []

        # Check that period is compatible with buffer duration.
        if (_period[1] - _period[0]) > self.settings.buffer_dur:
            ez.logger.warning(
                f"Sampling failed: trigger period {_period=} >= buffer capacity {self.settings.buffer_dur=}"
            )
            return []

        trigger_ts: float = message.timestamp
        if not self.settings.estimate_alignment:
            # Override the trigger timestamp with the next sample's likely timestamp.
            trigger_ts = self._state.buffer.axis_final_value + self._state.buffer.axis_gain

        new_trig_msg = replace(message, timestamp=trigger_ts, period=_period, value=_value)
        self._state.triggers.append(new_trig_msg)
        return []


class Sampler(BaseTransformerUnit[SamplerSettings, AxisArray, AxisArray, SamplerTransformer]):
    SETTINGS = SamplerSettings

    INPUT_TRIGGER = ez.InputStream(SampleTriggerMessage)
    OUTPUT_SIGNAL = ez.OutputStream(SampleMessage)

    @ez.subscriber(INPUT_TRIGGER)
    async def on_trigger(self, msg: SampleTriggerMessage) -> None:
        _ = self.processor.push_trigger(msg)

    @ez.subscriber(BaseConsumerUnit.INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        try:
            for sample in self.processor(message):
                yield self.OUTPUT_SIGNAL, sample
        except Exception as e:
            ez.logger.info(f"{traceback.format_exc()} - {e}")


def sampler(
    buffer_dur: float,
    axis: str | None = None,
    period: tuple[float, float] | None = None,
    value: typing.Any = None,
    estimate_alignment: bool = True,
) -> SamplerTransformer:
    """
    Sample data into a buffer, accept triggers, and return slices of sampled
    data around the trigger time.

    Returns:
        A generator that expects `.send` either an :obj:`AxisArray` containing streaming data messages,
        or a :obj:`SampleTriggerMessage` containing a trigger, and yields the list of :obj:`SampleMessage` s.
    """
    return SamplerTransformer(
        settings=SamplerSettings(
            buffer_dur=buffer_dur,
            axis=axis,
            period=period,
            value=value,
            estimate_alignment=estimate_alignment,
        )
    )


class TriggerGeneratorSettings(ez.Settings):
    period: tuple[float, float]
    """The period around the trigger event."""

    prewait: float = 0.5
    """The time before the first trigger (sec)"""

    publish_period: float = 5.0
    """The period between triggers (sec)"""


@processor_state
class TriggerGeneratorState:
    output: int = 0


class TriggerProducer(BaseStatefulProducer[TriggerGeneratorSettings, SampleTriggerMessage, TriggerGeneratorState]):
    def _reset_state(self) -> None:
        self._state.output = 0

    async def _produce(self) -> SampleTriggerMessage:
        await asyncio.sleep(self.settings.publish_period)
        out_msg = SampleTriggerMessage(period=self.settings.period, value=self._state.output)
        self._state.output += 1
        return out_msg


class TriggerGenerator(
    BaseProducerUnit[
        TriggerGeneratorSettings,
        SampleTriggerMessage,
        TriggerProducer,
    ]
):
    SETTINGS = TriggerGeneratorSettings
