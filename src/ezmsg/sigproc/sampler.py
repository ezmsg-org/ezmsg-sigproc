import asyncio
from collections import deque
import traceback
import typing

import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
)
from ezmsg.util.messages.util import replace

from .util.profile import profile_subpub
from .util.message import SampleMessage, SampleTriggerMessage
from .base import (
    BaseStatefulTransformer,
    BaseConsumerUnit,
    BaseTransformerUnit,
    BaseStatefulProducer,
    BaseProducerUnit,
    processor_state,
)


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
    If true, use message timestamp fields and reported sampling rate to estimate sample-accurate alignment for samples.
    If false, sampling will be limited to incoming message rate -- "Block timing"
    NOTE: For faster-than-realtime playback --  Incoming timestamps must reflect
    "realtime" operation for estimate_alignment to operate correctly.
    """


@processor_state
class SamplerState:
    fs: float = 0.0
    offset: float | None = None
    buffer: npt.NDArray | None = None
    triggers: deque[SampleTriggerMessage] | None = None
    n_samples: int = 0


class SamplerTransformer(
    BaseStatefulTransformer[SamplerSettings, AxisArray, AxisArray, SamplerState]
):
    def __call__(
        self, message: AxisArray | SampleTriggerMessage
    ) -> list[SampleMessage]:
        if isinstance(message, AxisArray):
            return super().__call__(message)
        else:
            return self.push_trigger(message)

    def _hash_message(self, message: AxisArray) -> int:
        # Compute hash based on message properties that require state reset
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        fs = 1.0 / message.get_axis(axis).gain
        sample_shape = (
            message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        )
        return hash((fs, sample_shape, axis_idx, message.key))

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        axis_info = message.get_axis(axis)
        self._state.fs = 1.0 / axis_info.gain
        self._state.buffer = None
        if self._state.triggers is None:
            self._state.triggers = deque()
        self._state.triggers.clear()
        self._state.n_samples = message.data.shape[axis_idx]

    def _process(self, message: AxisArray) -> list[SampleMessage]:
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        axis_info = message.get_axis(axis)
        self._state.offset = axis_info.offset

        # Update buffer
        self._state.buffer = (
            message.data
            if self._state.buffer is None
            else np.concatenate((self._state.buffer, message.data), axis=axis_idx)
        )

        # Calculate timestamps associated with buffer.
        buffer_offset = np.arange(self._state.buffer.shape[axis_idx], dtype=float)
        buffer_offset -= buffer_offset[-message.data.shape[axis_idx]]
        buffer_offset *= axis_info.gain
        buffer_offset += axis_info.offset

        # ... for each trigger, collect the message (if possible) and append to msg_out
        msg_out: list[SampleMessage] = []
        for trig in list(self._state.triggers):
            if trig.period is None:
                # This trigger was malformed; drop it.
                self._state.triggers.remove(trig)

            # If the previous iteration had insufficient data for the trigger timestamp + period,
            #  and buffer-management removed data required for the trigger, then we will never be able
            #  to accommodate this trigger. Discard it. An increase in buffer_dur is recommended.
            if (trig.timestamp + trig.period[0]) < buffer_offset[0]:
                ez.logger.warning(
                    f"Sampling failed: Buffer span {buffer_offset[0]} is beyond the "
                    f"requested sample period start: {trig.timestamp + trig.period[0]}"
                )
                self._state.triggers.remove(trig)

            t_start = trig.timestamp + trig.period[0]
            if t_start >= buffer_offset[0]:
                start = np.searchsorted(buffer_offset, t_start)
                stop = start + int(
                    np.round(self._state.fs * (trig.period[1] - trig.period[0]))
                )
                if self._state.buffer.shape[axis_idx] > stop:
                    # Trigger period fully enclosed in buffer.
                    msg_out.append(
                        SampleMessage(
                            trigger=trig,
                            sample=replace(
                                message,
                                data=slice_along_axis(
                                    self._state.buffer, slice(start, stop), axis_idx
                                ),
                                axes={
                                    **message.axes,
                                    axis: replace(
                                        axis_info, offset=buffer_offset[start]
                                    ),
                                },
                            ),
                        )
                    )
                    self._state.triggers.remove(trig)

        # Trim buffer
        buf_len = int(self.settings.buffer_dur * self._state.fs)
        self._state.buffer = slice_along_axis(
            self._state.buffer, np.s_[-buf_len:], axis_idx
        )

        return msg_out

    def push_trigger(self, message: SampleTriggerMessage) -> list[SampleMessage]:
        # Input is a trigger message that we will use to sample the buffer.

        if (
            self._state.buffer is None
            or not self._state.fs
            or self._state.offset is None
        ):
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
        max_buf_len = int(np.round(self.settings.buffer_dur * self._state.fs))
        req_buf_len = int(np.round((_period[1] - _period[0]) * self._state.fs))
        if req_buf_len >= max_buf_len:
            ez.logger.warning(
                f"Sampling failed: {_period=} >= {self.settings.buffer_dur=}"
            )
            return []

        trigger_ts: float = message.timestamp
        if not self.settings.estimate_alignment:
            # Override the trigger timestamp with the next sample's likely timestamp.
            trigger_ts = (
                self._state.offset + (self.state.n_samples + 1) / self._state.fs
            )

        new_trig_msg = replace(
            message, timestamp=trigger_ts, period=_period, value=_value
        )
        self._state.triggers.append(new_trig_msg)
        return []


class Sampler(
    BaseTransformerUnit[SamplerSettings, AxisArray, AxisArray, SamplerTransformer]
):
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


class TriggerProducer(
    BaseStatefulProducer[
        TriggerGeneratorSettings, SampleTriggerMessage, TriggerGeneratorState
    ]
):
    def _reset_state(self) -> None:
        self._state.output = 0

    async def _produce(self) -> SampleTriggerMessage:
        await asyncio.sleep(self.settings.publish_period)
        out_msg = SampleTriggerMessage(
            period=self.settings.period, value=self._state.output
        )
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
