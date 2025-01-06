from abc import ABC, abstractmethod
import math
import traceback
import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import GenState

from .util.profile import profile_subpub
from .sampler import SampleMessage
from .util.asio import run_coroutine_sync


# Type variables
SettingsType = typing.TypeVar("SettingsType", bound=ez.Settings)
MessageType = typing.TypeVar("MessageType")
StateType = typing.TypeVar("StateType", bound=ez.State)


class SignalTransformer(typing.Protocol[StateType, SettingsType, MessageType]):
    def check_metadata(self, message: MessageType) -> bool: ...

    def reset(self, message: MessageType) -> None: ...

    @property
    def state(self) -> StateType: ...

    @state.setter
    def state(self, state: StateType) -> None: ...

    def _process(self, message: MessageType) -> MessageType: ...

    def transform(self, message: MessageType) -> MessageType: ...

    @staticmethod
    def stateful_op(
        state: StateType, message: MessageType
    ) -> tuple[StateType, MessageType]: ...


class BaseSignalTransformer(ABC, typing.Generic[StateType, SettingsType, MessageType]):
    """
    Abstract base class implementing common transformer functionality.

    Create a concrete transformer by subclassing this class and implementing the abstract methods.
    Additionally, set the class variable `state_type` to the State class that the transformer will use.
    e.g. `state_type = MyStateClass`
    """

    state_type: typing.Type[StateType]

    def __init__(self, settings: SettingsType):
        self.settings = settings
        self._state: StateType = self.state_type()

    @abstractmethod
    def check_metadata(self, message: MessageType) -> bool: ...

    @abstractmethod
    def reset(self, message: MessageType) -> None: ...

    @property
    def state(self) -> StateType:
        return self._state

    @state.setter
    def state(self, state: StateType) -> None:
        if state is not None:
            self._state = state

    @abstractmethod
    def _process(self, message: MessageType) -> MessageType: ...

    def transform(self, message: MessageType) -> MessageType:
        if self.check_metadata(message):
            self.reset(message)

        return self._process(message)

    def stateful_op(
        self, state: StateType, message: MessageType
    ) -> tuple[StateType, MessageType]:
        self.state = state
        result = self.transform(message)
        return self.state, result

    def __iter__(self):
        self._state: StateType = self.state_type()
        return self

    def send(self, message: MessageType) -> MessageType:
        return self.transform(message)


class BaseSignalTransformerUnit(
    ez.Unit, typing.Generic[StateType, SettingsType, MessageType]
):
    """
    Implement a new Unit as follows:

    class CustomUnit(BaseSignalTransformerUnit[
        CustomTransformerState        # StateType
        CustomTransformerSettings,    # SettingsType
        AxisArray,                    # MessageType
    ]):
        SETTINGS = CustomTransformerSettings
        transformer_type = CustomTransformer

    ... that's all!

    Where CustomTransformerState, CustomTransformerSettings, and CustomTransformer
    are custom implementations of ez.State, ez.Settings, and BaseSignalTransformer, respectively.
    """

    INPUT_SIGNAL = ez.InputStream(MessageType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageType)
    INPUT_SETTINGS = ez.InputStream(SettingsType)

    # Class variable that concrete classes will override
    transformer_type: typing.Type[
        SignalTransformer[StateType, SettingsType, MessageType]
    ]

    async def initialize(self) -> None:
        self.transformer = self.create_transformer()

    def create_transformer(
        self,
    ) -> SignalTransformer[StateType, SettingsType, MessageType]:
        """Create the transformer instance from settings."""
        # return self.transformer_type(**dataclasses.asdict(self.SETTINGS))
        return self.transformer_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: SettingsType) -> None:
        self.apply_settings(msg)
        self.create_transformer()

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_signal(self, message: MessageType) -> typing.AsyncGenerator:
        try:
            ret = self.transformer.transform(message)
            if math.prod(ret.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, ret
        except Exception:
            ez.logger.info(traceback.format_exc())


class AdaptiveSignalTransformer(
    SignalTransformer, typing.Protocol[StateType, SettingsType, MessageType]
):
    def partial_fit(self, message: SampleMessage) -> None:
        """Update transformer state using labeled training data.

        This method should update the internal state/parameters of the transformer
        based on the provided labeled samples, without performing any transformation.
        """
        ...


class BaseAdaptiveSignalTransformer(
    BaseSignalTransformer, ABC, typing.Generic[StateType, SettingsType, MessageType]
):
    def send(self, message: MessageType | SampleMessage) -> MessageType:
        if hasattr(message, "trigger"):  # SampleMessage
            # y = message.trigger.value.data
            # X = message.sample.data
            self.partial_fit(message)
            message = message.sample
            # TODO: Slice message so it is empty.
        return self.transform(message)


class BaseAdaptiveSignalTransformerUnit(
    BaseSignalTransformerUnit, typing.Generic[StateType, SettingsType, MessageType]
):
    INPUT_SAMPLE = ez.InputStream(SampleMessage)

    transformer_type: typing.Type[
        AdaptiveSignalTransformer[StateType, SettingsType, MessageType]
    ]

    @ez.subscriber(INPUT_SAMPLE)
    async def on_sample(self, msg: SampleMessage) -> None:
        self.transformer.partial_fit(msg)


class AsyncSignalTransformer(
    SignalTransformer, typing.Protocol[StateType, SettingsType, MessageType]
):
    async def _aprocess(self, message: MessageType) -> MessageType: ...

    async def atransform(self, message: MessageType) -> MessageType: ...


class BaseAsyncSignalTransformer(
    BaseSignalTransformer, ABC, typing.Generic[StateType, SettingsType, MessageType]
):
    def process(self, message: MessageType) -> MessageType:
        return run_coroutine_sync(self._aprocess(message))

    @abstractmethod
    async def _aprocess(self, message: MessageType) -> MessageType: ...

    async def atransform(self, message: MessageType) -> MessageType:
        if self.check_metadata(message):
            self.reset(message)
        return await self._aprocess(message)

    async def asend(self, message: MessageType) -> MessageType:
        return await self.atransform(message)


class BaseAsyncSignalTransformerUnit(
    BaseSignalTransformerUnit, typing.Generic[StateType, SettingsType, MessageType]
):
    INPUT_SIGNAL = ez.InputStream(MessageType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageType)
    transformer_type: typing.Type[
        AsyncSignalTransformer[StateType, SettingsType, MessageType]
    ]

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_signal(self, message: MessageType) -> typing.AsyncGenerator:
        try:
            ret = await self.transformer.atransform(message)
            if math.prod(ret.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, ret
        except Exception:
            ez.logger.info(traceback.format_exc())


class GenAxisArray(ez.Unit):
    STATE = GenState

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    INPUT_SETTINGS = ez.InputStream(ez.Settings)

    async def initialize(self) -> None:
        self.construct_generator()

    # Method to be implemented by subclasses to construct the specific generator
    def construct_generator(self):
        raise NotImplementedError

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: ez.Settings) -> None:
        self.apply_settings(msg)
        self.construct_generator()

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        try:
            ret = self.STATE.gen.send(message)
            if math.prod(ret.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, ret
        except (StopIteration, GeneratorExit):
            ez.logger.debug(f"Generator closed in {self.address}")
        except Exception:
            ez.logger.info(traceback.format_exc())
