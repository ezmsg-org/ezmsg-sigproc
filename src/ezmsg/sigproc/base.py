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


class StatefulProcessor(typing.Protocol[StateType, SettingsType, MessageType]):
    def check_metadata(self, message: MessageType) -> bool: ...

    def reset(self, message: MessageType) -> None: ...

    @property
    def state(self) -> StateType: ...

    @state.setter
    def state(self, state: StateType) -> None: ...

    def __call__(self, message: MessageType): ...

    @staticmethod
    def stateful_op(
        state: StateType, message: MessageType
    ) -> tuple[StateType]: ...


class BaseStatefulProcessor(ABC, typing.Generic[StateType, SettingsType, MessageType]):
    """
    Abstract base class implementing common transformer functionality.

    Create a concrete transformer by subclassing this class and implementing the abstract methods.
    """

    def __init__(self, *args, settings: typing.Optional[SettingsType] = None, **kwargs):
        state_type, settings_type, _ = typing.get_args(self.__orig_bases__[0])
        if settings is None:
            if len(args) > 0 and isinstance(args[0], settings_type):
                settings = args[0]
            elif len(args) > 0 or len(kwargs) > 0:
                settings = settings_type(*args, **kwargs)
            else:
                settings = settings_type()
        self.settings = settings
        self._state: StateType = state_type()

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
    def _process(self, message: MessageType): ...

    def __call__(self, message: MessageType):
        if self.check_metadata(message):
            self.reset(message)
        return self._process(message)

    def stateful_op(
        self, state: StateType, message: MessageType
    ) -> StateType:
        self.state = state
        self.process(message)
        return self.state

    def __iter__(self):
        state_type = typing.get_args(self.__orig_bases__[0])[0]
        self._state: StateType = state_type()
        return self

    def send(self, message: MessageType):
        # Alias method name
        return self(message)


ProcessorType = typing.TypeVar("ProcessorType", bound=BaseStatefulProcessor)


class BaseProcessorUnit(
    ez.Unit, typing.Generic[StateType, SettingsType, MessageType, ProcessorType]
):
    """
    Implement a new Unit as follows:

    class CustomUnit(BaseProcessorUnit[
        CustomTransformerState        # StateType
        CustomTransformerSettings,    # SettingsType
        AxisArray,                    # MessageType
    ]):
        SETTINGS = CustomTransformerSettings

    ... that's all!

    Where CustomTransformerState, CustomTransformerSettings, and CustomTransformer
    are custom implementations of ez.State, ez.Settings, and BaseStatefulProcessor, respectively.
    """
    INPUT_SIGNAL = ez.InputStream(MessageType)
    INPUT_SETTINGS = ez.InputStream(SettingsType)

    async def initialize(self) -> None:
        self.create_processor()

    def create_processor(self):
        #  -> StatefulProcessor[StateType, SettingsType, MessageType]
        """Create the transformer instance from settings."""
        processor_type = typing.get_args(self.__orig_bases__[0])[3]
        # return transformer_type(**dataclasses.asdict(self.SETTINGS))
        self.processor = processor_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: SettingsType) -> None:
        self.apply_settings(msg)
        self.create_transformer()

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    async def on_signal(self, message: MessageType) -> typing.AsyncGenerator:
        try:
            self.processor(message)
        except Exception:
            ez.logger.info(traceback.format_exc())


class SignalTransformer(StatefulProcessor, typing.Protocol[StateType, SettingsType, MessageType]):
    # Override __call__ annotation because _process now returns a message.
    def __call__(self, message: MessageType) -> MessageType: ...

    # Override stateful_op from parent protocol to return a message.
    @staticmethod
    def stateful_op(
        state: StateType, message: MessageType
    ) -> tuple[StateType, MessageType]: ...


class BaseSignalTransformer(
    BaseStatefulProcessor, ABC, typing.Generic[StateType, SettingsType, MessageType]
):
    """
    Abstract base class implementing common transformer functionality.

    Create a concrete transformer by subclassing this class and implementing the abstract methods.
    """
    @abstractmethod
    def _process(self, message: MessageType) -> MessageType: ...

    def stateful_op(
        self, state: StateType, message: MessageType
    ) -> tuple[StateType, MessageType]:
        self.state = state
        result = self(message)
        return self.state, result


TransformerType = typing.TypeVar("TransformerType", bound=BaseSignalTransformer)


class BaseSignalTransformerUnit(
    BaseProcessorUnit, typing.Generic[StateType, SettingsType, MessageType, TransformerType]
):
    """
    Implement a new Unit as follows:

    class CustomUnit(BaseSignalTransformerUnit[
        CustomTransformerState        # StateType
        CustomTransformerSettings,    # SettingsType
        AxisArray,                    # MessageType
    ]):
        SETTINGS = CustomTransformerSettings

    ... that's all!

    Where CustomTransformerState, CustomTransformerSettings, and CustomTransformer
    are custom implementations of ez.State, ez.Settings, and BaseSignalTransformer, respectively.
    """
    INPUT_SIGNAL = ez.InputStream(MessageType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageType)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: MessageType) -> typing.AsyncGenerator:
        try:
            ret = self.processor(message)
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
    def __call__(self, message: typing.Union[MessageType, SampleMessage]) -> None:
        """"
        Adapt transformer with training data (and optionally labels)
        in SampleMessage

        Args:
            message: An instance of SampleMessage with optional
             labels (y) in message.trigger.value.data and
             data (X) in message.sample.data

        Returns: None
        """
        if hasattr(message, "trigger"):
            return self.partial_fit(message)
        return super().__call__(message)


class BaseAdaptiveSignalTransformerUnit(
    BaseSignalTransformerUnit, typing.Generic[StateType, SettingsType, MessageType, TransformerType]
):
    INPUT_SAMPLE = ez.InputStream(SampleMessage)

    @ez.subscriber(INPUT_SAMPLE)
    async def on_sample(self, msg: SampleMessage) -> None:
        self.processor(msg)


class AsyncSignalTransformer(
    SignalTransformer, typing.Protocol[StateType, SettingsType, MessageType]
):
    async def _aprocess(self, message: MessageType) -> MessageType: ...


class BaseAsyncSignalTransformer(
    BaseSignalTransformer, ABC, typing.Generic[StateType, SettingsType, MessageType]
):
    async def __acall__(self, message: MessageType) -> MessageType:
        # Note: In Python 3.12, we can invoke this with `await obj(message)`
        # Earlier versions must be explicit: `await obj.__acall__(message)`
        if self.check_metadata(message):
            self.reset(message)
        return await self._aprocess(message)

    def __call__(self, message: MessageType) -> MessageType:
        # Override (synchronous) __call__ to run coroutine `aprocess`.
        return run_coroutine_sync(self.__acall__(message))

    @abstractmethod
    async def _aprocess(self, message: MessageType) -> MessageType: ...

    # Alias asend = __acall__
    async def asend(self, message: MessageType) -> MessageType:
        return await self.__acall__(message)


class BaseAsyncSignalTransformerUnit(
    BaseSignalTransformerUnit, typing.Generic[StateType, SettingsType, MessageType, TransformerType]
):
    INPUT_SIGNAL = ez.InputStream(MessageType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageType)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: MessageType) -> typing.AsyncGenerator:
        try:
            # Python >= 3.12: ret = await self.processor(message)
            ret = await self.processor.__acall__(message)
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
        """
        Update unit settings and reset generator.
        Note: Not all units will require a full reset with new settings.
        Override this method to implement a selective reset.

        Args:
            msg: Instance of SETTINGS object.
        """
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
