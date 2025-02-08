from abc import ABC, abstractmethod
import math
import pickle
import traceback
import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import GenState

from .util.profile import profile_subpub
from .sampler import SampleMessage
from .util.asio import run_coroutine_sync


# --- All processor state classes must inherit from this or at least have .hash ---
class ProcessorState(ez.State):
    hash: int = 0


# --- Type variables for protocols and processors ---
MessageType = typing.TypeVar("MessageType")
SettingsType = typing.TypeVar("SettingsType", bound=ez.Settings)
StateType = typing.TypeVar("StateType", bound=ProcessorState)


# --- Protocols for processors ---
class Processor(typing.Protocol[SettingsType, MessageType]):
    """
    Protocol for processors.
    You probably will not implement this protocol directly.
    Refer instead to the less ambiguous Consumer and Transformer protocols.
    """

    def __call__(self, message: MessageType) -> typing.Optional[MessageType]: ...


class Consumer(Processor[SettingsType, MessageType], typing.Protocol):
    """Protocol for consumers that receive messages but do not return a result."""

    def __call__(self, message: MessageType) -> None: ...


class Transformer(Processor[SettingsType, MessageType], typing.Protocol):
    """Protocol for transformers that receive messages and return a result of the same class."""

    def __call__(self, message: MessageType) -> MessageType: ...


class StatefulProcessor(typing.Protocol[SettingsType, MessageType, StateType]):
    """
    Base protocol for _stateful_ message processors.
    You probably will not implement this protocol directly.
    Refer instead to the less ambiguous StatefulConsumer and StatefulTransformer protocols.
    """

    @property
    def state(self) -> StateType: ...

    @state.setter
    def state(self, state: StateType | bytes | None) -> None: ...

    def __call__(self, message: MessageType) -> typing.Optional[MessageType]: ...

    def stateful_op(
        self,
        state: StateType,
        message: MessageType,
    ) -> tuple[StateType, typing.Optional[MessageType]]: ...


class StatefulConsumer(
    StatefulProcessor[SettingsType, MessageType, StateType], typing.Protocol
):
    """Protocol specifically for processors that consume messages without producing output."""

    def __call__(self, message: MessageType) -> None: ...

    def stateful_op(
        self,
        state: StateType,
        message: MessageType,
    ) -> tuple[StateType, None]: ...

    """
    Note: The return type is still a tuple even though the second entry is always None.
    This is intentional so we can use the same protocol for both consumers and transformers,
    and chain them together in a pipeline (e.g., `CompositeProcessor`).
    """


class StatefulTransformer(
    StatefulProcessor[SettingsType, MessageType, StateType], typing.Protocol
):
    """
    Protocol specifically for processors that transform messages.
    """

    def __call__(self, message: MessageType) -> MessageType: ...

    def stateful_op(
        self,
        state: StateType,
        message: MessageType,
    ) -> tuple[StateType, MessageType]: ...


class AdaptiveTransformer(
    StatefulTransformer, typing.Protocol[SettingsType, MessageType, StateType]
):
    def partial_fit(self, message: SampleMessage) -> None:
        """Update transformer state using labeled training data.

        This method should update the internal state/parameters of the transformer
        based on the provided labeled samples, without performing any transformation.
        """
        ...


class AsyncTransformer(
    StatefulTransformer, typing.Protocol[SettingsType, MessageType, StateType]
):
    async def __acall__(self, message: MessageType) -> MessageType: ...


# --- Base implementation classes for processors ---
class BaseProcessor(ABC, typing.Generic[SettingsType, MessageType]):
    """
    Base class for processors.
    You probably do not want to inherit from this class directly.
    Refer instead to the more specific base classes.
    If your operation is stateless then use BaseConsumer or BaseTransformer for operations that
    do not or do return a result, respectively.

    If your operation is stateful then refer to `BaseStatefulProcessor`.
    If your operation requires metadata checking and state reset, then refer to `BaseHashingProcessor`.
    """

    def __init__(self, *args, settings: typing.Optional[SettingsType] = None, **kwargs):
        settings_type = typing.get_args(self.__orig_bases__[0])[0]
        if settings is None:
            if len(args) > 0 and isinstance(args[0], settings_type):
                settings = args[0]
            elif len(args) > 0 or len(kwargs) > 0:
                settings = settings_type(*args, **kwargs)
            else:
                settings = settings_type()
        self.settings = settings

    @abstractmethod
    def _process(self, message: MessageType) -> typing.Optional[MessageType]: ...

    def __call__(self, message: MessageType) -> typing.Optional[MessageType]:
        # Note: We use the indirection to `_process` because this allows us to
        #  modify __call__ in derived classes with common functionality while
        #  minimizing the boilerplate code in derived classes as they only need to
        #  implement `_process`.
        return self._process(message)

    def send(self, message: MessageType) -> typing.Optional[MessageType]:
        """Alias for __call__."""
        return self(message)


class BaseConsumer(BaseProcessor[SettingsType, MessageType]):
    """
    Base class for consumers -- processors that receive messages but don't produce output.
    This base overrides type annotations of BaseProcessor.
    """

    @abstractmethod
    def _process(self, message: MessageType) -> None: ...

    def __call__(self, message: MessageType) -> None:
        super().__call__(message)


class BaseTransformer(BaseProcessor[SettingsType, MessageType]):
    """
    Base class for transformers -- processors which receive messages and produce output.
    This base simply overrides type annotations of BaseProcessor.
    """

    @abstractmethod
    def _process(self, message: MessageType) -> MessageType: ...

    def __call__(self, message: MessageType) -> MessageType:
        return super().__call__(message)


class BaseStatefulProcessor(
    BaseProcessor[SettingsType, MessageType],
    typing.Generic[SettingsType, MessageType, StateType],
):
    """
    Base class implementing common stateful processor functionality.
    You probably do not want to inherit from this class directly.
    Refer instead to the more specific base classes.
    Use BaseStatefulConsumer for operations that do not return a result,
    or BaseStatefulTransformer for operations that do return a result.
    """

    # TODO: Enforce that StateType has .hash: int field.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, state_type = typing.get_args(self.__orig_bases__[0])
        self._state: StateType = state_type()

    @property
    def state(self) -> StateType:
        return self._state

    @state.setter
    def state(self, state: StateType | bytes | None) -> None:
        if state is not None:
            if isinstance(state, bytes):
                self._state = pickle.loads(state)
            else:
                self._state = state

    def _hash_message(self, message: MessageType) -> int:
        """
        Check if the message metadata indicates a need for state reset.

        This method is not abstract because there are some processors that might only
        need to reset once but are otherwise insensitive to the message structure.

        For example, an activation function that benefits greatly from pre-computed values should
        do this computation in `_reset_state` and attach those values to the processor state,
        but if it e.g. operates elementwise on the input then it doesn't care if the incoming
        data changes shape or sample rate so you don't need to reset again.

        All processors' initial state should have `.hash = 0` then by returning `-1` here
        we force an update on the first message.
        """
        return -1

    @abstractmethod
    def _reset_state(self, message: MessageType) -> None:
        """
        Reset internal state based on new message metadata.
        This method will only be called when there is a significant change in the message metadata,
        such as sample rate or shape (criteria defined by `_hash_message`), and not for every message,
        so use it to do all the expensive pre-allocation and caching of variables that can speed up
        the processing of subsequent messages in `_process`.
        """
        ...

    @abstractmethod
    def _process(self, message: MessageType) -> typing.Optional[MessageType]: ...

    def __call__(self, message: MessageType) -> typing.Optional[MessageType]:
        msg_hash = self._hash_message(message)
        if msg_hash != self.state.hash:
            self._reset_state(message)
            self.state.hash = msg_hash
        return self._process(message)

    def send(self, message: MessageType) -> typing.Optional[MessageType]:
        """Alias for __call__."""
        return self(message)

    def stateful_op(
        self,
        state: StateType,
        message: MessageType,
    ) -> tuple[StateType, typing.Optional[MessageType]]:
        self.state = state
        result = self(message)
        return self.state, result


class BaseStatefulConsumer(BaseStatefulProcessor[SettingsType, MessageType, StateType]):
    """
    Base class for stateful message consumers that don't produce output.
    This class merely overrides the type annotations of BaseStatefulProcessor.
    """

    @abstractmethod
    def _process(self, message: MessageType) -> None: ...

    def __call__(self, message: MessageType) -> None:
        super().__call__(message)

    def stateful_op(
        self,
        state: StateType,
        message: MessageType,
    ) -> tuple[StateType, None]:
        state, _ = super().stateful_op(state, message)
        return state, None


class BaseStatefulTransformer(
    BaseStatefulProcessor[SettingsType, MessageType, StateType]
):
    """
    Base class for stateful message transformers that produce output.
    This class merely overrides the type annotations of BaseStatefulProcessor.
    """

    @abstractmethod
    def _process(self, message: MessageType) -> MessageType: ...

    def __call__(self, message: MessageType) -> MessageType:
        return super().__call__(message)

    def stateful_op(
        self,
        state: StateType,
        message: MessageType,
    ) -> tuple[StateType, MessageType]:
        return super().stateful_op(state, message)


class BaseAdaptiveTransformer(
    BaseStatefulTransformer, ABC, typing.Generic[SettingsType, MessageType, StateType]
):
    @abstractmethod
    def partial_fit(self, message: SampleMessage) -> None: ...

    def __call__(self, message: typing.Union[MessageType, SampleMessage]) -> typing.Optional[MessageType]:
        """
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


class BaseAsyncTransformer(
    BaseStatefulTransformer, ABC, typing.Generic[SettingsType, MessageType, StateType]
):
    async def __acall__(self, message: MessageType) -> MessageType:
        # Note: In Python 3.12, we can invoke this with `await obj(message)`
        # Earlier versions must be explicit: `await obj.__acall__(message)`
        if self._hash_message(message) != self.state.hash:
            self._reset_state(message)
        return await self._aprocess(message)

    def __call__(self, message: MessageType) -> MessageType:
        # Override (synchronous) __call__ to run coroutine `aprocess`.
        return run_coroutine_sync(self.__acall__(message))

    @abstractmethod
    async def _aprocess(self, message: MessageType) -> MessageType: ...

    # Alias asend = __acall__
    async def asend(self, message: MessageType) -> MessageType:
        return await self.__acall__(message)


# Composite processor for building pipelines
class CompositeProcessor(BaseProcessor[SettingsType, MessageType]):
    """
    A processor that chains multiple processor together.
    The individual processors may be stateless or stateful.
    The last processor may be a consumer or a transformer; all but the last must be transformers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._procs = self.__class__._initialize_processors(self.settings)

    @staticmethod
    @abstractmethod
    def _initialize_processors(
        settings: SettingsType,
    ) -> dict[str, StatefulProcessor]: ...

    @property
    def state(self) -> dict[str, ProcessorState]:
        return {k: proc.state for k, proc in self._procs.items()}

    @state.setter
    def state(self, state: dict[str, ProcessorState] | bytes | None) -> None:
        if state is not None:
            if isinstance(state, bytes):
                state = pickle.loads(state)
            for k, v in state.items():
                self._procs[k].state = v

    def _process(self, message: MessageType) -> MessageType:
        result = message
        for k, proc in self._procs.items():
            result = proc.send(result)  # `send` allows mixing with legacy generators
        return result

    def stateful_op(
        self,
        state: dict[str, ProcessorState],
        message: MessageType,
    ) -> tuple[dict[str, ProcessorState], typing.Optional[MessageType]]:
        result = message
        for k, proc in self._procs.items():
            if hasattr(proc, "stateful_op"):
                state[k], result = proc.stateful_op(state[k], result)
            else:
                result = proc(result)
        return state, result


# --- Type variables for protocols and processors ---
ConsumerType = typing.TypeVar(
    "ConsumerType", bound=typing.Union[BaseConsumer, BaseStatefulConsumer]
)
TransformerType = typing.TypeVar(
    "TransformerType",
    bound=typing.Union[BaseTransformer, BaseStatefulTransformer, CompositeProcessor],
)
AdaptiveTransformerType = typing.TypeVar(
    "AdaptiveTransformerType", bound=BaseAdaptiveTransformer
)
AsyncTransformerType = typing.TypeVar(
    "AsyncTransformerType", bound=BaseAsyncTransformer
)


# --- Base classes for ezmsg Unit with specific processing capabilities ---
class BaseConsumerUnit(
    ez.Unit, typing.Generic[SettingsType, MessageType, ConsumerType]
):
    """
    Base class for consumer units -- i.e. units that receive messages but do not return results.
    Implement a new Unit as follows:

    class CustomUnit(BaseConsumerUnit[
        CustomConsumerSettings,    # SettingsType
        AxisArray,                 # MessageType
        CustomConsumer,            # ConsumerType
    ]):
        SETTINGS = CustomConsumerSettings

    ... that's all!

    Where CustomConsumerSettings, and CustomConsumer
    are custom implementations of ez.Settings, and BaseConsumer or BaseStatefulConsumer, respectively.
    """

    INPUT_SIGNAL = ez.InputStream(MessageType)
    INPUT_SETTINGS = ez.InputStream(SettingsType)

    async def initialize(self) -> None:
        self.create_processor()

    def create_processor(self):
        # self.processor: StatefulProcessor[SettingsType, MessageType, StateType]
        """Create the transformer instance from settings."""
        processor_type = typing.get_args(self.__orig_bases__[0])[2]
        # return transformer_type(**dataclasses.asdict(self.SETTINGS))
        self.processor = processor_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: SettingsType) -> None:
        """
        Receive a settings message, override self.SETTINGS, and re-create the processor.
        Child classes that wish to have fine-grained control over whether the
        core processor resets on settings changes should override this method.

        Args:
            msg: a settings message.
        """
        self.apply_settings(msg)
        self.create_processor()

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    async def on_signal(self, message: MessageType):
        """
        Consume the message.
        Args:
            message:
        """
        try:
            self.processor(message)
        except Exception:
            ez.logger.info(traceback.format_exc())


class BaseTransformerUnit(
    BaseConsumerUnit,
    typing.Generic[SettingsType, MessageType, TransformerType],
):
    """
    Implement a new Unit as follows:

    class CustomUnit(TransformerUnit[
        CustomTransformerSettings,    # SettingsType
        AxisArray,                    # MessageType
        CustomTransformer,            # TransformerType
    ]):
        SETTINGS = CustomTransformerSettings

    ... that's all!

    Where CustomTransformerSettings, and CustomTransformer
    are custom implementations of ez.Settings, and a transformer type, respectively.
    Eligible base transformer types: BaseTransformer, BaseStatefulTransformer, CompositeProcessor
    """

    OUTPUT_SIGNAL = ez.OutputStream(MessageType)

    @ez.subscriber(BaseConsumerUnit.INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: MessageType) -> typing.AsyncGenerator:
        try:
            result = self.processor(message)
            if result is not None and math.prod(result.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, result
        except Exception as e:
            ez.logger.info(f"{traceback.format_exc()} - {e}")


class BaseAdaptiveTransformerUnit(
    BaseTransformerUnit,
    typing.Generic[SettingsType, MessageType, AdaptiveTransformerType],
):
    INPUT_SAMPLE = ez.InputStream(SampleMessage)

    @ez.subscriber(INPUT_SAMPLE)
    async def on_sample(self, msg: SampleMessage) -> None:
        self.processor.partial_fit(msg)


class BaseAsyncTransformerUnit(
    BaseTransformerUnit,
    typing.Generic[SettingsType, MessageType, AsyncTransformerType],
):
    @ez.subscriber(BaseConsumerUnit.INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(BaseTransformerUnit.OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: MessageType) -> typing.AsyncGenerator:
        try:
            # Python >= 3.12: ret = await self.processor(message)
            ret = await self.processor.__acall__(message)
            if math.prod(ret.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, ret
        except Exception as e:
            ez.logger.info(f"{traceback.format_exc()} - {e}")


# Legacy class
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
