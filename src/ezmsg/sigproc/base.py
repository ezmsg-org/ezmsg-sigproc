from abc import ABC, abstractmethod
import inspect
import math
import pickle
import traceback
import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import GenState

from .util.profile import profile_subpub
from .util.message import SampleMessage
from .util.asio import run_coroutine_sync


# --- All processor state classes must inherit from this or at least have .hash ---
class ProcessorState(ez.State):
    hash: int = -1


# --- Type variables for protocols and processors ---
MessageInType = typing.TypeVar("MessageInType")
MessageOutType = typing.TypeVar("MessageOutType")
SettingsType = typing.TypeVar("SettingsType", bound=ez.Settings)
StateType = typing.TypeVar("StateType", bound=ProcessorState)


# --- Protocols for processors ---
class Processor(typing.Protocol[SettingsType, MessageInType, MessageOutType]):
    """
    Protocol for processors.
    You probably will not implement this protocol directly.
    Refer instead to the less ambiguous Consumer and Transformer protocols, and the base classes
    in this module which implement them.

    Note: In Python 3.12+, we can invoke `__acall__` directly using `await obj(message)`,
     but to support earlier versions we need to use `await obj.__acall__(message)`.
    """

    def __call__(self, message: MessageInType) -> typing.Optional[MessageOutType]: ...
    async def __acall__(
        self, message: MessageInType
    ) -> typing.Optional[MessageOutType]: ...


class Producer(typing.Protocol[SettingsType, MessageOutType]):
    """
    Protocol for producers that generate messages.
    """

    def __call__(self) -> MessageOutType: ...
    async def __acall__(self) -> MessageOutType: ...


class Consumer(Processor[SettingsType, MessageInType, None], typing.Protocol):
    """
    Protocol for consumers that receive messages but do not return a result.
    """

    def __call__(self, message: MessageInType) -> None: ...
    async def __acall__(self, message: MessageInType) -> None: ...


class Transformer(
    Processor[SettingsType, MessageInType, MessageOutType], typing.Protocol
):
    """Protocol for transformers that receive messages and return a result of the same class."""

    def __call__(self, message: MessageInType) -> MessageOutType: ...
    async def __acall__(self, message: MessageInType) -> MessageOutType: ...


class StatefulProcessor(
    typing.Protocol[SettingsType, MessageInType, MessageOutType, StateType]
):
    """
    Base protocol for _stateful_ message processors.
    You probably will not implement this protocol directly.
    Refer instead to the less ambiguous StatefulConsumer and StatefulTransformer protocols.
    """

    @property
    def state(self) -> StateType: ...

    @state.setter
    def state(self, state: StateType | bytes | None) -> None: ...

    def __call__(self, message: MessageInType) -> typing.Optional[MessageOutType]: ...
    async def __acall__(
        self, message: MessageInType
    ) -> typing.Optional[MessageOutType]: ...

    def stateful_op(
        self,
        state: StateType,
        message: MessageInType,
    ) -> tuple[StateType, typing.Optional[MessageOutType]]: ...


class StatefulProducer(typing.Protocol[SettingsType, MessageOutType, StateType]):
    """Protocol for producers that generate messages without consuming inputs."""

    @property
    def state(self) -> StateType: ...

    @state.setter
    def state(self, state: StateType | bytes | None) -> None: ...

    def __call__(self) -> MessageOutType: ...
    async def __acall__(self) -> MessageOutType: ...

    async def stateful_op(
        self,
        state: StateType,
    ) -> tuple[StateType, MessageOutType]: ...


class StatefulConsumer(
    StatefulProcessor[SettingsType, MessageInType, None, StateType], typing.Protocol
):
    """Protocol specifically for processors that consume messages without producing output."""

    def __call__(self, message: MessageInType) -> None: ...
    async def __acall__(self, message: MessageInType) -> None: ...

    def stateful_op(
        self,
        state: StateType,
        message: MessageInType,
    ) -> tuple[StateType, None]: ...

    """
    Note: The return type is still a tuple even though the second entry is always None.
    This is intentional so we can use the same protocol for both consumers and transformers,
    and chain them together in a pipeline (e.g., `CompositeProcessor`).
    """


class StatefulTransformer(
    StatefulProcessor[SettingsType, MessageInType, MessageOutType, StateType],
    typing.Protocol,
):
    """
    Protocol specifically for processors that transform messages.
    """

    def __call__(self, message: MessageInType) -> MessageOutType: ...
    async def __acall__(self, message: MessageInType) -> MessageOutType: ...

    def stateful_op(
        self,
        state: StateType,
        message: MessageInType,
    ) -> tuple[StateType, MessageOutType]: ...


class AdaptiveTransformer(
    StatefulTransformer,
    typing.Protocol[SettingsType, MessageInType, MessageOutType, StateType],
):
    def partial_fit(self, message: SampleMessage) -> None:
        """Update transformer state using labeled training data.

        This method should update the internal state/parameters of the transformer
        based on the provided labeled samples, without performing any transformation.
        """
        ...

    async def apartial_fit(self, message: SampleMessage) -> None: ...


# --- Base implementation classes for processors ---


def _unify_settings(
    obj: typing.Any, settings: SettingsType, *args, **kwargs
) -> SettingsType:
    settings_type = typing.get_args(obj.__orig_bases__[0])[0]
    if settings is None:
        if len(args) > 0 and isinstance(args[0], settings_type):
            settings = args[0]
        elif len(args) > 0 or len(kwargs) > 0:
            settings = settings_type(*args, **kwargs)
        else:
            settings = settings_type()
    return settings


class BaseProcessor(ABC, typing.Generic[SettingsType, MessageInType, MessageOutType]):
    """
    Base class for processors. You probably do not want to inherit from this class directly.
    Refer instead to the more specific base classes.
      * Use :obj:`BaseConsumer` or :obj:`BaseTransformer` for ops that return a result or not, respectively.
      * Use :obj:`BaseStatefulProcessor` and its children for operations that require state.

    Note that `BaseProcessor` and its children are sync by default. If you need async by defualt,
      then override the async methods and call them from the sync methods. Look to `BaseProducer` for examples of
      calling async methods from sync methods.
    """

    @classmethod
    def get_settings_type(cls) -> typing.Type[SettingsType]:
        return typing.get_args(cls.__orig_bases__[0])[0]

    @classmethod
    def get_message_type(cls, dir: str) -> typing.Type[MessageInType | MessageOutType]:
        return typing.get_args(cls.__orig_bases__[0])[1 if dir == "in" else 2]

    def __init__(self, *args, settings: typing.Optional[SettingsType] = None, **kwargs):
        self.settings = _unify_settings(self, settings, *args, **kwargs)

    @abstractmethod
    def _process(self, message: MessageInType) -> typing.Optional[MessageOutType]: ...

    async def _aprocess(
        self, message: MessageInType
    ) -> typing.Optional[MessageOutType]:
        """Override this for native async processing."""
        return self._process(message)

    def __call__(self, message: MessageInType) -> typing.Optional[MessageOutType]:
        # Note: We use the indirection to `_process` because this allows us to
        #  modify __call__ in derived classes with common functionality while
        #  minimizing the boilerplate code in derived classes as they only need to
        #  implement `_process`.
        return self._process(message)

    async def __acall__(
        self, message: MessageInType
    ) -> typing.Optional[MessageOutType]:
        """
        In Python 3.12+, we can invoke this method simply with `await obj(message)`,
        but earlier versions require direct syntax: `await obj.__acall__(message)`.
        """
        return await self._aprocess(message)

    def send(self, message: MessageInType) -> typing.Optional[MessageOutType]:
        """Alias for __call__."""
        return self(message)

    async def asend(self, message: MessageInType) -> typing.Optional[MessageOutType]:
        """Alias for __acall__."""
        return await self.__acall__(message)


class BaseProducer(ABC, typing.Generic[SettingsType, MessageOutType]):
    """
    Base class for producers -- processors that generate messages without consuming inputs.

    Note that `BaseProducer` and its children are async by default, and the sync methods simply wrap
      the async methods. This is the opposite of :obj:`BaseProcessor` and its children which are sync by default.
      These classes are designed this way because it is highly likely that a producer, which (probably) does not
      receive inputs, will require some sort of IO which will benefit from being async.
    """

    @classmethod
    def get_settings_type(cls) -> typing.Type[SettingsType]:
        return typing.get_args(cls.__orig_bases__[0])[0]

    @classmethod
    def get_message_type(cls, dir: str) -> typing.Type[MessageOutType]:
        if dir == "out":
            return typing.get_args(cls.__orig_bases__[0])[1]
        return None

    def __init__(self, *args, settings: typing.Optional[SettingsType] = None, **kwargs):
        self.settings = _unify_settings(self, settings, *args, **kwargs)

    @abstractmethod
    async def _produce(self) -> MessageOutType: ...

    async def __acall__(self) -> MessageOutType:
        return await self._produce()

    def __call__(self) -> MessageOutType:
        # Warning: This is a bit slow. Override this method in derived classes if performance is critical.
        return run_coroutine_sync(self.__acall__())

    def __iter__(self) -> typing.Iterator[MessageOutType]:
        # Make self an iterator
        return self

    async def __anext__(self):
        # So this can be used as an async generator.
        return await self.__acall__()

    def __next__(self) -> MessageOutType:
        # So this can be used as a generator.
        return self()


class BaseConsumer(
    BaseProcessor[SettingsType, MessageInType, None],
    typing.Generic[SettingsType, MessageInType],
):
    """
    Base class for consumers -- processors that receive messages but don't produce output.
    This base simply overrides type annotations of BaseProcessor to remove the outputs.
    (We don't bother overriding `send` and `asend` because those are deprecated.)
    """

    @classmethod
    def get_message_type(cls, dir: str) -> typing.Type[MessageInType]:
        if dir == "in":
            return typing.get_args(cls.__orig_bases__[0])[1]
        return None

    @abstractmethod
    def _process(self, message: MessageInType) -> None: ...

    async def _aprocess(self, message: MessageInType):
        """Override this for native async processing."""
        return self._process(message)

    def __call__(self, message: MessageInType) -> None:
        super().__call__(message)

    async def __acall__(self, message: MessageInType):
        return await super().__acall__(message)


class BaseTransformer(
    BaseProcessor[SettingsType, MessageInType, MessageOutType],
    typing.Generic[SettingsType, MessageInType, MessageOutType],
):
    """
    Base class for transformers -- processors which receive messages and produce output.
    This base simply overrides type annotations of :obj:`BaseProcessor` to indicate that outputs are not optional.
    (We don't bother overriding `send` and `asend` because those are deprecated.)
    """

    @abstractmethod
    def _process(self, message: MessageInType) -> MessageOutType: ...

    async def _aprocess(self, message: MessageInType) -> MessageOutType:
        """Override this for native async processing."""
        return self._process(message)

    def __call__(self, message: MessageInType) -> MessageOutType:
        return super().__call__(message)

    async def __acall__(self, message: MessageInType) -> MessageOutType:
        return await super().__acall__(message)


def _get_state_type(cls):
    for base in cls.__mro__:
        if hasattr(base, "__orig_bases__"):
            for orig_base in base.__orig_bases__:
                if hasattr(orig_base, "__origin__"):
                    args = typing.get_args(orig_base)
                    if (
                        args
                        and hasattr(args[-1], "__name__")
                        and "state" in args[-1].__name__.lower()
                    ):
                        return args[-1]
    return None


class BaseStatefulProcessor(
    BaseProcessor[SettingsType, MessageInType, MessageOutType],
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    """
    Base class implementing common stateful processor functionality.
    You probably do not want to inherit from this class directly.
    Refer instead to the more specific base classes.
    Use BaseStatefulConsumer for operations that do not return a result,
    or BaseStatefulTransformer for operations that do return a result.
    """

    @classmethod
    def get_state_type(cls):
        return _get_state_type(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_type = self.__class__.get_state_type()
        self._state: StateType = state_type()
        # TODO: Enforce that StateType has .hash: int field.

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

    def _hash_message(self, message: MessageInType) -> int:
        """
        Check if the message metadata indicates a need for state reset.

        This method is not abstract because there are some processors that might only
        need to reset once but are otherwise insensitive to the message structure.

        For example, an activation function that benefits greatly from pre-computed values should
        do this computation in `_reset_state` and attach those values to the processor state,
        but if it e.g. operates elementwise on the input then it doesn't care if the incoming
        data changes shape or sample rate so you don't need to reset again.

        All processors' initial state should have `.hash = -1` then by returning `0` here
        we force an update on the first message.
        """
        return 0

    @abstractmethod
    def _reset_state(self, message: MessageInType) -> None:
        """
        Reset internal state based on new message metadata.
        This method will only be called when there is a significant change in the message metadata,
        such as sample rate or shape (criteria defined by `_hash_message`), and not for every message,
        so use it to do all the expensive pre-allocation and caching of variables that can speed up
        the processing of subsequent messages in `_process`.
        """
        ...

    @abstractmethod
    def _process(self, message: MessageInType) -> typing.Optional[MessageOutType]: ...

    def __call__(self, message: MessageInType) -> typing.Optional[MessageOutType]:
        msg_hash = self._hash_message(message)
        if msg_hash != self.state.hash:
            self._reset_state(message)
            self.state.hash = msg_hash
        return self._process(message)

    async def __acall__(
        self, message: MessageInType
    ) -> typing.Optional[MessageOutType]:
        msg_hash = self._hash_message(message)
        if msg_hash != self.state.hash:
            self._reset_state(message)
            self.state.hash = msg_hash
        return await self._aprocess(message)

    def stateful_op(
        self,
        state: StateType,
        message: MessageInType,
    ) -> tuple[StateType, typing.Optional[MessageOutType]]:
        self.state = state
        result = self(message)
        return self.state, result


class BaseStatefulProducer(
    BaseProducer[SettingsType, MessageOutType],
    typing.Generic[SettingsType, MessageOutType, StateType],
):
    """
    Base class implementing common stateful producer functionality.
      Examples of stateful producers are things that require counters, clocks,
      or to cycle through a set of values.

    Unlike BaseStatefulProcessor, this class does not message hashing because there
      are no input messages. We still use self.state.hash to simply track the transition from
      initialization (.hash == -1) to state reset (.hash == 0).
    """

    @classmethod
    def get_state_type(cls):
        return _get_state_type(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # .settings
        state_type = self.__class__.get_state_type()
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

    @abstractmethod
    def _reset_state(self) -> None:
        """
        Reset internal state upon first call.
        """
        ...

    async def __acall__(self) -> MessageOutType:
        if self.state.hash == -1:
            self._reset_state()
            self.state.hash = 0
        return await self._produce()

    def stateful_op(
        self,
        state: StateType,
    ) -> tuple[StateType, MessageOutType]:
        self.state = state  # Update state via setter
        result = self()  # Uses synchronous call
        return self.state, result


class BaseStatefulConsumer(
    BaseStatefulProcessor[SettingsType, MessageInType, None, StateType],
    typing.Generic[SettingsType, MessageInType, StateType],
):
    """
    Base class for stateful message consumers that don't produce output.
    This class merely overrides the type annotations of BaseStatefulProcessor.
    """

    @classmethod
    def get_message_type(cls, dir: str) -> typing.Type[MessageInType]:
        if dir == "in":
            return typing.get_args(cls.__orig_bases__[0])[1]
        return None

    @abstractmethod
    def _process(self, message: MessageInType) -> None: ...

    async def _aprocess(self, message: MessageInType):
        return self._process(message)

    def __call__(self, message: MessageInType) -> None:
        super().__call__(message)

    async def __acall__(self, message: MessageInType):
        return await super().__acall__(message)

    def stateful_op(
        self,
        state: StateType,
        message: MessageInType,
    ) -> tuple[StateType, None]:
        state, _ = super().stateful_op(state, message)
        return state, None


class BaseStatefulTransformer(
    BaseStatefulProcessor[SettingsType, MessageInType, MessageOutType, StateType],
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    """
    Base class for stateful message transformers that produce output.
    This class merely overrides the type annotations of BaseStatefulProcessor.
    """

    @abstractmethod
    def _process(self, message: MessageInType) -> MessageOutType: ...

    async def _aprocess(self, message: MessageInType) -> MessageOutType:
        return self._process(message)

    def __call__(self, message: MessageInType) -> MessageOutType:
        return super().__call__(message)

    async def __acall__(self, message: MessageInType) -> MessageOutType:
        return await super().__acall__(message)

    def stateful_op(
        self,
        state: StateType,
        message: MessageInType,
    ) -> tuple[StateType, MessageOutType]:
        return super().stateful_op(state, message)


class BaseAdaptiveTransformer(
    BaseStatefulTransformer,
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    @abstractmethod
    def partial_fit(self, message: SampleMessage) -> None: ...

    async def apartial_fit(self, message: SampleMessage) -> None:
        """Override me if you need async partial fitting."""
        return self.partial_fit(message)

    def __call__(
        self, message: typing.Union[MessageInType, SampleMessage]
    ) -> typing.Optional[MessageOutType]:
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

    async def __acall__(
        self, message: typing.Union[MessageInType, SampleMessage]
    ) -> typing.Optional[MessageOutType]:
        if hasattr(message, "trigger"):
            return await self.apartial_fit(message)
        return await super().__acall__(message)


class BaseAsyncTransformer(
    BaseStatefulTransformer,
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    """
    This reverses the priority of async and sync methods from :obj:`BaseStatefulTransformer`.
    Whereas in :obj:`BaseStatefulTransformer`, the async methods simply called the sync methods,
    here the sync methods call the async methods, more similar to :obj:`BaseStatefulProducer`.
    """

    def _process(self, message: MessageInType) -> MessageOutType:
        return run_coroutine_sync(self._aprocess(message))

    @abstractmethod
    async def _aprocess(self, message: MessageInType) -> MessageOutType: ...

    def __call__(self, message: MessageInType) -> MessageOutType:
        # Override (synchronous) __call__ to run coroutine `aprocess`.
        return run_coroutine_sync(self.__acall__(message))

    async def __acall__(self, message: MessageInType) -> MessageOutType:
        # Note: In Python 3.12, we can invoke this with `await obj(message)`
        # Earlier versions must be explicit: `await obj.__acall__(message)`
        if self._hash_message(message) != self.state.hash:
            self._reset_state(message)
        return await self._aprocess(message)


# Composite processor for building pipelines
def _get_processor_message_type(
    proc: BaseProcessor | BaseProducer | typing.Generator, dir: str
) -> typing.Optional[type]:
    """Extract the input type from a processor."""
    if inspect.isgenerator(proc):
        gen_func = proc.gi_frame.f_globals[proc.gi_frame.f_code.co_name]
        args = typing.get_args(gen_func.__annotations__.get("return"))
        return args[0] if dir == "out" else args[1]  # yield type / send type
    return proc.__class__.get_message_type(dir)


def _check_message_type_compatibility(type1: type, type2: type) -> bool:
    """
    Check if two types are compatible for message passing.

    Returns True if:
    - Both are None
    - Either is typing.Any
    - One is None and the other is typing.Optional
    - type1 is a subclass of type2, or of the inner type if type2 is Optional

    Args:
        type1: First type to compare
        type2: Second type to compare

    Returns:
        bool: True if the types are compatible, False otherwise
    """
    # Both None is compatible
    if type1 is None and type2 is None:
        return True

    # If either is Any, they are compatible
    if type1 is typing.Any or type2 is typing.Any:
        return True

    # Handle None with Optional
    if type1 is None:
        return hasattr(type2, "__origin__") and type2.__origin__ is typing.Union and type(None) in typing.get_args(type2)
    if type2 is None:
        return hasattr(type1, "__origin__") and type1.__origin__ is typing.Union and type(None) in typing.get_args(type1)

    # Handle Optional types
    if hasattr(type2, "__origin__") and type2.__origin__ is typing.Union and type(None) in typing.get_args(type2):
        # Extract the non-None type from Optional
        non_none_type = next(arg for arg in typing.get_args(type2) if arg is not type(None))
        return issubclass(type1, non_none_type)

    # Regular issubclass check
    return issubclass(type1, type2)


class CompositeProcessor(
    BaseProcessor[SettingsType, MessageInType, MessageOutType],
    typing.Generic[SettingsType, MessageInType, MessageOutType],
):
    """
    A processor that chains multiple processor together in a feedforward non-branching graph.
    The individual processors may be stateless or stateful.
    The first processor may be a producer, the last processor may be a consumer,
     otherwise processors must be transformers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # .settings
        self._procs = self.__class__._initialize_processors(self.settings)

    @staticmethod
    @abstractmethod
    def _initialize_processors(
        settings: SettingsType,
    ) -> dict[str, BaseProducer | BaseProcessor]: ...

    @property
    def state(self) -> dict[str, ProcessorState]:
        return {
            k: proc.state for k, proc in self._procs.items() if hasattr(proc, "state")
        }

    @state.setter
    def state(self, state: dict[str, ProcessorState] | bytes | None) -> None:
        if state is not None:
            if isinstance(state, bytes):
                state = pickle.loads(state)
            for k, v in state.items():
                self._procs[k].state = v

    def _process(self, message: MessageInType | None = None) -> MessageOutType | None:
        """
        Process a message through the pipeline of processors. If the message is None, or no message is provided,
        then it will be assumed that the first processor is a producer and will be called without arguments.
        This will be invoked via `__call__` or `send`.
        We use `__next__` and `send` to allow using legacy generators that have yet to be converted to transformers.

        Warning: All processors will be called using their synchronous API, which may invoke a slow sync->async wrapper
         for processors that are async-first (i.e., children of BaseProducer or BaseAsyncTransformer).
         If you are in an async context, please use instead this object's `asend` or `__acall__`,
         which is much faster for async processors and does not incur penalty on sync processors.
        """
        procs = list(self._procs.values())
        sig = inspect.signature(procs[0].__call__)
        if len(sig.parameters) == 0:
            # Accommodate first processor being a producer (no arguments).
            result = procs[0].__next__()
        else:
            result = procs[0].send(message)

        for proc in procs[1:]:
            result = proc.send(result)
        return result

    async def _aprocess(
        self, message: MessageInType | None = None
    ) -> MessageOutType | None:
        """
        Process a message through the pipeline of processors using their async APIs.
        If the message is None, or no message is provided, then it will be assumed that the first processor
        is a producer and will be called without arguments.
        We use `__anext__` and `asend` to allow using legacy generators that have yet to be converted to transformers.
        """
        procs = list(self._procs.values())
        sig = inspect.signature(procs[0].__acall__)
        if len(sig.parameters) == 0:
            result = await procs[0].__anext__()
        else:
            result = await procs[0].asend(message)
        for proc in procs[1:]:
            result = await proc.asend(result)
        return result

    def stateful_op(
        self,
        state: dict[str, ProcessorState],
        message: MessageInType | None,
    ) -> tuple[dict[str, ProcessorState], MessageOutType | None]:
        result = message
        for k, proc in self._procs.items():
            if hasattr(proc, "stateful_op"):
                state[k], result = proc.stateful_op(state[k], result)
            else:
                result = proc(result)
        return state, result


# --- Type variables for protocols and processors ---
ProducerType = typing.TypeVar(
    "ProducerType", bound=typing.Union[BaseProducer, BaseStatefulProducer]
)
ConsumerType = typing.TypeVar(
    "ConsumerType", bound=typing.Union[BaseConsumer, BaseStatefulConsumer]
)
TransformerType = typing.TypeVar(
    "TransformerType",
    bound=typing.Union[
        BaseTransformer,
        BaseStatefulTransformer,
        BaseAsyncTransformer,
        CompositeProcessor,
    ],
)
AdaptiveTransformerType = typing.TypeVar(
    "AdaptiveTransformerType", bound=BaseAdaptiveTransformer
)


# --- Base classes for ezmsg Unit with specific processing capabilities ---
class BaseProducerUnit(
    ez.Unit, typing.Generic[SettingsType, MessageOutType, ProducerType]
):
    """
    Base class for producer units -- i.e. units that generate messages without consuming inputs.
    Implement a new Unit as follows:

    class CustomUnit(BaseProducerUnit[
        CustomProducerSettings,    # SettingsType
        AxisArray,                 # MessageOutType
        CustomProducer,            # ProducerType
    ]):
        SETTINGS = CustomProducerSettings

    ... that's all!

    Where CustomProducerSettings, and CustomProducer
    are custom implementations of ez.Settings, and BaseProducer or BaseStatefulProducer, respectively.
    """

    INPUT_SETTINGS = ez.InputStream(SettingsType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageOutType)

    async def initialize(self) -> None:
        self.create_producer()

    def create_producer(self):
        # self.producer: ProducerType
        """Create the producer instance from settings."""
        producer_type = typing.get_args(self.__orig_bases__[0])[-1]
        self.producer = producer_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: SettingsType) -> None:
        """
        Receive a settings message, override self.SETTINGS, and re-create the producer.
        Child classes that wish to have fine-grained control over whether the
        core producer resets on settings changes should override this method.

        Args:
            msg: a settings message.
        """
        self.apply_settings(msg)
        self.create_producer()

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        try:
            while True:
                out = await self.producer.__acall__()
                yield self.OUTPUT_SIGNAL, out
        except Exception:
            ez.logger.info(traceback.format_exc())


class BaseConsumerUnit(
    ez.Unit, typing.Generic[SettingsType, MessageInType, ConsumerType]
):
    """
    Base class for consumer units -- i.e. units that receive messages but do not return results.
    Implement a new Unit as follows:

    class CustomUnit(BaseConsumerUnit[
        CustomConsumerSettings,    # SettingsType
        AxisArray,                 # MessageInType
        CustomConsumer,            # ConsumerType
    ]):
        SETTINGS = CustomConsumerSettings

    ... that's all!

    Where CustomConsumerSettings, and CustomConsumer
    are custom implementations of ez.Settings, and BaseConsumer or BaseStatefulConsumer, respectively.
    """

    INPUT_SIGNAL = ez.InputStream(MessageInType)
    INPUT_SETTINGS = ez.InputStream(SettingsType)

    async def initialize(self) -> None:
        self.create_processor()

    def create_processor(self):
        # self.processor: StatefulProcessor[SettingsType, MessageInType, StateType]
        """Create the transformer instance from settings."""
        processor_type = typing.get_args(self.__orig_bases__[0])[-1]
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
    async def on_signal(self, message: MessageInType):
        """
        Consume the message.
        Args:
            message:
        """
        try:
            await self.processor.__acall__(message)
        except Exception:
            ez.logger.info(traceback.format_exc())


class BaseTransformerUnit(
    BaseConsumerUnit[SettingsType, MessageInType, TransformerType],
    typing.Generic[SettingsType, MessageInType, MessageOutType, TransformerType],
):
    """
    Implement a new Unit as follows:

    class CustomUnit(BaseTransformerUnit[
        CustomTransformerSettings,    # SettingsType
        AxisArray,                    # MessageInType
        AxisArray,                    # MessageOutType
        CustomTransformer,            # TransformerType
    ]):
        SETTINGS = CustomTransformerSettings

    ... that's all!

    Where CustomTransformerSettings, and CustomTransformer
    are custom implementations of ez.Settings, and a transformer type, respectively.
    Eligible base transformer types: BaseTransformer, BaseStatefulTransformer, CompositeProcessor
    """

    OUTPUT_SIGNAL = ez.OutputStream(MessageOutType)

    @ez.subscriber(BaseConsumerUnit.INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: MessageInType) -> typing.AsyncGenerator:
        try:
            result = await self.processor.__acall__(message)
            if result is not None and math.prod(result.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, result
        except Exception as e:
            ez.logger.info(f"{traceback.format_exc()} - {e}")


class BaseAdaptiveTransformerUnit(
    BaseTransformerUnit[
        SettingsType, MessageInType, MessageOutType, AdaptiveTransformerType
    ],
    typing.Generic[
        SettingsType, MessageInType, MessageOutType, AdaptiveTransformerType
    ],
):
    INPUT_SAMPLE = ez.InputStream(SampleMessage)

    @ez.subscriber(INPUT_SAMPLE)
    async def on_sample(self, msg: SampleMessage) -> None:
        await self.processor.apartial_fit(msg)


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
