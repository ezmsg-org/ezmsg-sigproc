from abc import ABC, abstractmethod
import dataclasses
import functools
import inspect
import math
import pickle
import traceback
from types import GeneratorType
import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import GenState

from ezmsg.sigproc.util.typeresolution import (
    check_message_type_compatibility,
    resolve_typevar,
)

from .util.profile import profile_subpub
from .util.message import SampleMessage, is_sample_message
from .util.asio import SyncToAsyncGeneratorWrapper, run_coroutine_sync


# --- All processor state classes must inherit from this or at least have .hash ---
# processor_settings = functools.partial(dataclasses.dataclass, frozen=True, init=True)
processor_state = functools.partial(
    dataclasses.dataclass, unsafe_hash=True, frozen=False, init=False
)

# --- Type variables for protocols and processors ---
MessageInType = typing.TypeVar("MessageInType")
MessageOutType = typing.TypeVar("MessageOutType")
SettingsType = typing.TypeVar("SettingsType")
StateType = typing.TypeVar("StateType")


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

    def __call__(self, message: typing.Any) -> typing.Any: ...
    async def __acall__(self, message: typing.Any) -> typing.Any: ...


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

    def __call__(self, message: typing.Any) -> typing.Any: ...
    async def __acall__(self, message: typing.Any) -> typing.Any: ...

    def stateful_op(
        self,
        state: typing.Any,
        message: typing.Any,
    ) -> tuple[typing.Any, typing.Any]: ...


class StatefulProducer(typing.Protocol[SettingsType, MessageOutType, StateType]):
    """Protocol for producers that generate messages without consuming inputs."""

    @property
    def state(self) -> StateType: ...

    @state.setter
    def state(self, state: StateType | bytes | None) -> None: ...

    def __call__(self) -> MessageOutType: ...
    async def __acall__(self) -> MessageOutType: ...

    def stateful_op(
        self,
        state: typing.Any,
    ) -> tuple[typing.Any, typing.Any]: ...


class StatefulConsumer(
    StatefulProcessor[SettingsType, MessageInType, None, StateType], typing.Protocol
):
    """Protocol specifically for processors that consume messages without producing output."""

    def __call__(self, message: MessageInType) -> None: ...
    async def __acall__(self, message: MessageInType) -> None: ...

    def stateful_op(
        self,
        state: tuple[StateType, int],
        message: MessageInType,
    ) -> tuple[tuple[StateType, int], None]: ...

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
        state: tuple[StateType, int],
        message: MessageInType,
    ) -> tuple[tuple[StateType, int], MessageOutType]: ...


class AdaptiveTransformer(StatefulTransformer, typing.Protocol):
    def partial_fit(self, message: SampleMessage) -> None:
        """Update transformer state using labeled training data.

        This method should update the internal state/parameters of the transformer
        based on the provided labeled samples, without performing any transformation.
        """
        ...

    async def apartial_fit(self, message: SampleMessage) -> None: ...


# --- Base implementation classes for processors ---


def _get_base_processor_settings_type(cls: type) -> type:
    try:
        return resolve_typevar(cls, SettingsType)
    except TypeError as e:
        raise TypeError(
            f"Could not resolve settings type for {cls}. "
            f"Ensure that the class is properly annotated with a SettingsType."
        ) from e


def _get_base_processor_message_in_type(cls: type) -> type:
    return resolve_typevar(cls, MessageInType)


def _get_base_processor_message_out_type(cls: type) -> type:
    return resolve_typevar(cls, MessageOutType)


def _unify_settings(
    obj: typing.Any, settings: object | None, *args, **kwargs
) -> typing.Any:
    """Helper function to unify settings for processor initialization."""
    settings_type = _get_base_processor_settings_type(obj.__class__)

    if settings is None:
        if len(args) > 0 and isinstance(args[0], settings_type):
            settings = args[0]
        elif len(args) > 0 or len(kwargs) > 0:
            settings = settings_type(*args, **kwargs)
        else:
            settings = settings_type()
    assert isinstance(settings, settings_type), "Settings must be of type " + str(
        settings_type
    )
    return settings


class BaseProcessor(ABC, typing.Generic[SettingsType, MessageInType, MessageOutType]):
    """
    Base class for processors. You probably do not want to inherit from this class directly.
    Refer instead to the more specific base classes.
      * Use :obj:`BaseConsumer` or :obj:`BaseTransformer` for ops that return a result or not, respectively.
      * Use :obj:`BaseStatefulProcessor` and its children for operations that require state.

    Note that `BaseProcessor` and its children are sync by default. If you need async by defualt, then
    override the async methods and call them from the sync methods. Look to `BaseProducer` for examples of
    calling async methods from sync methods.
    """

    settings: SettingsType

    @classmethod
    def get_settings_type(cls) -> type[SettingsType]:
        return _get_base_processor_settings_type(cls)

    @classmethod
    def get_message_type(cls, dir: str) -> typing.Any:
        if dir == "in":
            return _get_base_processor_message_in_type(cls)
        elif dir == "out":
            return _get_base_processor_message_out_type(cls)
        else:
            raise ValueError(f"Invalid direction: {dir}. Use 'in' or 'out'.")

    def __init__(self, *args, settings: SettingsType | None = None, **kwargs) -> None:
        self.settings = _unify_settings(self, settings, *args, **kwargs)

    @abstractmethod
    def _process(self, message: typing.Any) -> typing.Any: ...

    async def _aprocess(self, message: typing.Any) -> typing.Any:
        """Override this for native async processing."""
        return self._process(message)

    def __call__(self, message: typing.Any) -> typing.Any:
        # Note: We use the indirection to `_process` because this allows us to
        #  modify __call__ in derived classes with common functionality while
        #  minimizing the boilerplate code in derived classes as they only need to
        #  implement `_process`.
        return self._process(message)

    async def __acall__(self, message: typing.Any) -> typing.Any:
        """
        In Python 3.12+, we can invoke this method simply with `await obj(message)`,
        but earlier versions require direct syntax: `await obj.__acall__(message)`.
        """
        return await self._aprocess(message)

    def send(self, message: typing.Any) -> typing.Any:
        """Alias for __call__."""
        return self(message)

    async def asend(self, message: typing.Any) -> typing.Any:
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
    def get_settings_type(cls) -> type[SettingsType]:
        return _get_base_processor_settings_type(cls)

    @classmethod
    def get_message_type(cls, dir: str) -> type[MessageOutType] | None:
        if dir == "out":
            return _get_base_processor_message_out_type(cls)
        elif dir == "in":
            return None
        else:
            raise ValueError(f"Invalid direction: {dir}. Use 'in' or 'out'.")

    def __init__(self, *args, settings: SettingsType | None = None, **kwargs) -> None:
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

    async def __anext__(self) -> MessageOutType:
        # So this can be used as an async generator.
        return await self.__acall__()

    def __next__(self) -> MessageOutType:
        # So this can be used as a generator.
        return self()


class BaseConsumer(
    BaseProcessor[SettingsType, MessageInType, None],
    ABC,
    typing.Generic[SettingsType, MessageInType],
):
    """
    Base class for consumers -- processors that receive messages but don't produce output.
    This base simply overrides type annotations of BaseProcessor to remove the outputs.
    (We don't bother overriding `send` and `asend` because those are deprecated.)
    """

    @classmethod
    def get_message_type(cls, dir: str) -> type[MessageInType] | None:
        if dir == "in":
            return _get_base_processor_message_in_type(cls)
        elif dir == "out":
            return None
        else:
            raise ValueError(f"Invalid direction: {dir}. Use 'in' or 'out'.")

    @abstractmethod
    def _process(self, message: MessageInType) -> None: ...

    async def _aprocess(self, message: MessageInType) -> None:
        """Override this for native async processing."""
        return self._process(message)

    def __call__(self, message: MessageInType) -> None:
        return super().__call__(message)

    async def __acall__(self, message: MessageInType) -> None:
        return await super().__acall__(message)


class BaseTransformer(
    BaseProcessor[SettingsType, MessageInType, MessageOutType],
    ABC,
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


def _get_base_processor_state_type(cls: type) -> type:
    try:
        return resolve_typevar(cls, StateType)
    except TypeError as e:
        raise TypeError(
            f"Could not resolve state type for {cls}. "
            f"Ensure that the class is properly annotated with a StateType."
        ) from e


class Stateful(ABC, typing.Generic[StateType]):
    """
    Mixin class for stateful processors. DO NOT use this class directly.
    Used to enforce that the processor/producer has a state attribute and stateful_op method.
    """

    _state: StateType

    @classmethod
    def get_state_type(cls) -> type[StateType]:
        return _get_base_processor_state_type(cls)

    @property
    def state(self) -> StateType:
        return self._state

    @state.setter
    def state(self, state: StateType | bytes | None) -> None:
        if state is not None:
            if isinstance(state, bytes):
                self._state = pickle.loads(state)
            else:
                self._state = state  # type: ignore

    def _hash_message(self, message: typing.Any) -> int:
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
    def _reset_state(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """
        Reset internal state based on
            - new message metadata (processors), or
            - after first call (producers).
        """
        ...

    @abstractmethod
    def stateful_op(self, *args: typing.Any, **kwargs: typing.Any) -> tuple: ...


class BaseStatefulProcessor(
    BaseProcessor[SettingsType, MessageInType, MessageOutType],
    Stateful[StateType],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    """
    Base class implementing common stateful processor functionality.
    You probably do not want to inherit from this class directly.
    Refer instead to the more specific base classes.
    Use BaseStatefulConsumer for operations that do not return a result,
    or BaseStatefulTransformer for operations that do return a result.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hash = -1
        state_type = self.__class__.get_state_type()
        self._state: StateType = state_type()
        # TODO: Enforce that StateType has .hash: int field.

    @abstractmethod
    def _reset_state(self, message: typing.Any) -> None:
        """
        Reset internal state based on new message metadata.
        This method will only be called when there is a significant change in the message metadata,
        such as sample rate or shape (criteria defined by `_hash_message`), and not for every message,
        so use it to do all the expensive pre-allocation and caching of variables that can speed up
        the processing of subsequent messages in `_process`.
        """
        ...

    @abstractmethod
    def _process(self, message: typing.Any) -> typing.Any: ...

    def __call__(self, message: typing.Any) -> typing.Any:
        msg_hash = self._hash_message(message)
        if msg_hash != self._hash:
            self._reset_state(message)
            self._hash = msg_hash
        return self._process(message)

    async def __acall__(self, message: typing.Any) -> typing.Any:
        msg_hash = self._hash_message(message)
        if msg_hash != self._hash:
            self._reset_state(message)
            self._hash = msg_hash
        return await self._aprocess(message)

    def stateful_op(
        self,
        state: tuple[StateType, int] | None,
        message: typing.Any,
    ) -> tuple[tuple[StateType, int], typing.Any]:
        if state is not None:
            self.state, self._hash = state
        result = self(message)
        return (self.state, self._hash), result


class BaseStatefulProducer(
    BaseProducer[SettingsType, MessageOutType],
    Stateful[StateType],
    ABC,
    typing.Generic[SettingsType, MessageOutType, StateType],
):
    """
    Base class implementing common stateful producer functionality.
      Examples of stateful producers are things that require counters, clocks,
      or to cycle through a set of values.

    Unlike BaseStatefulProcessor, this class does not message hashing because there
      are no input messages. We still use self._hash to simply track the transition from
      initialization (.hash == -1) to state reset (.hash == 0).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # .settings
        self._hash = -1
        state_type = self.__class__.get_state_type()
        self._state: StateType = state_type()

    @abstractmethod
    def _reset_state(self) -> None:
        """
        Reset internal state upon first call.
        """
        ...

    async def __acall__(self) -> MessageOutType:
        if self._hash == -1:
            self._reset_state()
            self._hash = 0
        return await self._produce()

    def stateful_op(
        self,
        state: tuple[StateType, int] | None,
    ) -> tuple[tuple[StateType, int], MessageOutType]:
        if state is not None:
            self.state, self._hash = state  # Update state via setter
        result = self()  # Uses synchronous call
        return (self.state, self._hash), result


class BaseStatefulConsumer(
    BaseStatefulProcessor[SettingsType, MessageInType, None, StateType],
    ABC,
    typing.Generic[SettingsType, MessageInType, StateType],
):
    """
    Base class for stateful message consumers that don't produce output.
    This class merely overrides the type annotations of BaseStatefulProcessor.
    """

    @classmethod
    def get_message_type(cls, dir: str) -> type[MessageInType] | None:
        if dir == "in":
            return _get_base_processor_message_in_type(cls)
        elif dir == "out":
            return None
        else:
            raise ValueError(f"Invalid direction: {dir}. Use 'in' or 'out'.")

    @abstractmethod
    def _process(self, message: MessageInType) -> None: ...

    async def _aprocess(self, message: MessageInType) -> None:
        return self._process(message)

    def __call__(self, message: MessageInType) -> None:
        return super().__call__(message)

    async def __acall__(self, message: MessageInType) -> None:
        return await super().__acall__(message)

    def stateful_op(
        self,
        state: tuple[StateType, int] | None,
        message: MessageInType,
    ) -> tuple[tuple[StateType, int], None]:
        state, _ = super().stateful_op(state, message)
        return state, None


class BaseStatefulTransformer(
    BaseStatefulProcessor[SettingsType, MessageInType, MessageOutType, StateType],
    ABC,
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
        state: tuple[StateType, int] | None,
        message: MessageInType,
    ) -> tuple[tuple[StateType, int], MessageOutType]:
        return super().stateful_op(state, message)


class BaseAdaptiveTransformer(
    BaseStatefulTransformer[
        SettingsType,
        MessageInType | SampleMessage,
        MessageOutType | None,
        StateType,
    ],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, StateType],
):
    @abstractmethod
    def partial_fit(self, message: SampleMessage) -> None: ...

    async def apartial_fit(self, message: SampleMessage) -> None:
        """Override me if you need async partial fitting."""
        return self.partial_fit(message)

    def __call__(self, message: MessageInType | SampleMessage) -> MessageOutType | None:
        """
        Adapt transformer with training data (and optionally labels)
        in SampleMessage

        Args:
            message: An instance of SampleMessage with optional
             labels (y) in message.trigger.value.data and
             data (X) in message.sample.data

        Returns: None
        """
        if is_sample_message(message):
            return self.partial_fit(message)
        return super().__call__(message)

    async def __acall__(
        self, message: MessageInType | SampleMessage
    ) -> MessageOutType | None:
        if is_sample_message(message):
            return await self.apartial_fit(message)
        return await super().__acall__(message)


class BaseAsyncTransformer(
    BaseStatefulTransformer[SettingsType, MessageInType, MessageOutType, StateType],
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
        msg_hash = self._hash_message(message)
        if msg_hash != self._hash:
            self._reset_state(message)
            self._hash = msg_hash
        return await self._aprocess(message)


# Composite processor for building pipelines
def _get_processor_message_type(
    proc: BaseProcessor | BaseProducer | GeneratorType | SyncToAsyncGeneratorWrapper,
    dir: str,
) -> type | None:
    """Extract the input type from a processor."""
    if isinstance(proc, GeneratorType) or isinstance(proc, SyncToAsyncGeneratorWrapper):
        gen_func = proc.gi_frame.f_globals[proc.gi_frame.f_code.co_name]
        args = typing.get_args(gen_func.__annotations__.get("return"))
        return args[0] if dir == "out" else args[1]  # yield type / send type
    return proc.__class__.get_message_type(dir)


def _has_stateful_op(proc: typing.Any) -> typing.TypeGuard[Stateful]:
    """
    Check if the processor has a stateful_op method.
    This is used to determine if the processor is stateful or not.
    """
    return hasattr(proc, "stateful_op")


class CompositeStateful(
    Stateful[dict[str, typing.Any]], ABC, typing.Generic[SettingsType, MessageOutType]
):
    """
    Mixin class for composite processor/producer chains. DO NOT use this class directly.
    Used to enforce statefulness of the composite processor/producer chain and provide
    initialization and validation methods.
    """

    _procs: dict[
        str, BaseProducer | BaseProcessor | GeneratorType | SyncToAsyncGeneratorWrapper
    ]
    _processor_type: typing.Literal["producer", "processor"]

    def _validate_processor_chain(self) -> None:
        """Validate the composite chain types at runtime."""
        if not self._procs:
            raise ValueError(
                f"Composite {self._processor_type} requires at least one processor"
            )

        expected_in_type = _get_processor_message_type(self, "in")
        expected_out_type = _get_processor_message_type(self, "out")

        procs = [p for p in self._procs.items() if p[1] is not None]
        in_type = _get_processor_message_type(procs[0][1], "in")
        if not check_message_type_compatibility(expected_in_type, in_type):
            raise TypeError(
                f"Input type mismatch: Composite {self._processor_type} expects {expected_in_type}, "
                f"but its first processor (name: {procs[0][0]}, type: {procs[0][1].__class__.__name__}) accepts {in_type}"
            )

        out_type = _get_processor_message_type(procs[-1][1], "out")
        if not check_message_type_compatibility(out_type, expected_out_type):
            raise TypeError(
                f"Output type mismatch: Composite {self._processor_type} wants to return {expected_out_type}, "
                f"but its last processor (name: {procs[-1][0]}, type: {procs[-1][1].__class__.__name__})  returns {out_type}"
            )

        # Check intermediate connections
        for i in range(len(procs) - 1):
            current_out_type = _get_processor_message_type(procs[i][1], "out")
            next_in_type = _get_processor_message_type(procs[i + 1][1], "in")

            if current_out_type is None or current_out_type is type(None):
                raise TypeError(
                    f"Processor {i} (name: {procs[i][0]}, type: {procs[i][1].__class__.__name__}) is a consumer "
                    f"or returns None. Consumers can only be the last processor of a composite {self._processor_type} chain."
                )
            if next_in_type is None or next_in_type is type(None):
                raise TypeError(
                    f"Processor {i + 1} (name: {procs[i + 1][0]}, type: {procs[i + 1][1].__class__.__name__}) is a producer "
                    f"or receives only None. Producers can only be the first processor of a composite producer chain."
                )
            if not check_message_type_compatibility(current_out_type, next_in_type):
                raise TypeError(
                    f"Message type mismatch between processors {i} (name: {procs[i][0]}, type: {procs[i][1].__class__.__name__}) "
                    f"and {i + 1} (name: {procs[i + 1][0]}, type: {procs[i + 1][1].__class__.__name__}): "
                    f"{procs[i][1].__class__.__name__} outputs {current_out_type}, "
                    f"but {procs[i + 1][1].__class__.__name__} expects {next_in_type}"
                )
            if inspect.isgenerator(procs[i][1]) and hasattr(procs[i][1], "send"):
                # If the processor is a generator, wrap it in a SyncToAsyncGeneratorWrapper
                procs[i] = (procs[i][0], SyncToAsyncGeneratorWrapper(procs[i][1]))
        if inspect.isgenerator(procs[-1][1]) and hasattr(procs[-1][1], "send"):
            # If the last processor is a generator, wrap it in a SyncToAsyncGeneratorWrapper
            procs[-1] = (procs[-1][0], SyncToAsyncGeneratorWrapper(procs[-1][1]))
        self._procs = {k: v for (k, v) in procs}

    @staticmethod
    @abstractmethod
    def _initialize_processors(
        settings: SettingsType,
    ) -> dict[str, typing.Any]: ...

    @property
    def state(self) -> dict[str, typing.Any]:
        return {
            k: getattr(proc, "state")
            for k, proc in self._procs.items()
            if hasattr(proc, "state")
        }

    @state.setter
    def state(self, state: dict[str, typing.Any] | bytes | None) -> None:
        if state is not None:
            if isinstance(state, bytes):
                state = pickle.loads(state)
            for k, v in state.items():  # type: ignore
                if k not in self._procs:
                    raise KeyError(
                        f"Processor (name: {k}) in provided state not found in composite {self._processor_type} chain. "
                        f"Available keys: {list(self._procs.keys())}"
                    )
                if hasattr(self._procs[k], "state"):
                    setattr(self._procs[k], "state", v)

    def _reset_state(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # By default, we don't expect to change the state of a composite processor/producer
        pass

    @abstractmethod
    def stateful_op(
        self,
        state: dict[str, tuple[typing.Any, int]] | None,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> tuple[
        dict[str, tuple[typing.Any, int]],
        MessageOutType | None,
    ]: ...


class CompositeProcessor(
    BaseProcessor[SettingsType, MessageInType, MessageOutType],
    CompositeStateful[SettingsType, MessageOutType],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType],
):
    """
    A processor that chains multiple processor together in a feedforward non-branching graph.
    The individual processors may be stateless or stateful. The last processor may be a consumer,
    otherwise processors must be transformers. Use CompositeProducer if you want the first
    processor to be a producer. Concrete subclasses must implement `_initialize_processors`.
    Optionally override `_reset_state` if you want adaptive state behaviour.
    Example implementation:

    class CustomCompositeProcessor(CompositeProcessor[CustomSettings, AxisArray, AxisArray]):
        @staticmethod
        def _initialize_processors(settings: CustomSettings) -> dict[str, BaseProcessor]:
            return {
                "stateful_transformer": CustomStatefulProducer(**settings),
                "transformer": CustomTransformer(**settings),
            }
    Where **settings should be replaced with initialisation arguments for each processor.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # .settings
        self._processor_type = "processor"
        self._procs = self._initialize_processors(self.settings)
        self._validate_processor_chain()
        first_proc = next(iter(self._procs.items()))
        first_proc_in_type = _get_processor_message_type(first_proc[1], "in")
        if first_proc_in_type is None or first_proc_in_type is type(None):
            raise TypeError(
                f"First processor (name: {first_proc[0]}, type: {first_proc[1].__class__.__name__}) "
                f"is a producer or receives only None. Please use CompositeProducer, not "
                f"CompositeProcessor for this composite chain."
            )
        self._hash = -1

    @staticmethod
    @abstractmethod
    def _initialize_processors(settings: SettingsType) -> dict[str, typing.Any]: ...

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
        result = message
        for proc in self._procs.values():
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
        result = message
        for proc in self._procs.values():
            result = await proc.asend(result)
        return result

    def stateful_op(
        self,
        state: dict[str, tuple[typing.Any, int]] | None,
        message: MessageInType | None,
    ) -> tuple[
        dict[str, tuple[typing.Any, int]],
        MessageOutType | None,
    ]:
        result = message
        state = state or {}
        try:
            state_keys = list(state.keys())
        except AttributeError as e:
            raise AttributeError(
                "state provided to stateful_op must be a dict or None"
            ) from e
        for key in state_keys:
            if key not in self._procs:
                raise KeyError(
                    f"Processor (name: {key}) in provided state not found in composite processor chain. "
                    f"Available keys: {list(self._procs.keys())}"
                )
        for k, proc in self._procs.items():
            if _has_stateful_op(proc):
                state[k], result = proc.stateful_op(state.get(k, None), result)
            else:
                result = proc.send(result)
        return state, result


class CompositeProducer(
    BaseProducer[SettingsType, MessageOutType],
    CompositeStateful[SettingsType, MessageOutType],
    ABC,
    typing.Generic[SettingsType, MessageOutType],
):
    """
    A producer that chains multiple processors (starting with a producer) together in a feedforward
    non-branching graph. The individual processors may be stateless or stateful.
    The first processor must be a producer, the last processor may be a consumer, otherwise
    processors must be transformers.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # .settings
        self._processor_type = "producer"
        self._procs = self._initialize_processors(self.settings)
        self._validate_processor_chain()
        first_proc = next(iter(self._procs.items()))
        first_proc_in_type = _get_processor_message_type(first_proc[1], "in")
        if first_proc_in_type is not None and first_proc_in_type is not type(None):
            raise TypeError(
                f"First processor (name: {first_proc[0]}, type: {first_proc[1].__class__.__name__}) "
                f"is not a producer. Please use CompositeProcessor, not "
                f"CompositeProducer for this composite chain."
            )
        self._hash = -1

    @staticmethod
    @abstractmethod
    def _initialize_processors(
        settings: SettingsType,
    ) -> dict[str, typing.Any]: ...

    async def _produce(self) -> MessageOutType:
        """
        Process a message through the pipeline of processors. If the message is None, or no message is provided,
        then it will be assumed that the first processor is a producer and will be called without arguments.
        This will be invoked via `__call__` or `send`.
        We use `__next__` and `send` to allow using legacy generators that have yet to be converted to transformers.

        Warning: All processors will be called using their asynchronous API, which is much faster for async
        processors and does not incur penalty on sync processors.
        """
        procs = list(self._procs.values())
        result = await procs[0].__anext__()
        for proc in procs[1:]:
            result = await proc.asend(result)
        return result

    def stateful_op(
        self,
        state: dict[str, tuple[typing.Any, int]] | None,
    ) -> tuple[
        dict[str, tuple[typing.Any, int]],
        MessageOutType | None,
    ]:
        state = state or {}
        try:
            state_keys = list(state.keys())
        except AttributeError as e:
            raise AttributeError(
                "state provided to stateful_op must be a dict or None"
            ) from e
        for key in state_keys:
            if key not in self._procs:
                raise KeyError(
                    f"Processor (name: {key}) in provided state not found in composite producer chain. "
                    f"Available keys: {list(self._procs.keys())}"
                )
        labeled_procs = list(self._procs.items())
        prod_name, prod = labeled_procs[0]
        if _has_stateful_op(prod):
            state[prod_name], result = prod.stateful_op(state.get(prod_name, None))
        else:
            result = prod.__next__()
        for k, proc in labeled_procs[1:]:
            if _has_stateful_op(proc):
                state[k], result = proc.stateful_op(state.get(k, None), result)
            else:
                result = proc.send(result)
        return state, result


# --- Type variables for protocols and processors ---
ProducerType = typing.TypeVar("ProducerType", bound=BaseProducer)
ConsumerType = typing.TypeVar("ConsumerType", bound=BaseConsumer | BaseStatefulConsumer)
TransformerType = typing.TypeVar(
    "TransformerType",
    bound=BaseTransformer | BaseStatefulTransformer | CompositeProcessor,
)
AdaptiveTransformerType = typing.TypeVar(
    "AdaptiveTransformerType", bound=BaseAdaptiveTransformer
)


def get_base_producer_type(cls: type) -> type:
    return resolve_typevar(cls, ProducerType)


def get_base_consumer_type(cls: type) -> type:
    return resolve_typevar(cls, ConsumerType)


def get_base_transformer_type(cls: type) -> type:
    return resolve_typevar(cls, TransformerType)


def get_base_adaptive_transformer_type(cls: type) -> type:
    return resolve_typevar(cls, AdaptiveTransformerType)


# --- Base classes for ezmsg Unit with specific processing capabilities ---
class BaseProducerUnit(
    ez.Unit, ABC, typing.Generic[SettingsType, MessageOutType, ProducerType]
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

    Where CustomProducerSettings, and CustomProducer are custom implementations of ez.Settings,
    and BaseProducer or BaseStatefulProducer, respectively.
    """

    INPUT_SETTINGS = ez.InputStream(SettingsType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageOutType)

    async def initialize(self) -> None:
        self.create_producer()

    def create_producer(self) -> None:
        # self.producer: ProducerType
        """Create the producer instance from settings."""
        producer_type = get_base_producer_type(self.__class__)
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
        self.apply_settings(msg)  # type: ignore
        self.create_producer()

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        while True:
            out = await self.producer.__acall__()
            if out is not None:  # and math.prod(out.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, out


class BaseProcessorUnit(ez.Unit, ABC, typing.Generic[SettingsType]):
    """
    Base class for processor units -- i.e. units that process messages.
    This is an abstract base class that provides common functionality for consumer and transformer
    units. You probably do not want to inherit from this class directly as you would need to define
    a custom implementation of `create_processor`.
    Refer instead to BaseConsumerUnit or BaseTransformerUnit.
    """

    INPUT_SETTINGS = ez.InputStream(SettingsType)

    async def initialize(self) -> None:
        self.create_processor()

    @abstractmethod
    def create_processor(self) -> None: ...

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: SettingsType) -> None:
        """
        Receive a settings message, override self.SETTINGS, and re-create the processor.
        Child classes that wish to have fine-grained control over whether the
        core processor resets on settings changes should override this method.

        Args:
            msg: a settings message.
        """
        self.apply_settings(msg)  # type: ignore
        self.create_processor()


class BaseConsumerUnit(
    BaseProcessorUnit[SettingsType],
    ABC,
    typing.Generic[SettingsType, MessageInType, ConsumerType],
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

    Where CustomConsumerSettings and CustomConsumer are custom implementations of:
    - ez.Settings for settings
    - BaseConsumer or BaseStatefulConsumer for the consumer implementation
    """

    INPUT_SIGNAL = ez.InputStream(MessageInType)

    def create_processor(self):
        # self.processor: ConsumerType[SettingsType, MessageInType, StateType]
        """Create the consumer instance from settings."""
        consumer_type = get_base_consumer_type(self.__class__)
        self.processor = consumer_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    async def on_signal(self, message: MessageInType):
        """
        Consume the message.
        Args:
            message: Input message to be consumed
        """
        await self.processor.__acall__(message)


class BaseTransformerUnit(
    BaseProcessorUnit[SettingsType],
    ABC,
    typing.Generic[SettingsType, MessageInType, MessageOutType, TransformerType],
):
    """
    Base class for transformer units -- i.e. units that transform input messages into output messages.
    Implement a new Unit as follows:

    class CustomUnit(BaseTransformerUnit[
        CustomTransformerSettings,    # SettingsType
        AxisArray,                    # MessageInType
        AxisArray,                    # MessageOutType
        CustomTransformer,            # TransformerType
    ]):
        SETTINGS = CustomTransformerSettings

    ... that's all!

    Where CustomTransformerSettings and CustomTransformer are custom implementations of:
    - ez.Settings for settings
    - One of these transformer types:
      * BaseTransformer
      * BaseStatefulTransformer
      * CompositeProcessor
    """

    INPUT_SIGNAL = ez.InputStream(MessageInType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageOutType)

    def create_processor(self):
        # self.processor: TransformerType[SettingsType, MessageInType, MessageOutType, StateType]
        """Create the transformer instance from settings."""
        transformer_type = get_base_transformer_type(self.__class__)
        self.processor = transformer_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: MessageInType) -> typing.AsyncGenerator:
        result = await self.processor.__acall__(message)
        if result is not None:  # and math.prod(result.data.shape) > 0:
            yield self.OUTPUT_SIGNAL, result


class BaseAdaptiveTransformerUnit(
    BaseProcessorUnit[SettingsType],
    ABC,
    typing.Generic[
        SettingsType, MessageInType, MessageOutType, AdaptiveTransformerType
    ],
):
    INPUT_SAMPLE = ez.InputStream(SampleMessage)
    INPUT_SIGNAL = ez.InputStream(MessageInType)
    OUTPUT_SIGNAL = ez.OutputStream(MessageOutType)

    def create_processor(self) -> None:
        # self.processor: AdaptiveTransformerType[SettingsType, MessageInType, MessageOutType, StateType]
        """Create the adaptive transformer instance from settings."""
        adaptive_transformer_type = get_base_adaptive_transformer_type(self.__class__)
        self.processor = adaptive_transformer_type(settings=self.SETTINGS)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: MessageInType) -> typing.AsyncGenerator:
        result = await self.processor.__acall__(message)
        if result is not None:  # and math.prod(result.data.shape) > 0:
            yield self.OUTPUT_SIGNAL, result

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
