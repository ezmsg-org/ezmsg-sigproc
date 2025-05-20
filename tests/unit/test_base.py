import dataclasses
import pickle
import pytest
from types import NoneType
from typing import Any, Generator
from unittest.mock import MagicMock

from ezmsg.sigproc.base import (
    CompositeProducer,
    _get_base_processor_message_in_type,
    _get_base_processor_message_out_type,
    _get_base_processor_settings_type,
    _get_base_processor_state_type,
    _get_processor_message_type,
    processor_state,
    BaseProcessor,
    BaseProducer,
    BaseConsumer,
    BaseTransformer,
    BaseStatefulProcessor,
    BaseStatefulProducer,
    BaseStatefulConsumer,
    BaseStatefulTransformer,
    BaseAdaptiveTransformer,
    BaseAsyncTransformer,
    CompositeProcessor,
    SampleMessage,
)
from ezmsg.sigproc.cheby import ChebyshevFilterTransformer
from ezmsg.sigproc.filter import FilterByDesignState
from ezmsg.util.generator import consumer

# -- Mock Classes for Testing --


@dataclasses.dataclass
class MockSettings:
    param1: int = 10
    param2: str = "test"


@processor_state
class MockState:
    iterations: int = 0
    hash: int = -1


# Simple mock message types for testing type compatibility
class MockMessageA:
    pass


class MockMessageB(MockMessageA):
    pass


class MockMessageC:
    pass


# Mock processors for testing
class MockProcessor(BaseProcessor[MockSettings, MockMessageA, MockMessageB]):
    def _process(self, message: MockMessageA) -> MockMessageB:
        return MockMessageB()


class DeepMockProcessor(MockProcessor):
    def _process(self, message: MockMessageA) -> MockMessageB:
        return MockMessageB()


class MockProducer(BaseProducer[MockSettings, MockMessageA]):
    async def _produce(self) -> MockMessageA:
        return MockMessageA()


class MockConsumer(BaseConsumer[MockSettings, MockMessageB]):
    def _process(self, message: MockMessageB) -> None:
        pass


class MockTransformer(BaseTransformer[MockSettings, MockMessageA, MockMessageC]):
    def _process(self, message: MockMessageA) -> MockMessageC:
        return MockMessageC()


class DeepMockTransformer(MockTransformer):
    pass


class DeeperMockTransformer(DeepMockTransformer):
    pass


class MockStatefulProcessor(
    BaseStatefulProcessor[MockSettings, MockMessageA, MockMessageB, MockState]
):
    def _reset_state(self, message: MockMessageA) -> None:
        self._state.iterations = 0

    def _process(self, message: MockMessageA) -> MockMessageB:
        self._state.iterations += 1
        return MockMessageB()


class DeepMockStatefulProcessor(MockStatefulProcessor):
    pass


class MockStatefulProducer(BaseStatefulProducer[MockSettings, MockMessageA, MockState]):
    def _reset_state(self) -> None:
        self._state.iterations = 0

    async def _produce(self) -> MockMessageA:
        self._state.iterations += 1
        return MockMessageA()


class MockStatefulConsumer(BaseStatefulConsumer[MockSettings, MockMessageA, MockState]):
    def _reset_state(self, message: MockMessageA) -> None:
        self._state.iterations = 0

    def _process(self, message: MockMessageA) -> None:
        self._state.iterations += 1


class MockStatefulTransformer(
    BaseStatefulTransformer[MockSettings, MockMessageA, MockMessageC, MockState]
):
    def _reset_state(self, message: MockMessageA) -> None:
        self._state.iterations = 0

    def _process(self, message: MockMessageA) -> MockMessageC:
        self._state.iterations += 1
        return MockMessageC()


class MockAdaptiveTransformer(
    BaseAdaptiveTransformer[MockSettings, MockMessageA, MockMessageB, MockState]
):
    def _reset_state(self, message: MockMessageA) -> None:
        self._state.iterations = 0

    def _process(self, message: MockMessageA | SampleMessage) -> MockMessageB:
        self._state.iterations += 1
        return MockMessageB()

    def partial_fit(self, message: SampleMessage) -> None:
        self._state.iterations += 1


class MockAsyncTransformer(
    BaseAsyncTransformer[MockSettings, MockMessageA, MockMessageB, MockState]
):
    def _reset_state(self, message: MockMessageA) -> None:
        self._state.iterations = 0

    async def _aprocess(self, message: MockMessageA) -> MockMessageB:
        self._state.iterations += 1
        return MockMessageB()


# Mock CompositeProcessor examples
class ValidSingleCompositeProcessor(
    CompositeProcessor[MockSettings, MockMessageA, MockMessageB]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {"processor": MockProcessor(settings=settings)}


class ValidMultipleCompositeProcessor(
    CompositeProcessor[MockSettings, MockMessageA, MockMessageB]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "stateful_processor": MockStatefulProcessor(settings=settings),
        }


class EmptyCompositeProcessor(
    CompositeProcessor[MockSettings, MockMessageA, MockMessageB]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {}


class InvalidOutputCompositeProcessor(
    CompositeProcessor[MockSettings, MockMessageA, MockMessageB]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {"transformer": MockTransformer(settings=settings)}


class InvalidInputCompositeProcessor(
    CompositeProcessor[MockSettings, None, MockMessageC]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {"transformer": MockTransformer(settings=settings)}


class InvalidProducerNotFirstCompositeProcessor(
    CompositeProcessor[MockSettings, MockMessageA, MockMessageA]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "producer": MockProducer(settings=settings),
        }


class InvalidConsumerNotLastCompositeProcessor(
    CompositeProcessor[MockSettings, MockMessageB, MockMessageC]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "consumer": MockConsumer(settings=settings),
            "transformer": MockTransformer(settings=settings),
        }


class InvalidConsumerProducerCompositeProcessor(
    CompositeProcessor[MockSettings, MockMessageB, MockMessageA]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "consumer": MockConsumer(settings=settings),
            "producer": MockProducer(settings=settings),
        }


class TypeMismatchCompositeProcessor(
    CompositeProcessor[MockSettings, MockMessageA, None]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "transformer": MockTransformer(settings=settings),
            "consumer": MockConsumer(settings=settings),
        }


class InvalidProducerFirstCompositeProcessor(
    CompositeProcessor[MockSettings, NoneType, MockMessageC]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "stateful_producer": MockStatefulProducer(settings=settings),
            "stateful_transformer": MockStatefulTransformer(settings=settings),
        }


class ChainedCompositeProcessor(CompositeProcessor[MockSettings, MockMessageA, None]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "stateful_consumer": MockStatefulConsumer(settings=settings),
        }


class ChainedCompositeProcessorWithDeepProcessors(
    CompositeProcessor[MockSettings, MockMessageA, MockMessageC]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "deep_processor": DeepMockProcessor(settings=settings),
            "deeper_transformer": DeeperMockTransformer(settings=settings),
        }


@consumer
def mock_generator() -> Generator[MockMessageA, MockMessageA, None]:
    """A mock generator function for testing purposes."""
    while True:
        yield MockMessageA()


class MockGeneratorCompositeProcessor(
    CompositeProcessor[MockSettings, MockMessageA, MockMessageB]
):
    @staticmethod
    def _initialize_processors(settings):
        return {
            "generator": mock_generator(),
            "stateful_processor": MockStatefulProcessor(settings=settings),
        }


# Mock CompositeProducer examples
class ValidSingleCompositeProducer(CompositeProducer[MockSettings, MockMessageA]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {"producer": MockProducer(settings=settings)}


class ValidMultipleCompositeProducer(CompositeProducer[MockSettings, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "producer": MockProducer(settings=settings),
            "stateful_processor": MockStatefulProcessor(settings=settings),
        }


class EmptyCompositeProducer(CompositeProducer[MockSettings, MockMessageA]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {}


class InvalidOutputCompositeProducer(CompositeProducer[MockSettings, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {"producer": MockProducer(settings=settings)}


class InvalidProducerNotFirstCompositeProducer(
    CompositeProducer[MockSettings, MockMessageA]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "processor": MockProcessor(settings=settings),
            "producer": MockProducer(settings=settings),
        }


class InvalidConsumerNotLastCompositeProducer(
    CompositeProducer[MockSettings, MockMessageC]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "producer": MockProducer(settings=settings),
            "stateful_consumer": MockStatefulConsumer(settings=settings),
            "transformer": MockTransformer(settings=settings),
        }


class TypeMismatchCompositeProducer(CompositeProducer[MockSettings, None]):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "producer": MockProducer(settings=settings),
            "consumer": MockConsumer(settings=settings),
        }


class ChainedCompositeProducerWithDeepProcessors(
    CompositeProducer[MockSettings, MockMessageC]
):
    @staticmethod
    def _initialize_processors(settings: MockSettings) -> dict[str, Any]:
        return {
            "stateful_producer": MockStatefulProducer(settings=settings),
            "deep_processor": DeepMockProcessor(settings=settings),
            "deeper_transformer": DeeperMockTransformer(settings=settings),
        }


@consumer
def mock_producer_generator() -> Generator[MockMessageA, None, None]:
    """A mock generator function for testing purposes."""
    while True:
        yield MockMessageA()


def mock_generator_unprimed() -> Generator[MockMessageA, MockMessageA, None]:
    """A mock generator function for testing purposes."""
    output = MockMessageA()
    while True:
        input = yield output
        output = input or output


class MockGeneratorCompositeProducer(CompositeProducer[MockSettings, MockMessageB]):
    @staticmethod
    def _initialize_processors(settings):
        return {
            "generator": mock_producer_generator(),
            "mock_generator_unprimed": mock_generator_unprimed(),
            "stateful_processor": MockStatefulProcessor(settings=settings),
        }


# -- Tests for Helper Functions --


class TestHelperFunctions:
    def test_processor_state_decorator(self):
        # Test that processor_state creates a dataclass with the right properties
        state = MockState()
        assert state.iterations == 0
        assert state.hash == -1

        # Test that we can modify it (not frozen)
        state.iterations = 5
        assert state.iterations == 5

        # Test hash functionality
        state1 = MockState()
        state2 = MockState()
        assert hash(state1) == hash(state2)

        state1.iterations = 10
        assert hash(state1) != hash(state2)

    def test_get_base_processor_settings_type(self):
        # Test with regular processor classes
        assert _get_base_processor_settings_type(MockProcessor) == MockSettings
        assert _get_base_processor_settings_type(MockProducer) == MockSettings
        assert _get_base_processor_settings_type(MockConsumer) == MockSettings
        assert _get_base_processor_settings_type(MockTransformer) == MockSettings

        # Test with stateful processor classes
        assert _get_base_processor_settings_type(MockStatefulProcessor) == MockSettings
        assert _get_base_processor_settings_type(MockStatefulProducer) == MockSettings
        assert _get_base_processor_settings_type(MockStatefulConsumer) == MockSettings
        assert (
            _get_base_processor_settings_type(MockStatefulTransformer) == MockSettings
        )

        # Test with derived classes
        assert _get_base_processor_settings_type(DeepMockProcessor) == MockSettings
        assert _get_base_processor_settings_type(DeepMockTransformer) == MockSettings
        assert _get_base_processor_settings_type(DeeperMockTransformer) == MockSettings

        # Test with composite processor
        assert (
            _get_base_processor_settings_type(ValidSingleCompositeProcessor)
            == MockSettings
        )

        # Test with no settings type should raise exception
        class NoSettingsTypeClass:
            pass

        with pytest.raises(TypeError, match="Could not resolve settings type"):
            _get_base_processor_settings_type(NoSettingsTypeClass)

    def test_get_base_processor_message_in_type(self):
        # Test with processor classes that have input types
        assert _get_base_processor_message_in_type(MockProcessor) == MockMessageA
        assert _get_base_processor_message_in_type(MockConsumer) == MockMessageB
        assert _get_base_processor_message_in_type(MockTransformer) == MockMessageA
        assert (
            _get_base_processor_message_in_type(MockStatefulProcessor) == MockMessageA
        )
        assert _get_base_processor_message_in_type(MockStatefulConsumer) == MockMessageA
        assert (
            _get_base_processor_message_in_type(MockStatefulTransformer) == MockMessageA
        )
        assert (
            _get_base_processor_message_in_type(MockAdaptiveTransformer) == MockMessageA
        )
        assert _get_base_processor_message_in_type(MockAsyncTransformer) == MockMessageA

        # Test with derived classes
        assert _get_base_processor_message_in_type(DeepMockProcessor) == MockMessageA
        assert _get_base_processor_message_in_type(DeepMockTransformer) == MockMessageA
        assert (
            _get_base_processor_message_in_type(DeeperMockTransformer) == MockMessageA
        )

        # Test with composite processor
        assert (
            _get_base_processor_message_in_type(ValidSingleCompositeProcessor)
            == MockMessageA
        )
        assert (
            _get_base_processor_message_in_type(ChainedCompositeProcessor)
            == MockMessageA
        )

        # Test with producers (should throw)
        with pytest.raises(TypeError, match=r"Could not resolve .*MessageInType"):
            _get_base_processor_message_in_type(MockProducer)
        with pytest.raises(TypeError, match=r"Could not resolve .*MessageInType"):
            _get_base_processor_message_in_type(MockStatefulProducer)

        # Test with no message in type should raise exception
        class NoMessageInTypeClass:
            pass

        with pytest.raises(TypeError, match=r"Could not resolve .*MessageInType"):
            _get_base_processor_message_in_type(NoMessageInTypeClass)

    def test_get_base_processor_message_out_type(self):
        # Test with processor classes that have output types
        assert _get_base_processor_message_out_type(MockProcessor) == MockMessageB
        assert _get_base_processor_message_out_type(MockProducer) == MockMessageA
        assert _get_base_processor_message_out_type(MockTransformer) == MockMessageC
        assert (
            _get_base_processor_message_out_type(MockStatefulProcessor) == MockMessageB
        )
        assert (
            _get_base_processor_message_out_type(MockStatefulProducer) == MockMessageA
        )
        assert (
            _get_base_processor_message_out_type(MockStatefulTransformer)
            == MockMessageC
        )
        assert (
            _get_base_processor_message_out_type(MockAdaptiveTransformer)
            == MockMessageB
        )
        assert (
            _get_base_processor_message_out_type(MockAsyncTransformer) == MockMessageB
        )

        # Test with derived classes
        assert _get_base_processor_message_out_type(DeepMockProcessor) == MockMessageB
        assert _get_base_processor_message_out_type(DeepMockTransformer) == MockMessageC
        assert (
            _get_base_processor_message_out_type(DeeperMockTransformer) == MockMessageC
        )

        # Test with composite processor
        assert (
            _get_base_processor_message_out_type(ValidSingleCompositeProcessor)
            == MockMessageB
        )

        # Test with consumers (should be None)
        assert _get_base_processor_message_out_type(MockConsumer) is NoneType
        assert _get_base_processor_message_out_type(MockStatefulConsumer) is NoneType

        # Test with no message out type should raise exception
        class NoMessageOutTypeClass:
            pass

        with pytest.raises(TypeError, match="Could not resolve .*MessageOutType"):
            _get_base_processor_message_out_type(NoMessageOutTypeClass)

    # Test _unify_settings function through the MockProcessor class __init__
    def test_unify_settings_with_provided_settings(self):
        settings = MockSettings(param1=20)
        obj = MockProcessor(settings=settings)
        assert obj.settings.param1 == 20
        assert obj.settings.param2 == "test"

    def test_unify_settings_with_settings_arg(self):
        obj = MockProcessor(MockSettings(param1=30, param2="new"))
        assert obj.settings.param1 == 30
        assert obj.settings.param2 == "new"

    def test_unify_settings_with_args(self):
        obj = MockProcessor(40, "newer")
        assert obj.settings.param1 == 40
        assert obj.settings.param2 == "newer"

    def test_unify_settings_with_kwargs(self):
        obj = MockProcessor(param1=50)
        assert obj.settings.param1 == 50
        assert obj.settings.param2 == "test"

    def test_unify_settings_with_args_kwargs(self):
        obj = MockProcessor(60, param2="newest")
        assert obj.settings.param1 == 60
        assert obj.settings.param2 == "newest"

    def test_unify_settings_with_default(self):
        obj = MockProcessor()
        assert obj.settings.param1 == 10
        assert obj.settings.param2 == "test"

    def test_get_base_processor_state_type(self):
        # Test with class that has state type
        assert _get_base_processor_state_type(MockStatefulProcessor) == MockState
        assert _get_base_processor_state_type(DeepMockStatefulProcessor) == MockState
        assert _get_base_processor_state_type(MockStatefulProducer) == MockState
        assert _get_base_processor_state_type(MockStatefulConsumer) == MockState
        assert _get_base_processor_state_type(MockStatefulTransformer) == MockState
        assert _get_base_processor_state_type(MockAdaptiveTransformer) == MockState
        assert _get_base_processor_state_type(MockAsyncTransformer) == MockState
        assert (
            _get_base_processor_state_type(ChebyshevFilterTransformer)
            == FilterByDesignState
        )
        assert (
            _get_base_processor_state_type(ValidMultipleCompositeProcessor)
            == dict[str, Any]
        )

        # Test with class that doesn't have state type
        with pytest.raises(Exception, match="Could not resolve state type"):
            _get_base_processor_state_type(MockProcessor)
        with pytest.raises(Exception, match="Could not resolve state type"):
            _get_base_processor_state_type(DeepMockProcessor)
        with pytest.raises(Exception, match="Could not resolve state type"):
            _get_base_processor_state_type(MockProducer)
        with pytest.raises(Exception, match="Could not resolve state type"):
            _get_base_processor_state_type(MockConsumer)
        with pytest.raises(Exception, match="Could not resolve state type"):
            _get_base_processor_state_type(MockTransformer)

    def test_get_processor_message_type(self):
        processor = MockProcessor(MockSettings())
        producer = MockProducer(MockSettings())
        consumer = MockConsumer(MockSettings())
        deep_transformer = DeeperMockTransformer(MockSettings())

        # Test getting input type
        assert _get_processor_message_type(processor, "in") == MockMessageA

        # Test getting output type
        assert _get_processor_message_type(processor, "out") == MockMessageB

        # Test with producer
        assert _get_processor_message_type(producer, "in") is None
        assert _get_processor_message_type(producer, "out") == MockMessageA

        # Test with consumer
        assert _get_processor_message_type(consumer, "in") == MockMessageB
        assert _get_processor_message_type(consumer, "out") is None

        # Test with deep subclass (transformer)
        assert _get_processor_message_type(deep_transformer, "in") == MockMessageA
        assert _get_processor_message_type(deep_transformer, "out") == MockMessageC

        # Test with invalid direction
        with pytest.raises(ValueError, match="Invalid direction"):
            _get_processor_message_type(processor, "invalid")


# -- Tests for BaseProcessor and derived classes --


class TestBaseProcessor:
    def test_initialization(self):
        processor = MockProcessor(param1=20)
        assert processor.settings.param1 == 20
        assert processor.settings.param2 == "test"

    def test_get_settings_type(self):
        assert MockProcessor.get_settings_type() == MockSettings

    def test_get_message_type(self):
        assert MockProcessor.get_message_type("in") == MockMessageA
        assert MockProcessor.get_message_type("out") == MockMessageB

    def test_call_methods(self):
        processor = MockProcessor()
        result = processor(MockMessageA())
        assert isinstance(result, MockMessageB)

    @pytest.mark.asyncio
    async def test_acall_methods(self):
        processor = MockProcessor()
        result = await processor.__acall__(MockMessageA())
        assert isinstance(result, MockMessageB)

    def test_send_alias(self):
        processor = MockProcessor()
        result = processor.send(MockMessageA())
        assert isinstance(result, MockMessageB)

    @pytest.mark.asyncio
    async def test_asend_alias(self):
        processor = MockProcessor()
        result = await processor.asend(MockMessageA())
        assert isinstance(result, MockMessageB)


class TestBaseConsumer:
    def test_initialization(self):
        consumer = MockConsumer(param1=20)
        assert consumer.settings.param1 == 20
        assert consumer.settings.param2 == "test"

    def test_get_message_type(self):
        assert MockConsumer.get_message_type("in") == MockMessageB
        assert MockConsumer.get_message_type("out") is None

    def test_call_method(self):
        consumer = MockConsumer()
        # Should not raise an exception
        consumer(MockMessageB())

    @pytest.mark.asyncio
    async def test_acall_method(self):
        consumer = MockConsumer()
        # Should not raise an exception
        await consumer.__acall__(MockMessageB())


class TestBaseTransformer:
    def test_call_method(self):
        transformer = MockTransformer()
        result = transformer(MockMessageA())
        assert isinstance(result, MockMessageC)

    @pytest.mark.asyncio
    async def test_acall_method(self):
        transformer = MockTransformer()
        result = await transformer.__acall__(MockMessageA())
        assert isinstance(result, MockMessageC)


# -- Tests for BaseProducer --


class TestBaseProducer:
    def test_initialization(self):
        producer = MockProducer(param1=20)
        assert producer.settings.param1 == 20
        assert producer.settings.param2 == "test"

    def test_get_settings_type(self):
        assert MockProducer.get_settings_type() == MockSettings

    def test_get_message_type(self):
        assert MockProducer.get_message_type("in") is None
        assert MockProducer.get_message_type("out") == MockMessageA

    def test_call_method(self):
        producer = MockProducer()
        result = producer()
        assert isinstance(result, MockMessageA)

    @pytest.mark.asyncio
    async def test_acall_method(self):
        producer = MockProducer()
        result = await producer.__acall__()
        assert isinstance(result, MockMessageA)

    def test_iterator_interface(self):
        producer = MockProducer()
        result = next(producer)
        assert isinstance(result, MockMessageA)

    @pytest.mark.asyncio
    async def test_async_iterator_interface(self):
        producer = MockProducer()
        result = await producer.__anext__()
        assert isinstance(result, MockMessageA)


# -- Tests for BaseStatefulProcessor and derived classes --


class TestBaseStatefulProcessor:
    def test_initialization(self):
        processor = MockStatefulProcessor(param1=20)
        assert processor.settings.param1 == 20
        assert processor._hash == -1
        assert processor.state.iterations == 0

    def test_state_property(self):
        processor = MockStatefulProcessor()
        assert processor.state.iterations == 0

        # Modify state
        processor.state.iterations = 5
        assert processor.state.iterations == 5

    def test_state_setter_with_state_object(self):
        processor = MockStatefulProcessor()
        new_state = MockState()
        new_state.iterations = 10
        processor.state = new_state
        assert processor.state.iterations == 10

    def test_state_setter_with_bytes(self):
        processor = MockStatefulProcessor()
        new_state = MockState()
        new_state.iterations = 15
        serialized = pickle.dumps(new_state)
        processor.state = serialized
        assert processor.state.iterations == 15

    def test_hash_message_default(self):
        processor = MockStatefulProcessor()
        assert processor._hash_message(MockMessageA()) == 0

    def test_state_reset_on_first_call(self):
        processor = MockStatefulProcessor()
        assert processor._hash == -1

        # First call should trigger state reset
        processor(MockMessageA())
        assert processor._hash == 0
        assert processor.state.iterations == 1

    def test_stateful_op(self):
        processor = MockStatefulProcessor()
        state = (MockState(), -1)
        new_state, result = processor.stateful_op(state, MockMessageA())
        assert isinstance(result, MockMessageB)
        assert new_state[0].iterations == 1
        assert new_state[1] == 0  # hash updated


class TestBaseStatefulConsumer:
    def test_stateful_op(self):
        consumer = MockStatefulConsumer()
        state = (MockState(), 0)
        new_state, result = consumer.stateful_op(state, MockMessageB())
        assert result is None
        assert new_state[0].iterations == 1


class TestBaseStatefulTransformer:
    def test_stateful_op(self):
        transformer = MockStatefulTransformer()
        state = (MockState(), 0)
        new_state, result = transformer.stateful_op(state, MockMessageA())
        assert isinstance(result, MockMessageC)
        assert new_state[0].iterations == 1


# Mock SampleMessage for testing BaseAdaptiveTransformer
def mock_sample_message():
    sample_message = MagicMock(spec=SampleMessage)
    return sample_message


class TestBaseAdaptiveTransformer:
    def test_partial_fit(self):
        transformer = MockAdaptiveTransformer()
        transformer.partial_fit(mock_sample_message())
        assert transformer.state.iterations == 1

    @pytest.mark.asyncio
    async def test_apartial_fit(self):
        transformer = MockAdaptiveTransformer()
        await transformer.apartial_fit(mock_sample_message())
        assert transformer.state.iterations == 1

    def test_call_with_sample_message(self):
        transformer = MockAdaptiveTransformer()
        # Create a sample message with a trigger attribute
        sample_msg = mock_sample_message()
        setattr(sample_msg, "trigger", None)
        result = transformer(sample_msg)
        assert result is None  # partial_fit returns None
        assert transformer.state.iterations == 1


class TestBaseAsyncTransformer:
    @pytest.mark.asyncio
    async def test_acall_method(self):
        transformer = MockAsyncTransformer()
        result = await transformer.__acall__(MockMessageA())
        assert isinstance(result, MockMessageB)
        assert transformer.state.iterations == 1

    def test_call_method(self):
        transformer = MockAsyncTransformer()
        result = transformer(MockMessageA())
        assert isinstance(result, MockMessageB)
        assert transformer.state.iterations == 1


# -- Test for BaseStatefulProducer --


class TestBaseStatefulProducer:
    def test_initialization(self):
        producer = MockStatefulProducer(param1=20)
        assert producer.settings.param1 == 20
        assert producer._hash == -1
        assert producer.state.iterations == 0

    @pytest.mark.asyncio
    async def test_state_reset_on_first_call(self):
        producer = MockStatefulProducer()
        assert producer._hash == -1

        # First call should trigger state reset
        result = await producer.__acall__()
        assert producer._hash == 0
        assert producer.state.iterations == 1
        assert isinstance(result, MockMessageA)

    def test_stateful_op(self):
        producer = MockStatefulProducer()
        state = (MockState(), 0)
        new_state, result = producer.stateful_op(state)
        assert isinstance(result, MockMessageA)
        assert new_state[0].iterations == 1


# -- Tests for CompositeProcessor --


class TestCompositeProcessor:
    def test_valid_single_processor_chain(self):
        processor = ValidSingleCompositeProcessor()
        result = processor(MockMessageA())
        assert isinstance(result, MockMessageB)

    def test_valid_multiple_processor_chain(self):
        processor = ValidMultipleCompositeProcessor()
        result = processor(MockMessageA())
        assert isinstance(result, MockMessageB)

    def test_empty_processor_chain(self):
        with pytest.raises(ValueError, match="requires at least one processor"):
            EmptyCompositeProcessor()

    def test_invalid_output_type(self):
        with pytest.raises(TypeError, match="Output type mismatch"):
            InvalidOutputCompositeProcessor()

    def test_invalid_input_type(self):
        with pytest.raises(TypeError, match="Input type mismatch"):
            InvalidInputCompositeProcessor()

    def test_invalid_producer_not_first(self):
        with pytest.raises(
            TypeError,
            match="Producers can only be the first processor of a composite producer chain.",
        ):
            InvalidProducerNotFirstCompositeProcessor()

    def test_invalid_consumer_not_last(self):
        with pytest.raises(
            TypeError,
            match="Consumers can only be the last processor of a composite processor chain.",
        ):
            InvalidConsumerNotLastCompositeProcessor()

    def test_invalid_consumer_to_producer(self):
        with pytest.raises(
            TypeError,
            match="Consumers can only be the last processor of a composite processor chain.",
        ):
            InvalidConsumerProducerCompositeProcessor()

    def test_type_mismatch_between_processors(self):
        with pytest.raises(TypeError, match="Message type mismatch between processors"):
            TypeMismatchCompositeProcessor()

    def test_composite_processor_with_producer_first(self):
        with pytest.raises(
            TypeError,
            match=r"First processor .* is a producer or receives only None. Please use CompositeProducer.*",
        ):
            InvalidProducerFirstCompositeProcessor()

    def test_chained_processors(self):
        processor = ChainedCompositeProcessor()
        result = processor(MockMessageA())
        assert isinstance(result, NoneType)

    def test_chained_processors_with_deep_classes(self):
        processor = ChainedCompositeProcessorWithDeepProcessors()
        result = processor(None)
        assert isinstance(result, MockMessageC)

    def test_composite_processor_with_generator_sync(self):
        processor = MockGeneratorCompositeProcessor()
        # Attempt to use the sync processing interface and assert it raises an error
        result = processor._process(MockMessageA())
        assert isinstance(result, MockMessageA)

    @pytest.mark.asyncio
    async def test_composite_processor_with_generator_async(self):
        processor = MockGeneratorCompositeProcessor()
        result = await processor._aprocess(MockMessageA())
        assert isinstance(result, MockMessageA)

    def test_state_property(self):
        processor1 = ValidMultipleCompositeProcessor()
        processor2 = ChainedCompositeProcessorWithDeepProcessors()
        processor3 = MockGeneratorCompositeProcessor()

        state1 = processor1.state
        assert "processor" not in state1
        assert "stateful_processor" in state1

        state2 = processor2.state
        assert "processor" not in state2
        assert "deep_processor" not in state2
        assert "deeper_transformer" not in state2

        state3 = processor3.state
        assert "generator" not in state3
        assert "stateful_processor" in state3

    def test_state_setter(self):
        processor = ChainedCompositeProcessor()

        # Create new states
        state1 = MockState()
        state1.iterations = 5
        state2 = MockState()
        state2.iterations = 10

        # Set composite state
        processor.state = {"processor": state1, "stateful_consumer": state2}

        assert not hasattr(processor._procs["processor"], "state")
        assert processor._procs["stateful_consumer"].state.iterations == 10

        # Attempt to set state for non-existent processor
        with pytest.raises(
            KeyError,
            match=r"Processor .* in provided state not found in composite processor chain.",
        ):
            processor.state = {"non_existent_processor": state1}

    def test_state_setter_with_bytes(self):
        processor = ChainedCompositeProcessor()

        # Create new states
        state1 = MockState()
        state1.iterations = 15
        state2 = MockState()
        state2.iterations = 20

        # Serialize composite state
        serialized = pickle.dumps({"processor": state1, "stateful_consumer": state2})
        processor.state = serialized
        assert not hasattr(processor._procs["processor"], "state")
        assert processor._procs["stateful_consumer"].state.iterations == 20

    @pytest.mark.asyncio
    async def test_aprocess_method(self):
        processor = ValidMultipleCompositeProcessor()
        result = await processor._aprocess(MockMessageA())
        assert isinstance(result, MockMessageB)

    def test_stateful_op(self):
        processor = ValidMultipleCompositeProcessor()
        state = {"processor": (MockState(), 2), "stateful_processor": (MockState(), 3)}
        new_state, result = processor.stateful_op(state, MockMessageA())
        assert isinstance(result, MockMessageB)
        # Do we really want to keep the 'state' of a stateless processor?
        # State of processor 1 not changed from default values
        assert new_state["processor"][0].iterations == 0
        assert new_state["processor"][0].hash == -1
        assert hasattr(processor._procs["processor"], "_state") is False
        # State of processor 2 updated
        assert new_state["stateful_processor"][0].iterations == 1
        assert new_state["stateful_processor"][0].hash == -1
        assert processor._procs["stateful_processor"].state.iterations == 1
        # Hash not set to 3 as expected as processor is called after setting the hash
        # Hash set to 0 via the _reset_state method.
        assert processor._procs["stateful_processor"]._hash == 0


# -- Tests for CompositeProducer --


class TestCompositeProducer:
    def test_valid_single_producer_chain(self):
        producer = ValidSingleCompositeProducer()
        result = producer()
        assert isinstance(result, MockMessageA)

    def test_valid_multiple_producer_chain(self):
        producer = ValidMultipleCompositeProducer()
        result = producer()
        assert isinstance(result, MockMessageB)

    def test_empty_producer_chain(self):
        with pytest.raises(ValueError, match="requires at least one processor"):
            EmptyCompositeProducer()

    def test_invalid_output_type(self):
        with pytest.raises(TypeError, match="Output type mismatch"):
            InvalidOutputCompositeProducer()

    def test_invalid_producer_not_first(self):
        with pytest.raises(
            TypeError, match="Input type mismatch: Composite producer expects None"
        ):
            InvalidProducerNotFirstCompositeProducer()

    def test_invalid_consumer_not_last(self):
        with pytest.raises(
            TypeError,
            match="Consumers can only be the last processor of a composite producer chain.",
        ):
            InvalidConsumerNotLastCompositeProducer()

    def test_type_mismatch_between_processors(self):
        with pytest.raises(TypeError, match="Message type mismatch between processors"):
            TypeMismatchCompositeProducer()

    def test_chained_producers_with_deep_classes(self):
        producer = ChainedCompositeProducerWithDeepProcessors()
        result = producer()
        assert isinstance(result, MockMessageC)

    @pytest.mark.asyncio
    async def test_composite_producer_with_generator(self):
        producer = MockGeneratorCompositeProducer()
        result = await producer._produce()
        assert isinstance(result, MockMessageA)

    def test_state_property(self):
        producer1 = ValidMultipleCompositeProducer()
        producer2 = ChainedCompositeProducerWithDeepProcessors()
        producer3 = MockGeneratorCompositeProducer()

        state1 = producer1.state
        assert "producer" not in state1
        assert "stateful_processor" in state1

        state2 = producer2.state
        assert "stateful_producer" in state2
        assert "deep_processor" not in state2
        assert "deeper_transformer" not in state2

        state3 = producer3.state
        assert "generator" not in state3
        assert "stateful_processor" in state3

    def test_state_setter(self):
        producer = ValidMultipleCompositeProducer()

        # Create new states
        state1 = MockState()
        state1.iterations = 5
        state2 = MockState()
        state2.iterations = 10

        # Set composite state
        producer.state = {"producer": state1, "stateful_processor": state2}

        assert not hasattr(producer._procs["producer"], "state")
        assert producer._procs["stateful_processor"].state.iterations == 10

        # Attempt to set state for non-existent processor
        with pytest.raises(
            KeyError,
            match=r"Processor .* in provided state not found in composite producer chain.",
        ):
            producer.state = {"non_existent_processor": state1}

    def test_state_setter_with_bytes(self):
        producer = ValidMultipleCompositeProducer()

        # Create new states
        state1 = MockState()
        state1.iterations = 15
        state2 = MockState()
        state2.iterations = 20

        # Serialize composite state
        serialized = pickle.dumps({"producer": state1, "stateful_processor": state2})
        producer.state = serialized
        assert not hasattr(producer._procs["producer"], "state")
        assert producer._procs["stateful_processor"].state.iterations == 20

    @pytest.mark.asyncio
    async def test_produce_method(self):
        producer = ValidMultipleCompositeProducer()
        result = await producer._produce()
        assert isinstance(result, MockMessageB)

    def test_stateful_op(self):
        composite_producer = ValidMultipleCompositeProducer()
        state = {"producer": (MockState(), 2), "stateful_processor": (MockState(), 3)}
        new_state, result = composite_producer.stateful_op(state)
        assert isinstance(result, MockMessageB)
        # Do we really want to keep the 'state' of a stateless processor?
        # State of producer not changed from default values
        assert new_state["producer"][0].iterations == 0
        assert new_state["producer"][0].hash == -1
        assert hasattr(composite_producer._procs["producer"], "_state") is False
        # State of stateful_processor updated
        assert new_state["stateful_processor"][0].iterations == 1
        assert new_state["stateful_processor"][0].hash == -1
        assert composite_producer._procs["stateful_processor"].state.iterations == 1
        # Hash not set to 3 as expected as processor is called after setting the hash
        # Hash set to 0 via the _reset_state method.
        assert composite_producer._procs["stateful_processor"]._hash == 0
