"""
Backwards-compatible re-exports from ezmsg.baseproc.

This module re-exports all symbols from ezmsg.baseproc to maintain backwards
compatibility for code that imports from ezmsg.sigproc.base.

New code should import directly from ezmsg.baseproc instead.
"""

import warnings

warnings.warn(
    "Importing from 'ezmsg.sigproc.base' is deprecated. Please import from 'ezmsg.baseproc' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from ezmsg.baseproc for backwards compatibility
from ezmsg.baseproc import (  # noqa: E402
    # Protocols
    AdaptiveTransformer,
    # Type variables
    AdaptiveTransformerType,
    # Stateful classes
    BaseAdaptiveTransformer,
    # Unit classes
    BaseAdaptiveTransformerUnit,
    BaseAsyncTransformer,
    # Base processor classes
    BaseConsumer,
    BaseConsumerUnit,
    BaseProcessor,
    BaseProcessorUnit,
    BaseProducer,
    BaseProducerUnit,
    BaseStatefulConsumer,
    BaseStatefulProcessor,
    BaseStatefulProducer,
    BaseStatefulTransformer,
    BaseTransformer,
    BaseTransformerUnit,
    # Composite classes
    CompositeProcessor,
    CompositeProducer,
    CompositeStateful,
    Consumer,
    ConsumerType,
    GenAxisArray,
    MessageInType,
    MessageOutType,
    Processor,
    Producer,
    ProducerType,
    # Message types
    SampleMessage,
    SettingsType,
    Stateful,
    StatefulConsumer,
    StatefulProcessor,
    StatefulProducer,
    StatefulTransformer,
    StateType,
    Transformer,
    TransformerType,
    # Type resolution helpers
    _get_base_processor_message_in_type,
    _get_base_processor_message_out_type,
    _get_base_processor_settings_type,
    _get_base_processor_state_type,
    _get_processor_message_type,
    # Type utilities
    check_message_type_compatibility,
    get_base_adaptive_transformer_type,
    get_base_consumer_type,
    get_base_producer_type,
    get_base_transformer_type,
    is_sample_message,
    # Decorators
    processor_state,
    # Profiling
    profile_subpub,
    resolve_typevar,
)

__all__ = [
    # Protocols
    "Processor",
    "Producer",
    "Consumer",
    "Transformer",
    "StatefulProcessor",
    "StatefulProducer",
    "StatefulConsumer",
    "StatefulTransformer",
    "AdaptiveTransformer",
    # Type variables
    "MessageInType",
    "MessageOutType",
    "SettingsType",
    "StateType",
    "ProducerType",
    "ConsumerType",
    "TransformerType",
    "AdaptiveTransformerType",
    # Decorators
    "processor_state",
    # Base processor classes
    "BaseProcessor",
    "BaseProducer",
    "BaseConsumer",
    "BaseTransformer",
    # Stateful classes
    "Stateful",
    "BaseStatefulProcessor",
    "BaseStatefulProducer",
    "BaseStatefulConsumer",
    "BaseStatefulTransformer",
    "BaseAdaptiveTransformer",
    "BaseAsyncTransformer",
    # Composite classes
    "CompositeStateful",
    "CompositeProcessor",
    "CompositeProducer",
    # Unit classes
    "BaseProducerUnit",
    "BaseProcessorUnit",
    "BaseConsumerUnit",
    "BaseTransformerUnit",
    "BaseAdaptiveTransformerUnit",
    "GenAxisArray",
    # Type resolution helpers
    "get_base_producer_type",
    "get_base_consumer_type",
    "get_base_transformer_type",
    "get_base_adaptive_transformer_type",
    "_get_base_processor_settings_type",
    "_get_base_processor_message_in_type",
    "_get_base_processor_message_out_type",
    "_get_base_processor_state_type",
    "_get_processor_message_type",
    # Message types
    "SampleMessage",
    "is_sample_message",
    # Profiling
    "profile_subpub",
    # Type utilities
    "check_message_type_compatibility",
    "resolve_typevar",
]
