## Processor Base Classes

The `ezmsg.sigproc.base` module contains the base classes for the signal processors. The base classes are designed to allow users to create custom signal processors with minimal errors and minimal repetition of boilerplate code.

> The information below was written at the time of a major refactor to `ezmsg.sigproc.base` to help collate the design decisions and to help with future refactoring.  However, it may be outdated or incomplete. Please refer to the source code for the most accurate information.

### Generic TypeVars

| Idx | Class             | Description                                                                     |
|-----|-------------------|---------------------------------------------------------------------------------|
| 1   | `MessageType` (M) | for messages                                                                    |
| 2   | `SettingsType`    | bound to ez.Settings                                                            |
| 3   | `StateType` (St)  | bound to ProcessorState which is simply ez.State with a `hash: int` field.      |
| 4   | `ProcessorType`   | bound to BaseMessageProcessor, BaseStatelessProcessor                           |
| 5   | `TransformerType` | bound to BaseStatelessTransformer, BaseMessageTransformer, CompositeTransformer |


### Protocols

| Idx | Class                  | Parent | State | `__call__` | `stateful_op` | @state | partial_fit | __acall__ |
|-----|------------------------|--------|-------|------------|---------------|--------|-------------|-----------|
| 1   | `Processor`            | -      | No    | [M, None]  | -             | -      | -           | -         |
| 2   | `Consumer`             | 1      | No    | None       | -             | -      | -           | -         |
| 3   | `Transformer`          | 1      | No    | M          | -             | -      | -           | -         |
| 4   | `StatefulProcessor`    | -      | Yes   | [M, None]  | St, [M, None] | Y      | -           | -         |
| 5   | `StatefulConsumer`     | 4      | Yes   | None       | St, None      | Y      | -           | -         |
| 6   | `StatefulTransformer`  | 4      | Yes   | M          | St, M         | Y      | -           | -         |
| 7   | `AdaptiveTransformer`  | 6      | Yes   | M          | St, M         | Y      | Y           | -         |
| 8   | `AsyncTransformer`     | 6      | Yes   | M          | St, M         | Y      | N           | Y         |

> I have yet to encounter a use case for `AdaptiveConsumer` but it can be added if requested.
> We probably need more `Async` processors. Please ask if you need `AsyncConsumer` or `AdaptiveAsyncTransformer`.

### Abstract implementations (Base Classes) for standalone processors

| Idx | Class                     | Parent | Protocol | Features                                                                      |
|-----|---------------------------|--------|----------|-------------------------------------------------------------------------------|
| 1   | `BaseProcessor`           | -      | 1        | `__init__` for settings; `__call__` (alias: `send`) wraps abstract `_process` |
| 2   | `BaseConsumer`            | 1      | 2        | Override return types only                                                    |
| 3   | `BaseTransformer`         | 1      | 3        | Override return types only                                                    |
| 4   | `BaseStatefulProcessor`   | -      | 4        | `state` setter unpickles arg; `stateful_op` wraps `__call__`                  |
| 5   | `BaseStatefulConsumer`    | 3      | 5        | Override return types only                                                    |
| 6   | `BaseStatefulTransformer` | 3      | 6        | Override return types only                                                    |
| 7   | `BaseAdaptiveTransformer` | 6      | 7        | `__call__` may call partial_fit if message has .trigger                       |
| 8   | `BaseAsyncTransformer`    | 6      | 8        | `__acall__` wraps abstract `_aprocess`; `asend` alias                         |
| 9   | `CompositeProcessor`      | 4      | 4        | `state` getter/setter, `_process`, and `stateful_op` interface self._procs.   |


### Generic TypeVars for ezmsg Units

| Idx | Class                     | Description                                                        |
|-----|---------------------------|--------------------------------------------------------------------|
| 1   | `ConsumerType`            | bound BaseConsumer, BaseStatefulConsumer                           |
| 2   | `TransformerType`         | bound BaseTransformer, BaseStatefulTransformer, CompositeProcessor |
| 3   | `AdaptiveTransformerType` | bound BaseAdaptiveTransformer                                      |
| 4   | `AsyncTransformerType`    | bound BaseAsyncTransformer                                         |


### Abstract implementations (Base Classes) for ezmsg Units using processors:

| Idx | Class                         | Parents | Expected TypeVars         |
|-----|-------------------------------|---------|---------------------------|
| 1   | `BaseConsumerUnit`            | -       | `ConsumerType`            |
| 2   | `BaseTransformerUnit`         | 1       | `TransformerType`         |
| 3   | `BaseAdaptiveTransformerUnit` | 2       | `AdaptiveTransformerType` |
| 4   | `BaseAsyncTransformerUnit`    | 2       | `AsyncTransformerType`    |


## Implementing a custom standalone processor

1. Create a new settings dataclass: `class MySettings(ez.Settings):`
2. Create a new state dataclass: `class MyState(ProcessorState):`
3. Decide which protocol to implement, then choose the appropriate base class from the table.
    * It is unlikely you will want to implement `BaseProcessor` or `BaseStatefulProcessor` directly.
    * It is more likely you want to implement a consumer or transformer.
    * Is your processor stateful? Most are.
    * Is your processor adaptive, i.e. does it learn from labels or other non-streaming representations of the data?
    * Do you need async processing?
    * Is your processor merely a composite of other processors?
4. Create the derived class and implement the abstract methods.
    * For any stateful processor, implement `_reset_state`.
    * For stateful processors that need to respond to a change in the incoming data, implement `_hash_message`.
    * For all processors, implement `_process` OR `_aprocess` for sync and async processors, respectively.
        * Consumers will return `None` and transformers will return a message.
    * For strictly composite processors, ignore the above and implement only `_initialize_processors`.
5. Override non-abstract methods if you need special behaviour. For example:
    * `WindowTransformer` overrides `__init__` to do some sanity checks on the provided settings.
    * `TransposeTransformer` and `WindowTransformer` override `__call__` to provide a passthrough shortcut when the settings allow for it.

## Implementing a custom ezmsg Unit

1. Create a custom standalone processor as above.
2. Decide which base unit to implement. This will be dictated by the type of processor you have created.
3. Create the derived class. Usually there is nothing to implement.

Often, all that is required is the following:

```Python
class CustomUnit(TransformerUnit[
        CustomTransformerSettings,    # SettingsType
        AxisArray,                    # MessageType
        CustomTransformer,            # TransformerType
    ]):
        SETTINGS = CustomTransformerSettings
```