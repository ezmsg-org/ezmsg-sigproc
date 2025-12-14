## Signal Processor Base Classes

The signal processors in `ezmsg-sigproc` are built on top of the base processor classes from `ezmsg-baseproc`. For comprehensive documentation on the processor architecture, including protocols, base classes, and implementation patterns, see the [ezmsg-baseproc documentation](https://www.ezmsg.org/ezmsg-baseproc/guides/ProcessorsBase.html).

### Quick Reference

The base classes provide a consistent pattern for building message processors:

* **Processors** - Transform input messages to output messages
* **Producers** - Generate output messages without requiring input
* **Consumers** - Accept input messages without producing output
* **Transformers** - A specific type of processor with typed input/output
* **Stateful variants** - Processors that maintain state across invocations
* **Adaptive transformers** - Transformers that can be trained via `partial_fit`
* **Composite processors** - Chain multiple processors together efficiently

### Signal Processing Example

Here's an example of a signal processing transformer using `AxisArray`:

```Python
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit


class ScaleTransformerSettings(ez.Settings):
    scale: float = 1.0


class ScaleTransformer(BaseTransformer[ScaleTransformerSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        return replace(message, data=message.data * self.settings.scale)


class ScaleUnit(BaseTransformerUnit[
        ScaleTransformerSettings,
        AxisArray,
        AxisArray,
        ScaleTransformer,
    ]):
        SETTINGS = ScaleTransformerSettings
```

### Existing Signal Processors

For examples of signal processing implementations, see the processors in `ezmsg.sigproc`:

* **Filtering** - `ChebyshevFilterTransformer`, `CombFilterTransformer`, etc.
* **Spectral** - `SpectrogramTransformer`, `SpectrumTransformer`, `WaveletTransformer`
* **Resampling** - `DownsampleTransformer`, `DecimateTransformer`, `ResampleTransformer`
* **Windowing** - `WindowTransformer`, `AggregateTransformer`
* **Math** - `ScalerTransformer`, `LogTransformer`, `AbsTransformer`, `DifferenceTransformer`

### Learn More

* [Processor Base Classes (ezmsg-baseproc)](https://www.ezmsg.org/ezmsg-baseproc/guides/ProcessorsBase.html) - Full documentation on the processor architecture
* [API Reference](../api/index) - Complete API documentation for ezmsg-sigproc
