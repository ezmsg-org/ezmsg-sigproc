How to turn a signal processor into an ``ezmsg`` Unit?
######################################################

For general guidance on converting processors to ezmsg Units, see the `ezmsg-baseproc documentation <https://www.ezmsg.org/ezmsg-baseproc/guides/ProcessorsBase.html#implementing-a-custom-ezmsg-unit>`_.

Example with Signal Processing
------------------------------

To convert a signal processor to an ``ezmsg`` Unit:

1. **Define the Processor**: Create a class that inherits from the appropriate base class (e.g., ``BaseTransformer``, ``BaseStatefulTransformer``).
2. **Implement the Processing Logic**: Override the ``_process`` method to implement the signal processing logic.
3. **Create the Unit**: Inherit from the appropriate Unit base class (e.g., ``BaseTransformerUnit``).

.. code-block:: python

   import ezmsg.core as ez
   from ezmsg.util.messages.axisarray import AxisArray
   from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit


   class MyProcessorSettings(ez.Settings):
       # Your settings here
       pass


   class MyProcessor(BaseTransformer[MyProcessorSettings, AxisArray, AxisArray]):
       def _process(self, message: AxisArray) -> AxisArray:
           # Your signal processing logic here
           return message


   class MyUnit(BaseTransformerUnit[
           MyProcessorSettings,
           AxisArray,
           AxisArray,
           MyProcessor,
       ]):
       SETTINGS = MyProcessorSettings

(under construction)
