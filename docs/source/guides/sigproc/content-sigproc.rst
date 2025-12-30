ezmsg-sigproc
===============

Timeseries signal processing implementations in ezmsg, leveraging numpy and scipy.
Most of the methods and classes in this extension are intended to be used in building signal processing pipelines.
They use :class:`ezmsg.util.messages.axisarray.AxisArray` as the primary data structure for passing signals between components.
The message's data are typically NumPy arrays, though many transformers support the
:doc:`Array API standard <../explanations/array_api>` for use with CuPy, PyTorch, and other backends.

.. note:: Some generators might yield valid :class:`AxisArray` messages with ``.data`` size of 0.
This may occur when the generator receives inadequate data to produce a valid output, such as when windowing or buffering.

`ezmsg-sigproc` contains two types of modules:

- base processors and units that provide fundamental building blocks for signal processing pipelines
- in-built signal processing modules that implement common signal processing techniques

.. toctree::
    :maxdepth: 1

    architecture
    base
    units
    processors
    ../explanations/array_api
