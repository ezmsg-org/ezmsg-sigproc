ezmsg.sigproc
==============

Timeseries signal processing modules for the `ezmsg <https://www.ezmsg.org>`_ framework.

Overview
--------

``ezmsg-sigproc`` provides signal processing primitives built on ezmsg, leveraging numpy, scipy, pywavelets, and sparse. The package offers both standalone processors for offline analysis and Unit wrappers for streaming pipelines.

Key features:

* **Filtering** - Various filter implementations (Chebyshev, comb filters, etc.)
* **Spectral analysis** - Spectrogram, spectrum, and wavelet transforms
* **Resampling** - Downsample, decimate, and resample operations
* **Windowing** - Sliding windows and buffering utilities
* **Math operations** - Arithmetic, log, abs, difference, and more
* **Signal generation** - Synthetic signal generators

All modules use :class:`ezmsg.util.messages.axisarray.AxisArray` as the primary data structure for passing signals between components.

.. note::
   Processors can be used standalone for offline analysis or integrated into ezmsg pipelines for real-time streaming applications.

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install ezmsg-sigproc

Or install the latest development version:

.. code-block:: bash

   pip install git+https://github.com/ezmsg-org/ezmsg-sigproc@dev

Dependencies
^^^^^^^^^^^^

Core dependencies:

* ``ezmsg`` - Core messaging framework
* ``numpy`` - Numerical computing
* ``scipy`` - Scientific computing and signal processing
* ``pywavelets`` - Wavelet transforms
* ``sparse`` - Sparse array operations
* ``numba`` - JIT compilation for performance

Quick Start
-----------

For general ezmsg tutorials and guides, visit `ezmsg.org <https://www.ezmsg.org>`_.

For package-specific documentation:

* **Processor Architecture** - See :doc:`guides/ProcessorsBase` for details on the processor hierarchy
* **How-To Guides** - See :doc:`guides/how-tos/signalprocessing/content-signalprocessing` for usage patterns
* **API Reference** - See :doc:`api/index` for complete API documentation

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guides/ProcessorsBase
   guides/HybridBuffer
   guides/how-tos/signalprocessing/content-signalprocessing
   guides/tutorials/signalprocessing
   guides/sigproc/content-sigproc
   api/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
