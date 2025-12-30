Array API Support
=================

ezmsg-sigproc provides support for the `Python Array API standard
<https://data-apis.org/array-api/>`_, enabling many transformers to work with
arrays from different backends such as NumPy, CuPy, PyTorch, and JAX.

What is the Array API?
----------------------

The Array API is a standardized interface for array operations across different
Python array libraries. By coding to this standard, ezmsg-sigproc transformers
can process data regardless of which array library created it, enabling:

- **GPU acceleration** via CuPy or PyTorch tensors
- **Framework interoperability** for integration with ML pipelines
- **Hardware flexibility** without code changes

How It Works
------------

Compatible transformers use `array-api-compat <https://github.com/data-apis/array-api-compat>`_
to detect the input array's namespace and use the appropriate operations:

.. code-block:: python

    from array_api_compat import get_namespace

    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)  # numpy, cupy, torch, etc.
        result = xp.abs(message.data)     # Uses the correct backend
        return replace(message, data=result)

Usage Example
-------------

Using Array API compatible transformers with CuPy for GPU acceleration:

.. code-block:: python

    import cupy as cp
    from ezmsg.util.messages.axisarray import AxisArray
    from ezmsg.sigproc.math.abs import AbsTransformer
    from ezmsg.sigproc.math.clip import ClipTransformer, ClipSettings

    # Create data on GPU
    gpu_data = cp.random.randn(1000, 64).astype(cp.float32)
    message = AxisArray(gpu_data, dims=["time", "ch"])

    # Process entirely on GPU - no data transfer!
    abs_transformer = AbsTransformer()
    clip_transformer = ClipTransformer(ClipSettings(min=0.0, max=1.0))

    result = clip_transformer(abs_transformer(message))
    # result.data is still a CuPy array on GPU

Compatible Modules
------------------

The following transformers fully support the Array API standard:

Math Operations
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - :mod:`ezmsg.sigproc.math.abs`
     - Absolute value
   * - :mod:`ezmsg.sigproc.math.clip`
     - Clip values to a range
   * - :mod:`ezmsg.sigproc.math.log`
     - Logarithm with configurable base
   * - :mod:`ezmsg.sigproc.math.scale`
     - Multiply by a constant
   * - :mod:`ezmsg.sigproc.math.invert`
     - Compute 1/x
   * - :mod:`ezmsg.sigproc.math.difference`
     - Subtract a constant (ConstDifferenceTransformer)

Signal Processing
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - :mod:`ezmsg.sigproc.diff`
     - Compute differences along an axis
   * - :mod:`ezmsg.sigproc.transpose`
     - Transpose/permute array dimensions
   * - :mod:`ezmsg.sigproc.linear`
     - Per-channel linear transform (scale + offset)
   * - :mod:`ezmsg.sigproc.aggregate`
     - Aggregate operations (AggregateTransformer only)

Coordinate Transforms
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - :mod:`ezmsg.sigproc.coordinatespaces`
     - Cartesian/polar coordinate conversions

Limitations
-----------

Some operations remain NumPy-only due to lack of Array API equivalents:

- **Random number generation**: Modules using ``np.random`` (e.g., ``denormalize``)
- **SciPy operations**: Filtering (``scipy.signal.lfilter``), FFT, wavelets
- **Advanced indexing**: Some slicing operations for metadata handling
- **Memory layout**: ``np.require`` for contiguous array optimization (NumPy only)

Metadata arrays (axis labels, coordinates) typically remain as NumPy arrays
since they are not performance-critical.

Adding Array API Support
------------------------

When contributing new transformers, follow this pattern:

.. code-block:: python

    from array_api_compat import get_namespace
    from ezmsg.baseproc import BaseTransformer
    from ezmsg.util.messages.axisarray import AxisArray
    from ezmsg.util.messages.util import replace

    class MyTransformer(BaseTransformer[MySettings, AxisArray, AxisArray]):
        def _process(self, message: AxisArray) -> AxisArray:
            xp = get_namespace(message.data)

            # Use xp instead of np for array operations
            result = xp.sqrt(xp.abs(message.data))

            return replace(message, data=result)

Key guidelines:

1. Call ``get_namespace(message.data)`` at the start of ``_process``
2. Use ``xp.function_name`` instead of ``np.function_name``
3. Note that some functions have different names:
   - ``np.concatenate`` → ``xp.concat``
   - ``np.transpose`` → ``xp.permute_dims``
4. Keep metadata operations (axis labels, etc.) as NumPy
5. Use in-place operations (``/=``, ``*=``) where possible for efficiency
