Array API Support
=================

ezmsg-sigproc provides support for the `Python Array API standard
<https://data-apis.org/array-api/>`_, enabling many transformers to work with
arrays from different backends such as NumPy, CuPy, PyTorch, JAX, and MLX.

What is the Array API?
----------------------

The Array API is a standardized interface for array operations across different
Python array libraries. By coding to this standard, ezmsg-sigproc transformers
can process data regardless of which array library created it, enabling:

- **GPU acceleration** via CuPy, PyTorch, or JAX tensors
- **Apple Silicon acceleration** via MLX
- **Framework interoperability** for integration with ML pipelines
- **Hardware flexibility** without code changes

How It Works
------------

Compatible transformers use `array-api-compat <https://github.com/data-apis/array-api-compat>`_
to detect the input array's namespace and use the appropriate operations:

.. code-block:: python

    from array_api_compat import get_namespace

    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)  # numpy, cupy, torch, mlx.core, etc.
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
   * - :mod:`ezmsg.sigproc.spectrum`
     - FFT-based spectrum (SpectrumTransformer)
   * - :mod:`ezmsg.sigproc.aggregate`
     - Aggregate operations (AggregateTransformer, RangedAggregateTransformer)
   * - :mod:`ezmsg.sigproc.diff`
     - Compute differences along an axis
   * - :mod:`ezmsg.sigproc.transpose`
     - Transpose/permute array dimensions
   * - :mod:`ezmsg.sigproc.linear`
     - Per-channel linear transform (scale + offset)

Coordinate Transforms
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - :mod:`ezmsg.sigproc.coordinatespaces`
     - Cartesian/polar coordinate conversions

Composite Pipelines
^^^^^^^^^^^^^^^^^^^

These ``CompositeProcessor`` pipelines chain Array API-aware steps together.
When fed non-NumPy arrays, each step in the pipeline preserves the backend:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - :mod:`ezmsg.sigproc.bandpower`
     - BandPowerTransformer (spectrogram + ranged aggregate)
   * - :mod:`ezmsg.sigproc.singlebandpow`
     - RMSBandPowerTransformer (with explicit ``backend`` setting; only after initial IIR filter)

MLX on Apple Silicon
--------------------

`MLX <https://github.com/ml-explore/mlx>`_ is an array library for Apple Silicon
that provides GPU-accelerated operations with a NumPy-like API. ezmsg-sigproc's
Array API support enables MLX acceleration for spectral analysis and other
pipelines without code changes to the transformers themselves.

Basic usage
^^^^^^^^^^^

Pass MLX arrays in your ``AxisArray`` messages:

.. code-block:: python

    import mlx.core as mx
    import numpy as np
    from ezmsg.util.messages.axisarray import AxisArray
    from ezmsg.sigproc.spectrum import SpectrumTransformer, SpectrumSettings

    # Create data as MLX array
    np_data = np.random.randn(1000, 64).astype(np.float32)
    message = AxisArray(
        data=mx.array(np_data),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=1000.0)},
    )

    proc = SpectrumTransformer(SpectrumSettings(axis="time"))
    result = proc(message)
    # result.data is an mlx.core.array

Lazy evaluation and ``mx.eval``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MLX uses **lazy evaluation** — computations are not executed until their results
are needed. This allows MLX to fuse operations and optimize the computation
graph. However, it means that timing code or downstream consumers may see
artificially fast "processing" that is actually deferred.

To force evaluation, call ``mx.eval()``:

.. code-block:: python

    result = proc(message)
    mx.eval(result.data)  # Forces computation to complete

For ``CompositeProcessor`` pipelines (like ``BandPowerTransformer``), you can
override ``_post_process`` to call ``mx.eval()`` automatically so that every
output is fully materialized:

.. code-block:: python

    class BandPowerTransformer(CompositeProcessor[BandPowerSettings, AxisArray, AxisArray]):
        @staticmethod
        def _initialize_processors(settings):
            return {
                "spectrogram": SpectrogramTransformer(settings=settings.spectrogram_settings),
                "aggregate": RangedAggregateTransformer(...),
            }

        def _post_process(self, result: AxisArray | None) -> AxisArray | None:
            if result is not None:
                try:
                    import mlx.core as mx

                    if isinstance(result.data, mx.array):
                        mx.eval(result.data)
                except ImportError:
                    pass
            return result

This pattern is used by ``BandPowerTransformer`` and ``RMSBandPowerTransformer``.
It ensures downstream consumers (ezmsg Units, visualization, logging) receive
fully evaluated arrays without needing to know about MLX internals.
It also provides a safety valve so the lazy graph does not accumulate if the graph
is not evaluated at the right time downstream.

.. note::

    The ``_post_process`` hook is defined on ``CompositeProcessor`` in
    ezmsg-baseproc. It runs after the entire processor chain completes and
    receives the final output. The ``try``/``except ImportError`` pattern
    keeps MLX as an optional dependency.

MLX quirks
^^^^^^^^^^

MLX's Array API coverage is nearly complete but has a few gaps that
ezmsg-sigproc works around internally:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - MLX status
     - Workaround
   * - ``fft(norm=...)``
     - Not supported
     - Manual normalization (``/ n``, ``/ sqrt(n)``)
   * - ``fftshift(axes=int)``
     - Needs tuple
     - Always pass ``axes=(idx,)``
   * - ``fftfreq`` / ``rfftfreq``
     - Not available
     - Computed with NumPy (metadata only)
   * - ``dtype.kind``
     - No ``.kind`` attribute
     - ``is_complex_dtype()`` helper in ``ezmsg.sigproc.util.array``
   * - Window functions
     - Not available
     - Computed with NumPy, converted via ``xp.asarray()``
   * - ``nan*`` functions
     - Not available
     - Falls back to NumPy automatically
   * - Boolean indexing
     - Not supported
     - Avoided in hot paths; used only in NumPy metadata code
   * - Slice with ``np.int64``
     - Rejected
     - Slice bounds cast to Python ``int``

These workarounds are handled inside the transformers — user code does not need
to account for them.

Limitations
-----------

Some operations remain NumPy-only due to lack of Array API equivalents:

- **SciPy operations**: Butterworth filtering (``scipy.signal.sosfilt``) and
  other scipy-dependent steps. Use ``AsArrayTransformer`` to convert between
  backends at pipeline boundaries (see ``RMSBandPowerTransformer`` for an example).
- **Random number generation**: Modules using ``np.random`` (e.g., ``denormalize``)
- **Trapezoidal integration**: ``np.trapezoid`` has no Array API equivalent.
  ``RangedAggregateTransformer`` falls back to NumPy transparently.
- **Memory layout**: ``np.require`` for contiguous array optimization

Metadata arrays (axis labels, coordinates) always remain as NumPy arrays
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

1. Call ``get_namespace(message.data)`` at the start of ``_process`` (or
   ``_reset_state`` for stateful transformers).
2. Use ``xp.function_name`` instead of ``np.function_name`` for all operations
   on ``message.data``.
3. Note that some functions have different names:
   - ``np.concatenate`` → ``xp.concat``
   - ``np.transpose`` → ``xp.permute_dims``
4. Keep metadata operations (axis labels, etc.) as NumPy.
5. When a backend lacks a function (e.g., MLX has no ``nanmean``), fall back
   gracefully:

   .. code-block:: python

       func_name = "mean"
       if hasattr(xp, func_name):
           result = getattr(xp, func_name)(data, axis=axis_idx)
       else:
           result = np.mean(np.asarray(data), axis=axis_idx)

6. For ``CompositeProcessor`` subclasses that may produce MLX output, add
   a ``_post_process`` override to call ``mx.eval()`` (see the MLX section
   above).
7. Use portable helpers from ``ezmsg.sigproc.util.array`` when needed:
   ``is_complex_dtype``, ``is_float_dtype``, ``xp_asarray``.
