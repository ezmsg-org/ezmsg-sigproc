API Reference
=============

This page contains the complete API reference for ``ezmsg.sigproc``.

.. contents:: Modules
   :local:
   :depth: 1

Base Processors
---------------

Core processor protocols and base classes.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.base

Filtering
---------

Various filter implementations for signal processing.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.filter
   ezmsg.sigproc.butterworthfilter
   ezmsg.sigproc.cheby
   ezmsg.sigproc.combfilter
   ezmsg.sigproc.adaptive_lattice_notch
   ezmsg.sigproc.firfilter
   ezmsg.sigproc.kaiser
   ezmsg.sigproc.ewmfilter
   ezmsg.sigproc.filterbank
   ezmsg.sigproc.filterbankdesign
   ezmsg.sigproc.gaussiansmoothing

Spectral Analysis
-----------------

Spectral and frequency domain analysis tools.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.spectral
   ezmsg.sigproc.spectrogram
   ezmsg.sigproc.spectrum
   ezmsg.sigproc.wavelets
   ezmsg.sigproc.bandpower
   ezmsg.sigproc.fbcca

Sampling & Resampling
---------------------

Signal sampling, windowing, and resampling operations.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.sampler
   ezmsg.sigproc.window
   ezmsg.sigproc.resample
   ezmsg.sigproc.downsample
   ezmsg.sigproc.decimate

Signal Conditioning
-------------------

Signal preprocessing and conditioning operations.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.scaler
   ezmsg.sigproc.detrend
   ezmsg.sigproc.activation
   ezmsg.sigproc.quantize
   ezmsg.sigproc.ewma

Transformations
---------------

Geometric and structural transformations.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.affinetransform
   ezmsg.sigproc.transpose
   ezmsg.sigproc.extract_axis
   ezmsg.sigproc.slicer

Signal Operations
-----------------

Aggregation, difference, and other signal operations.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.aggregate
   ezmsg.sigproc.diff

Signal Generation
-----------------

Synthetic signal generators and injectors.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.synth
   ezmsg.sigproc.signalinjector

Messages & Data Structures
---------------------------

Message types and data structures.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.messages

Math Utilities
--------------

Mathematical operations on signals.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.math

Utilities
---------

Helper utilities for signal processing.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.sigproc.util
