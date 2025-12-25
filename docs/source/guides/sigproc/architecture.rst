Architecture & Usage Patterns
=============================

This page describes the architecture of ezmsg-sigproc and different ways to use its components.

Operating Styles: Standalone vs. Pipelines
------------------------------------------

While each processor is designed to be assembled into an ezmsg pipeline, the components are also well-suited for offline, ad-hoc analysis. You can instantiate processors directly in scripts or notebooks for quick prototyping or to validate results from other code. The companion Unit wrappers, however, are meant for assembling processors into a full ezmsg pipeline.

A fully defined ezmsg pipeline shines in online streaming scenarios where message routing, scheduling, and latency handling are crucial. Nevertheless, you can run the same pipeline offline—say, within a Jupyter notebook—if your analysis benefits from ezmsg's structured execution model.

Deciding between a standalone processor and a full pipeline comes down to the trade-off between simplicity and the operational overhead of the pipeline:

* **Standalone processors**: Low overhead, ideal for one-off or exploratory offline tasks.
* **Pipeline + Unit wrappers**: Additional setup cost but bring concurrency, standardized interfaces, and automatic message flow—useful when your offline experiment mirrors a live system or when you require fine-grained pipeline behavior.

Source Layout
-------------

All source resides under ``src/ezmsg/sigproc``, which contains:

* A suite of processors (for example, ``filter.py``, ``spectrogram.py``, ``spectrum.py``, ``sampler.py``)
* ``math/`` and ``util/`` subpackages for mathematical operations and utilities

Key Modules
^^^^^^^^^^^

* **base.py** (via ``ezmsg.baseproc``): Defines standard protocols—Processor, Producer, Consumer, and Transformer—that enable both stateless and stateful processing chains.
* **filter.py**: Provides settings dataclasses and a stateful transformer that applies supplied coefficients to incoming data.
* **spectrogram.py**: Implements spectral analysis using a composite transformer chaining windowing, spectrum computation, and axis adjustments.

Where to Learn Next
-------------------

* Study the :doc:`base` page to master the processor architecture.
* Explore unit tests in the repository for hands-on examples of composing processors and Units.
* Review the `ezmsg framework <https://www.ezmsg.org>`_ to understand the surrounding ecosystem.
* Experiment with the code—try running processors standalone and then integrate them into a small pipeline to observe the trade-offs firsthand.

This approach equips newcomers to choose the right level of abstraction—raw processor, Unit wrapper, or full pipeline—based on the demands of their analysis or streaming application.
