# ezmsg.sigproc

## Overview

ezmsg-sigproc offers timeseries signal‑processing primitives built atop the ezmsg message‑passing framework. Core dependencies include ezmsg, numpy, scipy, pywavelets, and sparse; the project itself is managed through hatchling and uses VCS hooks to populate __version__.py.

## Installation

Install the latest release from pypi with: `pip install ezmsg-sigproc` (or `uv add ...` or `poetry add ...`).

You can install pre-release versions directly from GitHub:

* Using `pip`: `pip install git+https://github.com/ezmsg-org/ezmsg-sigproc.git@dev`
* Using `uv`: `uv add git+https://github.com/ezmsg-org/ezmsg-sigproc --branch dev`
* Using `poetry`: `poetry add "git+https://github.com/ezmsg-org/ezmsg-sigproc.git@dev"`

> See the [Development](#development) section below for installing with the intention of developing.

## Source layout & key modules
* All source resides under src/ezmsg/sigproc, which contains a suite of processors (for example, filter.py, spectrogram.py, spectrum.py, sampler.py) and math and util subpackages.
* The framework’s backbone is base.py, defining standard protocols—Processor, Producer, Consumer, and Transformer—that enable both stateless and stateful processing chains.
* Filtering is implemented in filter.py, providing settings dataclasses and a stateful transformer that applies supplied coefficients to incoming data.
* Spectral analysis uses a composite spectrogram transformer chaining windowing, spectrum computation, and axis adjustments.

## Operating styles: Standalone processors vs. ezmsg pipelines
While each processor is designed to be assembled into an ezmsg pipeline, the components are also well‑suited for offline, ad‑hoc analysis. You can instantiate processors directly in scripts or notebooks for quick prototyping or to validate results from other code. The companion Unit wrappers, however, are meant for assembling processors into a full ezmsg pipeline.

A fully defined ezmsg pipeline shines in online streaming scenarios where message routing, scheduling, and latency handling are crucial. Nevertheless, you can run the same pipeline offline—say, within a Jupyter notebook—if your analysis benefits from ezmsg’s structured execution model. Deciding between a standalone processor and a full pipeline comes down to the trade‑off between simplicity and the operational overhead of the pipeline:

* Standalone processors: Low overhead, ideal for one‑off or exploratory offline tasks.
* Pipeline + Unit wrappers: Additional setup cost but bring concurrency, standardized interfaces, and automatic message flow—useful when your offline experiment mirrors a live system or when you require fine‑grained pipeline behavior.

## Documentation & tests
* `docs/ProcessorsBase.md` details the processor hierarchy and generic type patterns, providing a solid foundation for custom components.
* Unit tests (e.g., `tests/unit/test_sampler.py`) offer concrete examples of usage, showcasing sampler generation, windowing, and message handling.

## Where to learn next
* Study docs/ProcessorsBase.md to master the processor architecture.
* Explore unit tests for hands‑on examples of composing processors and Units.
* Review the ezmsg framework in pyproject.toml to understand the surrounding ecosystem.
* Experiment with the code—try running processors standalone and then integrate them into a small pipeline to observe the trade‑offs firsthand.

This approach equips newcomers to choose the right level of abstraction—raw processor, Unit wrapper, or full pipeline—based on the demands of their analysis or streaming application.

## Development

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development. It is not strictly required, but if you intend to contribute to ezmsg-sigproc then using `uv` will lead to the smoothest collaboration.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Fork ezmsg-sigproc and clone your fork to your local computer.
3. Open a terminal and `cd` to the cloned folder.
4. `uv sync` to create a .venv and install dependencies.
5. `uv run pre-commit install` to install pre-commit hooks to do linting and formatting.
6. Run the test suite before finalizing your edits: `uv run pytest tests`
7. Make a PR against the `dev` branch of the main repo.
