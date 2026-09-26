# ezmsg-sigproc

Signal processing primitives for the [ezmsg](https://www.ezmsg.org) message-passing framework.

## Features

* **Filtering** - Chebyshev, comb filters, and more
* **Spectral analysis** - Spectrogram, spectrum, and wavelet transforms
* **Resampling** - Downsample, decimate, and resample operations
* **Windowing** - Sliding windows and buffering utilities
* **Math operations** - Arithmetic, log, abs, difference, and more
* **Signal generation** - Synthetic signal generators
* More! Brows the API documentation for more details.

All modules use [`AxisArray`](https://www.ezmsg.org/ezmsg/reference/API/axisarray.html) as the primary data structure for passing signals between components. The default data backend is NumPy, but other backends are supported via the Array API such as CuPy and MLX.

## Component discovery

This package registers its Units and Collections in the `ezmsg.components`
entry-point group. Names are qualified by extension (for example,
`sigproc.ButterworthFilter`); values point directly to the defining class.
Generic base classes and processor-only implementations are not registered.

```python
from importlib.metadata import entry_points

components = {ep.name: ep for ep in entry_points(group="ezmsg.components")}
component_type = components["sigproc.ButterworthFilter"].load()
```

Enumerating entry points reads installed package metadata without importing
component modules. Calling `.load()` imports the selected component and may
raise if a runtime dependency is unavailable; it does not instantiate the Unit.
Consumers should retain unavailable entries and report their load errors.
Use the execution environment's Python interpreter to discover its components.

When adding a public component, add its entry point in `pyproject.toml` and
reinstall the package (including editable installs) to refresh the metadata.

## Installation

Install from PyPI:

```bash
pip install ezmsg-sigproc
```

Or install from GitHub for the latest development version:

```bash
pip install git+https://github.com/ezmsg-org/ezmsg-sigproc.git@dev
```

## Documentation

Full documentation is available at [ezmsg.org](https://www.ezmsg.org).

## Development

We use [`uv`](https://docs.astral.sh/uv/) for development.

1. Fork and clone the repository
2. `uv sync` to create a virtual environment and install dependencies
3. `uv run pre-commit install` to set up linting and formatting hooks
4. `uv run pytest tests` to run the test suite
5. Submit a PR against the `dev` branch
