# ezmsg-sigproc

Signal processing primitives for the [ezmsg](https://www.ezmsg.org) message-passing framework.

## Features

* **Filtering** - Chebyshev, comb filters, and more
* **Spectral analysis** - Spectrogram, spectrum, and wavelet transforms
* **Resampling** - Downsample, decimate, and resample operations
* **Windowing** - Sliding windows and buffering utilities
* **Math operations** - Arithmetic, log, abs, difference, and more
* **Signal generation** - Synthetic signal generators

All modules use `AxisArray` as the primary data structure for passing signals between components.

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
