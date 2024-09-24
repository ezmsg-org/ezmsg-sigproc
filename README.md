# ezmsg.sigproc

Timeseries signal processing implementations for ezmsg

## Dependencies

* `ezmsg`
* `numpy`
* `scipy`
* `pywavelets`

## Installation

### Release

Install the latest release from pypi with: `pip install ezmsg-sigproc` (or `uv add ...` or `poetry add ...`).

### Development Version

You can add the development version of `ezmsg-sigproc` to your project's dependencies in one of several ways.

You can clone it and add its path to your project dependencies. You may wish to do this if you intend to edit `ezmsg-sigproc`. If so, please refer to the [Developers](#developers) section below.

You can also add it directly from GitHub:

* Using `pip`: `pip install git+https://github.com/ezmsg-org/ezmsg-sigproc.git@dev`
* Using `poetry`: `poetry add "git+https://github.com/ezmsg-org/ezmsg-sigproc.git@dev"`
* Using `uv`: `uv add git+https://github.com/ezmsg-org/ezmsg-sigproc --branch dev`

## Developers

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development. It is not strictly required, but if you intend to contribute to ezmsg-sigproc then using `uv` will lead to the smoothest collaboration.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Fork ezmsg-sigproc and clone your fork to your local computer.
3. Open a terminal and `cd` to the cloned folder.
4. `uv sync` to create a .venv and install dependencies.
5. `uv run pre-commit install` to install pre-commit hooks to do linting and formatting.
6. After editing code and making commits, Run the test suite before making a PR: `uv run pytest tests`
