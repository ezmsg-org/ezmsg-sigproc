"""Data-primed filter start-up: build the filter state from the first samples of a stream."""

import math
import typing

import numpy as np
from array_api_compat import get_namespace
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.messages.util import replace

from ..spectrum import OptionsEnum
from .array import xp_flip

#: Fraction of a pole's initial contribution left when its settling length has passed.
SETTLE_TOL = 1e-3


class StartupMode(OptionsEnum):
    """What a filter assumes about the signal before the first sample it sees."""

    ZERO = "Zero pre-history"
    CONSTANT = "Constant pre-history at the first sample's value"
    FIRST_CHUNK = "Prime the state from a reflection of the first message"
    BUFFERED = "Hold output until enough samples arrive to prime the state, then release them"


def settle_samples(poles: typing.Any, order: int, tol: float = SETTLE_TOL) -> float:
    """Samples until a filter's state is (nearly) independent of its initial conditions.

    ``order`` for a filter with all poles at the origin (an FIR: its state is exactly the last
    ``order`` inputs), otherwise ``ceil(log(tol) / log(max|pole|))``, the length over which the
    slowest pole decays to ``tol``. ``inf`` for a pole on or outside the unit circle.
    """
    radius = float(np.max(np.abs(poles), initial=0.0))
    if radius == 0.0:
        return order
    if radius >= 1.0:
        return math.inf
    return math.ceil(math.log(tol) / math.log(radius))


def _linear_axis(message: AxisArray, axis: str) -> AxisArray.LinearAxis:
    ax = message.axes.get(axis)
    if not isinstance(ax, AxisArray.LinearAxis):
        raise TypeError(f"Data-primed start-up requires a LinearAxis on the '{axis}' dimension.")
    return ax


def startup_n_pad(n: float, message: AxisArray, axis: str, max_dur: float | None) -> int:
    """Cap a pad length ``n`` (samples) at ``max_dur`` seconds of ``message``'s stream (None: no cap)."""
    if max_dur is not None:
        n = min(n, int(max_dur / _linear_axis(message, axis).gain))
    if math.isinf(n):
        raise ValueError(
            "The filter has a pole on or outside the unit circle, so it never settles; "
            "set startup_max_dur to bound the start-up pad."
        )
    return int(n)


def prime_with_reflection(
    run: typing.Callable[[AxisArray], typing.Any], message: AxisArray, axis: str, n_pad: int
) -> None:
    """Prime a stateful processor from the start of ``message`` and discard what it emits.

    Builds up to ``n_pad`` samples of synthetic pre-history by odd reflection of the first samples about
    the first one, ``2 * x[0] - x[n_pad:0:-1]`` (the padding :func:`scipy.signal.filtfilt` uses), stamps
    them with the times just before ``message`` and passes them to ``run``, whose output is dropped.
    ``n_pad`` is cut to what ``message`` holds (one fewer than its length); with nothing to reflect,
    ``run`` is not called.

    Args:
        run: The processor, or any callable that advances its state on an AxisArray.
        message: The first real message. It needs a LinearAxis on ``axis``.
        axis: The stream dimension.
        n_pad: Pad length in samples.
    """
    ax_idx = message.get_axis_idx(axis)
    n_pad = min(n_pad, message.data.shape[ax_idx] - 1)
    if n_pad <= 0:
        return
    ax = _linear_axis(message, axis)
    x0 = slice_along_axis(message.data, slice(0, 1), ax_idx)
    pad = 2 * x0 - xp_flip(slice_along_axis(message.data, slice(1, n_pad + 1), ax_idx), axis=ax_idx)
    run(replace(message, data=pad, axes={**message.axes, axis: replace(ax, offset=ax.offset - n_pad * ax.gain)}))


def hold_for_startup(held: list[AxisArray], message: AxisArray, axis: str, n: int) -> AxisArray | None:
    """Collect messages until they hold ``n`` samples on ``axis``.

    Appends ``message`` to ``held``. Returns None while fewer than ``n`` samples are held; then
    empties ``held`` and returns everything as one message carrying the first held message's axes
    and the latest message's attrs.
    """
    held.append(message)
    ax_idx = message.get_axis_idx(axis)
    if sum(m.data.shape[ax_idx] for m in held) < n:
        return None
    data = get_namespace(message.data).concat([m.data for m in held], axis=ax_idx)
    released = replace(held[0], data=data, attrs=message.attrs)
    held.clear()
    return released
