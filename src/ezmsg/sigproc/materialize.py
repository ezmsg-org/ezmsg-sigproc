"""
Materialize (evaluate) lazy array data.

MLX arrays are lazily evaluated — computations are queued but not executed
until the result is needed. This module provides an explicit evaluation point
so that downstream processors receive fully-evaluated data.

Placing the evaluation point is a *graph* decision, not a per-node one: a lazy
backend only pays off while the graph stays lazy, so every forced evaluation
trades throughput for a guarantee. :obj:`MaterializeMode` names the three
positions on that trade so a graph can pick one per site.

.. note::

    Evaluating an **empty** message forces nothing. A branch that emits
    zero-length outputs on some cycles (a binning stage fed sub-bin chunks, say)
    leaves its upstream graph un-evaluated on exactly those cycles, so a
    :obj:`Materialize` placed downstream of the binning is a no-op when it
    matters most. Place it upstream of any stage that can withhold output.
"""

import enum

import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray


class MaterializeMode(str, enum.Enum):
    """How to force a lazy array backend to evaluate."""

    SYNC = "sync"
    """Evaluate and block until the device finishes (``mx.eval``).

    The strongest guarantee and the most expensive: the calling thread stalls
    for a full device round-trip, measured at ~0.1-0.15 ms per call on an M4 Pro
    regardless of array size. Use when the very next thing you do needs the
    values on the host, or when you are timing the work and want the cost
    attributed here rather than to whoever touches the array next."""

    ASYNC = "async"
    """Schedule the evaluation and return immediately (``mx.async_eval``).

    Detaches the computation graph just as :obj:`SYNC` does — so it is equally
    effective at stopping a lazy graph from accumulating across calls — without
    the round-trip stall, leaving the CPU free to build the next message while
    the device works. Prefer it wherever the guarantee you want is "the graph
    does not grow" rather than "the values are on the host now".

    The trade is that it puts no bound on how far the producer may run ahead of
    the device. That is a real risk only for a chain with no downstream
    backpressure and no other evaluation point."""

    OFF = "off"
    """Do nothing; leave the data lazy.

    The fastest option and the correct one when something downstream — a
    conversion back to NumPy, an outlet, a serializer — is already going to
    force evaluation on every cycle. Leaves the node wired as an exact
    pass-through so the mode can be changed without editing the graph."""


def materialize_array(data, mode: MaterializeMode | str = MaterializeMode.SYNC):
    """Apply a :obj:`MaterializeMode` to one array, and return it unchanged.

    A no-op for every backend but MLX: NumPy, CuPy and PyTorch arrays are
    already materialized by the time they are returned, so there is nothing to
    force. Returns ``data`` itself rather than a copy — the evaluation is a side
    effect on the array's pending computation graph, not a new value.

    Args:
        data: The array to evaluate. Non-MLX arrays pass through untouched.
        mode: The :obj:`MaterializeMode` to apply.

    Returns:
        ``data``, unchanged.

    Raises:
        ValueError: If ``mode`` is not a valid :obj:`MaterializeMode`.
    """
    mode = MaterializeMode(mode)
    if mode is MaterializeMode.OFF:
        return data
    try:
        import mlx.core as mx
    except ImportError:
        return data
    if isinstance(data, mx.array):
        if mode is MaterializeMode.ASYNC:
            mx.async_eval(data)
        else:
            mx.eval(data)
    return data


class MaterializeSettings(ez.Settings):
    mode: MaterializeMode = MaterializeMode.SYNC
    """How to force evaluation. Defaults to :obj:`MaterializeMode.SYNC`: this
    node exists only to be an evaluation barrier, so wiring one and getting
    anything weaker would be surprising. Set :obj:`MaterializeMode.ASYNC` to
    bound the graph without stalling, or :obj:`MaterializeMode.OFF` to disable
    the barrier while leaving it wired."""


class MaterializeTransformer(BaseTransformer[MaterializeSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        materialize_array(message.data, self.settings.mode)
        return message


class Materialize(BaseTransformerUnit[MaterializeSettings, AxisArray, AxisArray, MaterializeTransformer]):
    SETTINGS = MaterializeSettings


def materialize(mode: MaterializeMode | str = MaterializeMode.SYNC) -> MaterializeTransformer:
    """Construct a :obj:`MaterializeTransformer`. See :obj:`MaterializeSettings`."""
    return MaterializeTransformer(MaterializeSettings(mode=MaterializeMode(mode)))
