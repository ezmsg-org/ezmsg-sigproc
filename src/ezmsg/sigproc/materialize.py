"""
Materialize (evaluate) lazy array data.

MLX arrays are lazily evaluated — computations are queued but not executed
until the result is needed. This module provides an explicit evaluation point
so that downstream processors receive fully-evaluated data.

Placing the evaluation point is a *graph* decision, not a per-node one: a lazy
backend only pays off while the graph stays lazy, so every forced evaluation
trades throughput for a guarantee. :obj:`MaterializeMode` names the three
positions on that trade so a graph can pick one per site.

Where to place it
-----------------

``mx.async_eval`` never blocks — not on a dependency still in flight (four
chained stages enqueued in 2.4 ms against 71 ms of device work), and not when
the queue is already deep (24 chunks enqueued in 5.7 ms against 201 ms, with no
stall). Which makes it tempting to evaluate after every node. Don't: each
evaluation point costs ~17-21 µs of *host* time to submit a command buffer,
against ~1 µs to build the op graph it replaces — and in a live pipeline that
host time is the executor's, not the device's.

Measured on an M4 Pro across a three-node MLX segment (Metal SOS filter, scaler,
abs), µs per message:

==============  ============  ==========  ==========
chunk           SYNC at tail  ASYNC tail  every node
==============  ============  ==========  ==========
30-64 x 256 ch       300-334     139-172     201-223
128 x 1024 ch        594-641     338-342     328-331
1024 x 2048 ch     1961-2009   1602-1650   1502-1593
==============  ============  ==========  ==========

At streaming chunk sizes, evaluating per node is 1.3-1.7x *worse* than one
evaluation at the tail. It only breaks even once a single node's device work is
large enough to bury the submission cost, which is well above any chunk a
real-time graph carries. Fan-out is not a special case either: evaluating each
branch as it finishes measured 1.3x worse than evaluating once at the join.

What does pay is putting that one evaluation point at the end of the MLX
segment, *before* whatever else the executor has to do — downstream units, or
the NumPy conversion at a process boundary (shared memory cannot carry MLX
arrays, so that conversion forces completion regardless). Against deferring to
that conversion, as a function of how much other work sits in between:

===============  ========  ================
other host work  deferred  ASYNC at the end
===============  ========  ================
none                1.00x             1.00x
~0.5 ms             1.00x        1.30-1.41x
~2 ms               1.00x        1.63-1.69x
~6 ms               1.00x        1.39-1.42x
===============  ========  ================

With nothing else to do the two are identical: the gain is entirely overlap
between device work and host work, so it is worth exactly as much as the host
has to get on with. ``benchmarks/benchmark_mlx_async_eval_placement.py``
reproduces all of the above.

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
    does not grow" rather than "the values are on the host now". Roughly 2x
    :obj:`SYNC` on a streaming chain; see the module docstring for where to put
    it.

    The trade is that it puts no bound on how far the producer may run ahead of
    the device, and the run-ahead is held as in-flight intermediates. Replaying
    400 messages of 512x1024 float32 as fast as the host could build them:
    2083 MiB in flight at 235 ms wall, against 33 MiB at 366 ms for :obj:`SYNC`
    — 1.6x the throughput for 60x the transient memory. A live source that paces
    itself never gets near that; a file replay with no downstream backpressure
    and no other evaluation point can."""

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
