"""Utilities for catching transformers that retain a message's buffer.

Across a cross-process link a subscriber does not own the bytes of the message
it receives. ezmsg serializes with PEP 574 out-of-band buffers
(``pickle.dumps(..., protocol=5, buffer_callback=...)``), so a numpy array
deserializes as a *view* onto the publisher's shared-memory slot rather than as
the owner of a copy, and ``Subscriber.recv_zero_copy`` says so explicitly: the
message "should not be modified or stored beyond the context manager's scope".
The publisher writes into slot ``msg_id % num_buffers`` of a ring and is free to
reuse a slot as soon as the subscriber's context exits.

A transformer that keeps ``message.data`` -- or a *view* of it, such as a tail
slice carried across chunk boundaries -- therefore reads recycled bytes on its
next call. Nothing raises: the array stays a valid object of the right shape and
dtype and simply contains different numbers.

None of this is visible in an ordinary test. In-process publishers pass the
object by reference with no serialization at all, and even a marshalled message
is fine as long as its slot is never overwritten. So these helpers do two
things a normal test does not: they marshal each message through ezmsg's own
``Marshal``, and they compress the publisher's ring to a *single* slot so that
reuse is deterministic on the very next message instead of ``num_buffers``
(default 32) later.

Typical use -- run the same inputs both ways and require the outputs to agree::

    assert_survives_buffer_recycling(
        lambda: DiffTransformer(DiffSettings(axis="time")), messages
    )

Equal-sized messages make the strongest test: the next message lands on exactly
the bytes a retained view points at, so a retained view reads plausible garbage
rather than something obviously wrong.
"""

import contextlib
import typing

import numpy as np
from ezmsg.core.messagemarshal import Marshal
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

SLOT_BYTES = 1 << 22
"""Default slot size. Only has to fit one message; oversizing costs nothing."""


class RecycledSlot:
    """A publisher's shared-memory ring, compressed to a single slot.

    Each :meth:`publish` overwrites the one slot and yields a message whose
    arrays are views onto it, exactly as a subscriber would receive over a
    cross-process link once the real 32-slot ring has wrapped around.
    """

    def __init__(self, nbytes: int = SLOT_BYTES) -> None:
        self._slot = memoryview(bytearray(nbytes))
        self._msg_id = 0

    @contextlib.contextmanager
    def publish(self, msg: AxisArray) -> typing.Iterator[AxisArray]:
        """Serialize ``msg`` into the slot and yield the deserialized view of it."""
        Marshal.to_mem(self._msg_id, msg, self._slot)
        self._msg_id += 1
        with Marshal.obj_from_mem(self._slot) as received:
            yield received


def _detach(result):
    """Snapshot a result so later slot reuse cannot change it out from under us.

    The output of a transformer may itself alias the message it came from, which
    is legitimate -- it is handed straight downstream and not retained -- but it
    means the collected outputs have to be copied before the next publish.
    """
    if result is None:
        return None
    return replace(result, data=np.array(result.data))


def run_recycled(proc, messages: typing.Sequence[AxisArray], *, slot_bytes: int = SLOT_BYTES) -> list:
    """Push ``messages`` through ``proc`` with every one aliasing a reused slot."""
    slot = RecycledSlot(slot_bytes)
    outputs = []
    for msg in messages:
        with slot.publish(msg) as received:
            outputs.append(_detach(proc(received)))
    return outputs


def run_owned(proc, messages: typing.Sequence[AxisArray]) -> list:
    """Push ``messages`` through ``proc`` as ordinary, independently-owned arrays.

    Each message gets a fresh copy of its data, so nothing the transformer keeps
    can ever be invalidated -- the reference behaviour to compare against.
    """
    return [_detach(proc(replace(msg, data=np.array(msg.data)))) for msg in messages]


def assert_survives_buffer_recycling(
    make_proc: typing.Callable[[], typing.Any],
    messages: typing.Sequence[AxisArray],
    *,
    slot_bytes: int = SLOT_BYTES,
) -> list:
    """Assert a transformer's output does not depend on who owns the input bytes.

    Runs the same ``messages`` through two fresh transformers -- one on owned
    arrays, one on arrays aliasing a single recycled slot -- and requires the
    outputs to match exactly. They are the same arithmetic on the same numbers,
    so any difference at all means state was read back from recycled memory.

    :param make_proc: Zero-argument factory; called once per run so the two runs
        do not share state.
    :param messages: Inputs, in order. Equal-sized messages exercise the failure
        hardest (see the module docstring).
    :return: The outputs of the owned run, for further assertions.
    """
    owned = run_owned(make_proc(), messages)
    recycled = run_recycled(make_proc(), messages, slot_bytes=slot_bytes)

    assert len(owned) == len(recycled)
    for ix, (exp, got) in enumerate(zip(owned, recycled)):
        if exp is None or got is None:
            assert exp is got, f"message {ix}: one run returned None and the other did not"
            continue
        assert exp.dims == got.dims, f"message {ix}: dims {got.dims} != {exp.dims}"
        assert exp.data.shape == got.data.shape, f"message {ix}: shape {got.data.shape} != {exp.data.shape}"
        assert np.array_equal(exp.data, got.data), (
            f"message {ix}: output differs when the input buffer is recycled -- "
            f"the transformer is retaining message.data (or a view of it) in its state.\n"
            f"  owned:    {np.asarray(exp.data).ravel()[:8]}\n"
            f"  recycled: {np.asarray(got.data).ravel()[:8]}"
        )
        for name, axis in exp.axes.items():
            assert name in got.axes, f"message {ix}: missing axis {name!r}"
            assert axis == got.axes[name], f"message {ix}: axis {name!r} differs"
    return owned
