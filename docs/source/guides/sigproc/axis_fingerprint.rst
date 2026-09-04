When a coordinate axis is fingerprinted
========================================

:attr:`~ezmsg.util.messages.axisarray.CoordinateAxis.fingerprint` is a small
hashable stand-in for an axis's *contents*. It is derived from the data rather
than assigned, computed on first access, and cached on the axis object -- so
nothing has to remember to bump it, and the cost is paid once per axis rather
than once per consumer per message.

Transformers that cache anything resolved from coordinate *values* -- channel
labels into array indices (:obj:`~ezmsg.sigproc.slicer.Slicer`), or into output
labels (:obj:`~ezmsg.sigproc.flatten.Flatten`,
:obj:`~ezmsg.sigproc.affinetransform.AffineTransform`) -- fold it into their
state hash. Without it, a source that renames or reorders channels at a fixed
channel count keeps getting the previously resolved answer, and the operation
silently emits one channel's samples under another channel's label.

The diagram below traces a serial graph split across two processes. The middle
unit on the left is a filter that needs the channel *count* to size its state
but never looks at the axis values, so it never triggers a fingerprint.

.. mermaid::

   sequenceDiagram
       autonumber
       box transparent Process A
       participant SRC as Source<br/>builds ch axis
       participant BW as Butterworth<br/>reads shape only
       participant SL as Slicer<br/>rewrites ch axis
       end
       box transparent Process B
       participant RR as CommonRereference<br/>reads ch values
       participant FL as Flatten<br/>reads ch values
       end

       Note over SRC: build ch axis A once, keep as template<br/>A._fingerprint: absent

       rect rgba(128, 128, 128, 0.12)
       Note over SRC,FL: message 1
       SRC->>BW: msg(ch: A)
       Note over BW: _hash_message = (key, sample_shape)<br/>never touches axes, so never reads a fingerprint
       BW->>SL: msg(ch: A) — replace(msg, data=...)<br/>keeps the same axes dict
       Note over SL: _hash_message reads A.fingerprint<br/>COMPUTE, approx 1.06 us
       Note over SRC,SL: A._fingerprint is now cached. A *is* Source's<br/>template object, so the cache lands upstream.
       Note over SL: _reset_state builds B = replace(A, data=A.data[sel])<br/>fast_replace drops _fingerprint, so B is cold
       SL->>RR: serialize, then msg(ch: B)
       Note over RR: B' deserializes fresh and arrives cold<br/>reads B'.fingerprint, COMPUTE approx 1.06 us
       RR->>FL: msg(ch: B') in-process, same object
       Note over FL: reads B'.fingerprint, cached, approx 0.05 us
       end

       rect rgba(128, 128, 128, 0.12)
       Note over SRC,FL: messages 2..N, steady state
       SRC->>BW: msg(ch: A)
       BW->>SL: msg(ch: A)
       Note over SL: A.fingerprint cached, approx 0.05 us<br/>hash unchanged, no reset, re-emits the same B
       SL->>RR: serialize, then msg(ch: B)
       Note over RR: B is still cold in Process A, so every message<br/>deserializes cold and COMPUTEs again
       RR->>FL: msg(ch: B'')
       Note over FL: cached on B'', approx 0.05 us
       end

What the diagram is there to show
---------------------------------

**A downstream read mutates the upstream object.** ``Slicer`` reading
``A.fingerprint`` populates the cache on the very object ``Source`` holds as its
template, because ``replace(msg, data=...)`` passes axes along by reference.
That is the intended sharing: every later consumer of ``A`` in this process gets
the answer for free.

**A newly built axis crosses a process boundary cold.** ``Slicer`` *creates*
``B`` and only ever reads ``A``'s fingerprint, so ``B`` is serialized without
one and each message deserializes cold in Process B -- one digest per message
there, shared between its consumers but not free.

Touching the fingerprint once in whichever unit builds the axis fixes that.
Because the axis is a reused template, every subsequent serialization then
carries the cached value and the downstream process pays nothing::

    def _reset_state(self, message: AxisArray) -> None:
        ...
        self._state.new_axis = replace(message.axes[axis], data=out_data)
        _ = self._state.new_axis.fingerprint  # so it rides the wire precomputed

Forgetting it costs a microsecond, not correctness -- which is the difference
between this and a hand-maintained generation counter.

Is it worth pinning axes across the boundary?
----------------------------------------------

Measured on a 30x256x2 float32 message with a 256-channel ChannelMap axis
(27 kB) and a feature axis:

.. list-table::
   :header-rows: 1
   :widths: 55 15 15 15

   * - per message off the boundary
     - cost
     - vs. the hop
     - core @ 1 kHz
   * - the hop itself (serialize + deserialize)
     - 27.06 us
     - --
     - --
   * - re-fingerprint every message
     - 1.77 us
     - 6.6%
     - 0.18%
   * - sender warms the template (one line, above)
     - 0.31 us
     - 1.1%
     - 0.03%
   * - staging area with pinned template axes
     - 0.97 us
     - 3.6%
     - 0.10%

Re-fingerprinting costs 6.6% of a boundary crossing that already costs 27 us,
so the do-nothing case is affordable. A receive-side staging area that compared
each arriving axis against a pinned template cannot beat the one-line sender
warm, because it still has to *read* the fingerprints in order to compare them;
it only adds back object identity, worth about 0.03 us per consumer.

Why ``fast_replace`` drops the cache
-------------------------------------

``fast_replace`` is ``arr.__class__(**{**arr.__dict__, **kwargs})``. It is
called on *axes*, not just on messages -- ``replace(message.axes[axis],
data=...)`` appears in :mod:`~ezmsg.sigproc.slicer`,
:mod:`~ezmsg.sigproc.affinetransform`,
:mod:`~ezmsg.sigproc.butterworthzerophase` and :mod:`~ezmsg.sigproc.window`.
Once ``_fingerprint`` is in ``__dict__``, that call becomes
``CoordinateAxis(data=..., dims=..., unit=..., _fingerprint=...)`` and raises
``TypeError: unexpected keyword argument '_fingerprint'``.

So the drop is first of all what keeps ``replace()`` working, and only secondly
a correctness measure -- forwarding a digest of the *old* values onto a copy
that changes them would be silently wrong. ``replace()`` on an
:obj:`~ezmsg.util.messages.axisarray.AxisArray` is unaffected, since only
``CoordinateAxis`` ever gains the attribute.
