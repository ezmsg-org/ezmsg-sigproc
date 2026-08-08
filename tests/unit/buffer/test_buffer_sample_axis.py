"""Layout equivalence for :class:`HybridBuffer`'s ``sample_axis``.

The axis-0 buffer is the reference implementation — it is pinned down by the
suites in ``test_buffer.py`` / ``test_buffer_overflow.py``. These tests drive a
non-zero-``sample_axis`` buffer through the *same* operation sequences and assert
every observable comes back as the transpose of the axis-0 result. That covers
the whole state machine (deque flush, ring wraparound, all four overflow
strategies, grow, negative seek) rather than a handful of hand-picked shapes.
"""

import numpy as np
import pytest

from ezmsg.sigproc.util.buffer import HybridBuffer

N_CH = 3


def _block(n_samples: int, start: float, sample_axis: int) -> np.ndarray:
    """A deterministic (n_samples, N_CH) block, laid out for ``sample_axis``."""
    flat = np.arange(start, start + n_samples * N_CH, dtype=np.float32).reshape(n_samples, N_CH)
    return flat if sample_axis == 0 else np.ascontiguousarray(flat.T)


def _as_time_first(arr: np.ndarray, sample_axis: int) -> np.ndarray:
    return arr if sample_axis == 0 else arr.T


def _make(sample_axis: int, **overrides) -> HybridBuffer:
    params = {
        "array_namespace": np,
        "capacity": 20,
        "other_shape": (N_CH,),
        "dtype": np.float32,
        "sample_axis": sample_axis,
        "update_strategy": "on_demand",
        "overflow_strategy": "grow",
    }
    params.update(overrides)
    return HybridBuffer(**params)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample_axis, other_shape, expected",
    [
        (0, (3,), (20, 3)),
        (1, (3,), (3, 20)),
        (-1, (3,), (3, 20)),  # negative index normalizes to the last position
        (0, (), (20,)),
        (1, (4, 5), (4, 20, 5)),  # sample axis in the middle of a 3-D layout
        (2, (4, 5), (4, 5, 20)),
    ],
)
def test_ring_allocated_in_requested_layout(sample_axis, other_shape, expected):
    buf = HybridBuffer(
        array_namespace=np,
        capacity=20,
        other_shape=other_shape,
        dtype=np.float32,
        sample_axis=sample_axis,
    )
    assert buf._buffer.shape == expected
    assert buf.sample_axis == expected.index(20)


def test_write_rejects_mismatched_other_shape():
    buf = _make(1)
    # (n, N_CH) is the *wrong* layout for a sample_axis=1 buffer.
    with pytest.raises(ValueError):
        buf.write(np.zeros((5, N_CH), dtype=np.float32))


def test_1d_convenience_expands_on_the_non_sample_axis():
    """A single-channel buffer accepts a bare 1-D block in either layout."""
    for sample_axis, expected in [(0, (5, 1)), (1, (1, 5))]:
        buf = HybridBuffer(
            array_namespace=np,
            capacity=10,
            other_shape=(1,),
            dtype=np.float32,
            sample_axis=sample_axis,
            update_strategy="immediate",
        )
        buf.write(np.arange(5, dtype=np.float32))
        assert buf.available() == 5
        retrieved = buf.read(5)
        assert retrieved.shape == expected
        np.testing.assert_array_equal(retrieved.squeeze(), np.arange(5, dtype=np.float32))


# ---------------------------------------------------------------------------
# Differential equivalence against the axis-0 reference
# ---------------------------------------------------------------------------


def _run_script(sample_axis: int, script, **buf_kwargs):
    """Apply ``script`` to a buffer and return every observable it produced."""
    buf = _make(sample_axis, **buf_kwargs)
    out = []
    for op, arg in script:
        if op == "write":
            buf.write(_block(arg, start=len(out) * 100, sample_axis=sample_axis))
            out.append(("state", buf.available(), buf.tell(), buf.capacity))
        elif op == "read":
            out.append(("read", _as_time_first(buf.read(arg), sample_axis)))
        elif op == "peek":
            out.append(("peek", _as_time_first(buf.peek(arg), sample_axis)))
        elif op == "seek":
            out.append(("seek", buf.seek(arg)))
        elif op == "peek_at":
            out.append(("peek_at", _as_time_first(buf.peek_at(arg), sample_axis)))
        elif op == "peek_last":
            out.append(("peek_last", _as_time_first(buf.peek_last(), sample_axis)))
        else:  # pragma: no cover - guards against a typo in a script
            raise AssertionError(f"unknown op {op}")
    return out


def _assert_same(ref, got):
    assert len(ref) == len(got)
    for i, (r, g) in enumerate(zip(ref, got)):
        assert r[0] == g[0], f"op {i}: kind {g[0]} != {r[0]}"
        if isinstance(r[1], np.ndarray):
            np.testing.assert_array_equal(g[1], r[1], err_msg=f"op {i} ({r[0]})")
        else:
            assert r[1:] == g[1:], f"op {i} ({r[0]}): {g[1:]} != {r[1:]}"


_SCRIPTS = {
    # Plain FIFO: fill, drain, refill.
    "simple": [("write", 10), ("read", 4), ("peek", 3), ("read", 6), ("write", 5), ("read", 5)],
    # Force the ring to wrap: repeated write/read past capacity=20.
    "wraparound": [op for _ in range(6) for op in (("write", 7), ("read", 7))],
    # Discontiguous peek straddling the wrap point.
    "wrapped_peek": [
        ("write", 18),
        ("read", 15),
        ("write", 12),
        ("peek", 15),
        ("read", 15),
    ],
    # Exercise the deque: several writes before any read triggers one flush.
    "deque_flush": [("write", 3), ("write", 4), ("write", 5), ("peek", 12), ("read", 12)],
    # Rewind into already-read data, then re-read it.
    "negative_seek": [("write", 16), ("read", 12), ("seek", -8), ("peek", 10), ("read", 10)],
    # Non-advancing accessors that may bypass the flush.
    "peek_helpers": [
        ("write", 6),
        ("peek_at", 0),
        ("peek_at", 5),
        ("peek_last", None),
        ("write", 4),
        ("peek_last", None),
    ],
}


@pytest.mark.parametrize("script_name", sorted(_SCRIPTS))
@pytest.mark.parametrize("update_strategy", ["immediate", "on_demand"])
@pytest.mark.parametrize("sample_axis", [1, -1])
def test_matches_axis0_reference(script_name, update_strategy, sample_axis):
    script = _SCRIPTS[script_name]
    ref = _run_script(0, script, update_strategy=update_strategy)
    got = _run_script(sample_axis, script, update_strategy=update_strategy)
    _assert_same(ref, got)


@pytest.mark.parametrize("overflow_strategy", ["grow", "drop", "warn-overwrite"])
@pytest.mark.parametrize("sample_axis", [1])
def test_overflow_matches_axis0_reference(overflow_strategy, sample_axis):
    """Overflow past capacity=20 must evict/grow identically in either layout."""
    script = [("write", 12), ("write", 15), ("peek", None), ("read", None), ("write", 30), ("read", None)]
    kwargs = {"overflow_strategy": overflow_strategy, "update_strategy": "on_demand"}
    with pytest.warns(RuntimeWarning) if overflow_strategy == "warn-overwrite" else _noop():
        ref = _run_script(0, script, **kwargs)
    with pytest.warns(RuntimeWarning) if overflow_strategy == "warn-overwrite" else _noop():
        got = _run_script(sample_axis, script, **kwargs)
    _assert_same(ref, got)


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_overflow_raise_matches_axis0_reference():
    """A deque write beyond capacity raises eagerly, in either layout."""
    for sample_axis in (0, 1):
        buf = _make(sample_axis, overflow_strategy="raise", update_strategy="on_demand")
        with pytest.raises(OverflowError):
            buf.write(_block(21, start=0, sample_axis=sample_axis))

        # And a flush that would overflow raises too, even when no single
        # write exceeds capacity on its own.
        buf = _make(sample_axis, overflow_strategy="raise", update_strategy="on_demand")
        buf.write(_block(15, start=0, sample_axis=sample_axis))
        buf.read(15)  # leaves head mid-ring with 15 samples of read-behind
        buf.write(_block(12, start=100, sample_axis=sample_axis))
        with pytest.raises(OverflowError):
            buf.write(_block(12, start=200, sample_axis=sample_axis))


def test_grow_preserves_read_and_unread_in_either_layout():
    """`_grow_buffer` copies both the read-behind and unread spans; check both."""
    script = [("write", 15), ("read", 10), ("write", 25), ("seek", -10), ("read", None)]
    ref = _run_script(0, script)
    got = _run_script(1, script)
    _assert_same(ref, got)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_randomized_ops_match_axis0_reference(seed):
    """Random op sequences reach state combinations the scripted cases miss."""
    rng = np.random.default_rng(seed)
    script = []
    pending = 0  # samples the buffer will hold when this op executes
    read_behind = 0
    for _ in range(40):
        choice = rng.integers(0, 4)
        if choice == 0 or pending == 0:
            n = int(rng.integers(1, 9))
            script.append(("write", n))
            pending += n
        elif choice == 1:
            n = int(rng.integers(1, pending + 1))
            script.append(("read", n))
            pending -= n
            read_behind = min(read_behind + n, 20)
        elif choice == 2:
            script.append(("peek", int(rng.integers(1, pending + 1))))
        else:
            n = int(rng.integers(1, pending + 1))
            script.append(("seek", n))
            pending -= n
            read_behind = min(read_behind + n, 20)

    ref = _run_script(0, script)
    got = _run_script(1, script)
    _assert_same(ref, got)
