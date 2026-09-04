"""Transformers must not read back a message buffer they no longer own.

See :mod:`tests.helpers.recycled_shm` for why this is invisible to every other
test in the suite: in-process links never serialize, so an aliased array and an
owned one behave identically no matter what a transformer retains.
"""

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.baseproc import BaseStatefulTransformer, processor_state
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.messages.util import replace
from frozendict import frozendict

from ezmsg.sigproc.aggregate import AggregationFunction
from ezmsg.sigproc.binned_aggregate import BinnedAggregateSettings, BinnedAggregateTransformer
from ezmsg.sigproc.diff import DiffSettings, DiffTransformer
from tests.helpers.recycled_shm import (
    RecycledSlot,
    assert_survives_buffer_recycling,
)

FS = 100.0
N_CH = 2


def _msgs(blocks: list[np.ndarray], fs: float = FS) -> list[AxisArray]:
    msgs, offset = [], 0.0
    for blk in blocks:
        msgs.append(
            AxisArray(
                data=blk,
                dims=["time", "ch"],
                axes=frozendict(
                    {
                        "time": AxisArray.TimeAxis(fs=fs, offset=offset),
                        "ch": AxisArray.CoordinateAxis(data=np.arange(blk.shape[1]).astype(str), dims=["ch"]),
                    }
                ),
                key="test_buffer_recycling",
            )
        )
        offset += blk.shape[0] / fs
    return msgs


def _equal_blocks(n_msgs: int, n_time: int, seed: int = 0) -> list[np.ndarray]:
    """Equal-sized random blocks: the next message lands on the exact bytes a
    retained view points at, so corruption is silent rather than shape-obvious."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n_time, N_CH)) for _ in range(n_msgs)]


# -- the harness itself ------------------------------------------------------


class _RetainSettings(ez.Settings):
    pass


@processor_state
class _RetainState:
    last: object = None


class _RetainingTransformer(BaseStatefulTransformer[_RetainSettings, AxisArray, AxisArray, _RetainState]):
    """Canary: deliberately does the wrong thing, to prove the harness bites.

    If this ever stops failing, the harness has gone blind and the real tests
    below are passing for the wrong reason.
    """

    def _hash_message(self, message: AxisArray) -> int:
        return hash(message.key)

    def _reset_state(self, message: AxisArray) -> None:
        self._state.last = None

    def _process(self, message: AxisArray) -> AxisArray:
        prev = self._state.last
        # Retain a view of the tail -- the bug this whole module is about.
        self._state.last = slice_along_axis(message.data, slice(-1, None), axis=0)
        if prev is None:
            prev = np.zeros_like(self._state.last)
        return replace(message, data=np.concatenate((prev, message.data), axis=0)[:-1])


def test_harness_detects_a_retained_view():
    msgs = _msgs(_equal_blocks(4, 12, seed=7))
    with pytest.raises(AssertionError, match="retaining message.data"):
        assert_survives_buffer_recycling(lambda: _RetainingTransformer(_RetainSettings()), msgs)


def test_recycled_slot_actually_recycles():
    """The slot must alias, and be overwritten by the next publish."""
    msgs = _msgs(_equal_blocks(2, 8, seed=3))
    slot = RecycledSlot()
    with slot.publish(msgs[0]) as first:
        assert first.data.base is not None, "received array should be a view, not an owner"
        retained = first.data
        assert np.array_equal(retained, msgs[0].data)
    with slot.publish(msgs[1]) as _second:
        assert not np.array_equal(retained, msgs[0].data), "slot was not reused; harness would not catch anything"


# -- diff --------------------------------------------------------------------


@pytest.mark.parametrize("scale_by_fs", [False, True])
def test_diff_does_not_retain_message_data(scale_by_fs):
    """`last_dat` is the previous message's final sample, used on the next call."""
    msgs = _msgs(_equal_blocks(4, 4, seed=0))
    assert_survives_buffer_recycling(
        lambda: DiffTransformer(DiffSettings(axis="time", scale_by_fs=scale_by_fs)),
        msgs,
    )


def test_diff_boundary_sample_is_correct_across_recycling():
    """Pin the actual value of the cross-message diff, not just self-consistency."""
    blocks = _equal_blocks(2, 4, seed=0)
    msgs = _msgs(blocks)
    proc = DiffTransformer(DiffSettings(axis="time"))
    slot = RecycledSlot()
    outs = []
    for msg in msgs:
        with slot.publish(msg) as received:
            outs.append(np.array(proc(received).data))
    got = np.concatenate(outs, axis=0)

    a, b = blocks
    expected_boundary = b[0] - a[-1]
    assert np.allclose(got[a.shape[0]], expected_boundary)


# -- binned_aggregate --------------------------------------------------------


def test_binned_aggregate_no_bin_completed_does_not_retain():
    """`carry is None` and no bin completes: the whole message became the carry."""
    rng = np.random.default_rng(0)
    # 0.1 s bins at 100 Hz = 10 samples; a 4-sample chunk completes no bin.
    blocks = [rng.standard_normal((4, N_CH)), rng.standard_normal((8, N_CH)), rng.standard_normal((8, N_CH))]
    assert_survives_buffer_recycling(
        lambda: BinnedAggregateTransformer(BinnedAggregateSettings(axis="time", bin_duration=0.1)),
        _msgs(blocks),
    )


def test_binned_aggregate_leftover_tail_does_not_retain():
    """A bin completes with no prior carry, so the leftover tail was a view of
    `message.data`. Equal-size chunks put the next message on those bytes."""
    assert_survives_buffer_recycling(
        lambda: BinnedAggregateTransformer(BinnedAggregateSettings(axis="time", bin_duration=0.1)),
        _msgs(_equal_blocks(4, 12, seed=1)),
    )


@pytest.mark.parametrize("fractional", [True, False])
def test_binned_aggregate_fractional_grid_does_not_retain(fractional):
    """An off-nominal rate makes bin lengths vary, so the carry length varies too.

    Blocks are longer than a bin (101.3 samples at 1013 Hz) so the very first
    message closes a bin with no prior carry -- the branch that leaves the carry
    as a view of the message.
    """
    assert_survives_buffer_recycling(
        lambda: BinnedAggregateTransformer(
            BinnedAggregateSettings(axis="time", bin_duration=0.1, fractional=fractional)
        ),
        _msgs(_equal_blocks(6, 128, seed=2), fs=1013.0),
    )


def test_binned_aggregate_multi_op_does_not_retain():
    """As above, through the stacked-operation path.

    MIN/MAX only differ if the substituted samples are the extreme of their bin,
    so the seed is one where they are -- corruption of a two-sample carry is
    otherwise easy to average away.
    """
    assert_survives_buffer_recycling(
        lambda: BinnedAggregateTransformer(
            BinnedAggregateSettings(
                axis="time",
                bin_duration=0.1,
                operation=(AggregationFunction.MIN, AggregationFunction.MAX),
            )
        ),
        _msgs(_equal_blocks(4, 12, seed=0)),
    )


def _axis_equality_is_content_based() -> bool:
    """Whether the installed ezmsg compares coordinate axes by value.

    Before ezmsg-org/ezmsg#258's stack, ``CoordinateAxis.__eq__`` resolved
    through the MRO to ``AxisBase.__eq__``, which compares only ``unit`` -- so
    two axes with different labels compared equal and this suite could not see a
    retained coordinate axis at all.
    """
    ch = AxisArray.CoordinateAxis
    return ch(data=np.array(["A"]), dims=["ch"]) != ch(data=np.array(["B"]), dims=["ch"])


requires_content_axis_equality = pytest.mark.skipif(
    not _axis_equality_is_content_based(),
    reason="ezmsg's CoordinateAxis.__eq__ compares only `unit`; axis retention is undetectable",
)


class _RetainsChAxis:
    """Caches the first message's ch axis and re-emits it -- a view into the slot."""

    def __init__(self) -> None:
        self.cached = None

    def __call__(self, msg: AxisArray) -> AxisArray:
        if self.cached is None:
            self.cached = msg.axes["ch"]
        return replace(msg, axes={**msg.axes, "ch": self.cached})


class _CopiesChAxis(_RetainsChAxis):
    """The same thing done correctly: the cached axis owns its memory."""

    def __call__(self, msg: AxisArray) -> AxisArray:
        if self.cached is None:
            axis = msg.axes["ch"]
            self.cached = replace(axis, data=np.array(axis.data))
        return replace(msg, axes={**msg.axes, "ch": self.cached})


def _msgs_with_distinct_ch_labels(n: int = 3, n_time: int = 8) -> list[AxisArray]:
    """Messages whose ch labels differ, so a stale axis view is observable.

    With identical labels on every message a retained view reads the *same*
    bytes back out of the recycled slot and the corruption is invisible.
    """
    out, offset = [], 0.0
    for i in range(n):
        out.append(
            AxisArray(
                data=np.random.default_rng(i).standard_normal((n_time, N_CH)),
                dims=["time", "ch"],
                axes=frozendict(
                    {
                        "time": AxisArray.TimeAxis(fs=FS, offset=offset),
                        "ch": AxisArray.CoordinateAxis(data=np.array([f"m{i}c{c}" for c in range(N_CH)]), dims=["ch"]),
                    }
                ),
                key="test_buffer_recycling",
            )
        )
        offset += n_time / FS
    return out


@requires_content_axis_equality
def test_harness_detects_a_retained_coordinate_axis():
    """The suite's own check: a retained *axis* must fail, not just retained data.

    Coordinate axes are views onto the transport slot exactly as the samples
    are, so a transformer that caches one across calls reads recycled bytes.
    Without this, a helper that forgot to compare axes would leave every
    transformer here untested for axis retention and nothing would say so.
    """
    with pytest.raises(AssertionError, match="axis 'ch' differs"):
        assert_survives_buffer_recycling(_RetainsChAxis, _msgs_with_distinct_ch_labels())


@requires_content_axis_equality
def test_harness_accepts_a_copied_coordinate_axis():
    """...and does not cry wolf when the transformer copies it properly."""
    assert_survives_buffer_recycling(_CopiesChAxis, _msgs_with_distinct_ch_labels())
