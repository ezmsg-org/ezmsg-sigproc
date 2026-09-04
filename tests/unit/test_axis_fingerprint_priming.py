"""A transformer that builds a coordinate axis should hand it over ready to use.

``CoordinateAxis.fingerprint`` is what every stateful consumer keys its state on.
It is computed on first access and cached on the instance, and the cache pickles
with the axis -- so whoever touches it first pays, and everyone after gets it
free.

In-process that first toucher is usually the next stateful node, which primes the
axis as a side effect of hashing it. The gap is the boundary: unpickling builds a
*new* axis object per message, so an axis that left its producing process cold is
re-checksummed by the first consumer in every receiving process, on every
message, forever. :func:`~ezmsg.sigproc.util.message.with_fingerprint` closes
that by priming at the point of construction, and these tests pin it, since
nothing else would notice it stopping.

Deliberately not primed: coordinate axes along the chunk dimension. Their data is
new every message and no consumer reads their fingerprint -- the default hash
takes only ``gain`` from the chunk axis.
"""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.sigproc.affinetransform import AffineTransformSettings, AffineTransformTransformer
from ezmsg.sigproc.aggregate import (
    AggregationFunction,
    RangedAggregateSettings,
    RangedAggregateTransformer,
)
from ezmsg.sigproc.binned_aggregate import BinnedAggregateSettings, BinnedAggregateTransformer
from ezmsg.sigproc.concat import ConcatProcessor, ConcatSettings
from ezmsg.sigproc.coordinatespaces import (
    CoordinateMode,
    CoordinateSpacesSettings,
    CoordinateSpacesTransformer,
)
from ezmsg.sigproc.flatten import FlattenSettings, FlattenTransformer
from ezmsg.sigproc.slicer import SlicerSettings, SlicerTransformer
from ezmsg.sigproc.wavelets import CWTSettings, CWTTransformer


def signal(n_time: int = 64, labels: list[str] | None = None, fs: float = 100.0, key: str = "dev") -> AxisArray:
    labels = labels or [f"c{i}" for i in range(4)]
    return AxisArray(
        np.random.default_rng(0).standard_normal((n_time, len(labels))).astype(np.float32),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key=key,
        chunk_dim="time",
    )


def spectrum(n_win: int = 8, n_freq: int = 16, n_ch: int = 4) -> AxisArray:
    return AxisArray(
        np.random.default_rng(0).standard_normal((n_win, n_freq, n_ch)).astype(np.float32),
        dims=["win", "freq", "ch"],
        axes={
            "win": AxisArray.TimeAxis(fs=10.0),
            "freq": CoordinateAxis(data=np.arange(n_freq, dtype=float), dims=["freq"], unit="Hz"),
            "ch": CoordinateAxis(data=np.array([f"c{i}" for i in range(n_ch)]), dims=["ch"]),
        },
        key="dev",
        chunk_dim="win",
    )


def freshly_built(source: AxisArray, result: AxisArray) -> dict[str, CoordinateAxis]:
    """The coordinate axes on *result* that are not objects *source* handed in.

    Identity, not equality: an axis that merely passed through was primed by
    whichever consumer hashed it, which would mask a producer that never primes
    anything.
    """
    incoming = {id(axis) for axis in source.axes.values()}
    return {
        dim: axis for dim, axis in result.axes.items() if isinstance(axis, CoordinateAxis) and id(axis) not in incoming
    }


def assert_primed(source: AxisArray, result: AxisArray, expected: set[str]) -> None:
    built = freshly_built(source, result)
    assert set(built) >= expected, f"expected new axes {expected}, got {set(built)}"
    cold = sorted(dim for dim, axis in built.items() if "_fingerprint" not in axis.__dict__)
    assert not cold, f"axes handed downstream without a fingerprint: {cold}"


class TestCreatedAxesArePrimed:
    def test_affine_transform_output_labels(self):
        msg = signal()
        proc = AffineTransformTransformer(AffineTransformSettings(weights=np.ones((4, 2)), axis="ch"))
        assert_primed(msg, proc(msg), {"ch"})

    def test_slicer_selection(self):
        msg = signal()
        proc = SlicerTransformer(SlicerSettings(selection="0:2", axis="ch"))
        assert_primed(msg, proc(msg), {"ch"})

    def test_flatten_merged_axis(self):
        msg = signal()
        proc = FlattenTransformer(FlattenSettings(preserve_axis="time", sample_axis="time", flatten_axes=("ch",)))
        assert_primed(msg, proc(msg), {"ch"})

    def test_ranged_aggregate_band_axis(self):
        msg = spectrum()
        proc = RangedAggregateTransformer(
            RangedAggregateSettings(axis="freq", bands=[(0.0, 4.0), (8.0, 12.0)], operation=AggregationFunction.MEAN)
        )
        assert_primed(msg, proc(msg), {"freq"})

    def test_binned_aggregate_metric_axis(self):
        msg = signal()
        proc = BinnedAggregateTransformer(
            BinnedAggregateSettings(
                axis="time",
                bin_duration=0.1,
                operation=(AggregationFunction.MEAN, AggregationFunction.MAX),
                newaxis="metric",
            )
        )
        assert_primed(msg, proc(msg), {"metric"})

    def test_coordinate_spaces_relabel(self):
        msg = signal(labels=["x", "y"])
        proc = CoordinateSpacesTransformer(CoordinateSpacesSettings(axis="ch", mode=CoordinateMode.CART2POL))
        assert_primed(msg, proc(msg), {"ch"})

    def test_wavelet_frequency_axis(self):
        msg = signal(n_time=128)
        proc = CWTTransformer(CWTSettings(frequencies=(8.0, 12.0, 20.0), wavelet="morl", axis="time"))
        assert_primed(msg, proc(msg), {"freq"})

    def test_concat_merged_axis(self):
        a = signal(labels=["a0", "a1"], key="A")
        b = signal(labels=["b0", "b1"], key="B")
        proc = ConcatProcessor(ConcatSettings(axis="ch"))
        assert_primed(a, proc._concat(a, b), {"ch"})


class TestTheChunkAxisIsLeftAlone:
    """Priming a per-message chunk coordinate would be pure cost: its data is
    new every message and the default hash reads only ``gain`` from it."""

    def test_an_irregular_time_axis_is_not_primed(self):
        n_time = 32
        msg = AxisArray(
            np.zeros((n_time, 2), np.float32),
            dims=["time", "ch"],
            axes={
                "time": CoordinateAxis(data=np.arange(n_time, dtype=float), dims=["time"], unit="s"),
                "ch": CoordinateAxis(data=np.array(["a", "b"]), dims=["ch"]),
            },
            key="dev",
            chunk_dim="time",
        )
        SlicerTransformer(SlicerSettings(selection="0:1", axis="ch"))(msg)
        assert "_fingerprint" not in msg.axes["time"].__dict__


def test_priming_is_idempotent_and_returns_the_axis():
    from ezmsg.sigproc.util.message import with_fingerprint

    axis = CoordinateAxis(data=np.array(["a", "b"]), dims=["ch"])
    assert with_fingerprint(axis) is axis
    first = axis.__dict__["_fingerprint"]
    assert with_fingerprint(axis).__dict__["_fingerprint"] is first


@pytest.mark.parametrize("dtype", ["U8", "f8", "i4"])
def test_priming_survives_a_pickle_round_trip(dtype):
    """The whole point: the far side gets the answer without recomputing."""
    import pickle

    from ezmsg.sigproc.util.message import with_fingerprint

    axis = with_fingerprint(CoordinateAxis(data=np.arange(8).astype(dtype), dims=["ch"]))
    landed = pickle.loads(pickle.dumps(axis))
    assert "_fingerprint" in landed.__dict__
    assert landed.__dict__["_fingerprint"] == axis.fingerprint
