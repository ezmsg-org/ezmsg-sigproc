import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.util.channels import channel_clusters_from_field


def _msg(banks: list[str], n_data: int | None = None, field: str = "bank") -> AxisArray:
    """AxisArray with a structured ch CoordinateAxis carrying `field`=banks.

    `n_data` overrides the channel-dim size of the data array (to exercise the
    coord/data length-mismatch guard); defaults to len(banks).
    """
    dt = np.dtype([(field, "U2")])
    ch = np.zeros(len(banks), dtype=dt)
    ch[field] = banks
    n = len(banks) if n_data is None else n_data
    return AxisArray(
        data=np.zeros((3, n)),
        dims=["time", "ch"],
        axes={"ch": AxisArray.CoordinateAxis(data=ch, dims=["ch"])},
    )


def test_contiguous_blocks():
    clusters = channel_clusters_from_field(_msg(["A", "A", "B", "B"]), "ch", "bank")
    assert clusters == [[0, 1], [2, 3]]


def test_interleaved_first_appearance_order():
    """Non-contiguous channels group correctly and clusters keep first-seen order."""
    clusters = channel_clusters_from_field(_msg(["B", "A", "B", "A"]), "ch", "bank")
    assert clusters == [[0, 2], [1, 3]]  # 'B' seen first -> first cluster


def test_single_bank():
    clusters = channel_clusters_from_field(_msg(["A", "A", "A"]), "ch", "bank")
    assert clusters == [[0, 1, 2]]


def test_axis_defaults_to_last_dim():
    """axis=None resolves to the last dimension ('ch' here)."""
    clusters = channel_clusters_from_field(_msg(["A", "B"]), None, "bank")
    assert clusters == [[0], [1]]


def test_absent_field_returns_none():
    assert channel_clusters_from_field(_msg(["A", "B"]), "ch", "nonexistent") is None


def test_unstructured_axis_returns_none():
    """A plain (non-structured) coordinate axis has no fields -> None."""
    msg = AxisArray(
        data=np.zeros((3, 2)),
        dims=["time", "ch"],
        axes={"ch": AxisArray.CoordinateAxis(data=np.array(["0", "1"]), dims=["ch"])},
    )
    assert channel_clusters_from_field(msg, "ch", "bank") is None


def test_no_such_axis_returns_none():
    msg = AxisArray(
        data=np.zeros((3, 2)),
        dims=["time", "ch"],
        axes={"ch": AxisArray.CoordinateAxis(data=np.array(["0", "1"]), dims=["ch"])},
    )
    assert channel_clusters_from_field(msg, "missing_axis", "bank") is None


def test_linear_axis_returns_none():
    """A LinearAxis (no `.data`) yields None rather than raising."""
    msg = AxisArray(
        data=np.zeros((3, 2)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=100.0)},
    )
    assert channel_clusters_from_field(msg, "time", "bank") is None


def test_length_mismatch_returns_none():
    """Coord length != channel-dim size is treated as unusable metadata."""
    assert channel_clusters_from_field(_msg(["A", "A"], n_data=4), "ch", "bank") is None
