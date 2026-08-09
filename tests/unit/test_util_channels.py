import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.util.channels import (
    channel_groups_from_field,
    group_spec_fields,
    group_spec_fingerprint,
    resolve_channel_groups,
    validate_channel_groups,
)


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


def _multi_msg(banks: list[str], arrays: list[int]) -> AxisArray:
    dt = np.dtype([("bank", "U2"), ("array", "i4")])
    ch = np.zeros(len(banks), dtype=dt)
    ch["bank"] = banks
    ch["array"] = arrays
    return AxisArray(
        data=np.zeros((3, len(banks))),
        dims=["time", "ch"],
        axes={"ch": AxisArray.CoordinateAxis(data=ch, dims=["ch"])},
    )


def _as_lists(groups):
    return None if groups is None else [list(g) for g in groups]


def test_contiguous_blocks():
    assert _as_lists(channel_groups_from_field(_msg(["A", "A", "B", "B"]), "ch", "bank")) == [[0, 1], [2, 3]]


def test_interleaved_first_appearance_order():
    """Non-contiguous channels group correctly and groups keep first-seen order."""
    groups = channel_groups_from_field(_msg(["B", "A", "B", "A"]), "ch", "bank")
    assert _as_lists(groups) == [[0, 2], [1, 3]]  # 'B' seen first -> first group


def test_single_bank():
    assert _as_lists(channel_groups_from_field(_msg(["A", "A", "A"]), "ch", "bank")) == [[0, 1, 2]]


def test_axis_defaults_to_last_dim():
    """axis=None resolves to the last dimension ('ch' here)."""
    assert _as_lists(channel_groups_from_field(_msg(["A", "B"]), None, "bank")) == [[0], [1]]


def test_multiple_fields_group_by_tuple():
    """('array', 'bank') splits banks that repeat across arrays."""
    msg = _multi_msg(["A", "A", "B", "B"], [0, 1, 0, 1])
    assert _as_lists(channel_groups_from_field(msg, "ch", ("array", "bank"))) == [[0], [1], [2], [3]]
    assert _as_lists(channel_groups_from_field(msg, "ch", "bank")) == [[0, 1], [2, 3]]


def test_absent_field_returns_none():
    assert channel_groups_from_field(_msg(["A", "B"]), "ch", "nonexistent") is None
    assert channel_groups_from_field(_multi_msg(["A"], [0]), "ch", ("bank", "nope")) is None


def test_unstructured_axis_returns_none():
    """A plain (non-structured) coordinate axis has no fields -> None."""
    msg = AxisArray(
        data=np.zeros((3, 2)),
        dims=["time", "ch"],
        axes={"ch": AxisArray.CoordinateAxis(data=np.array(["0", "1"]), dims=["ch"])},
    )
    assert channel_groups_from_field(msg, "ch", "bank") is None


def test_no_such_axis_returns_none():
    msg = AxisArray(
        data=np.zeros((3, 2)),
        dims=["time", "ch"],
        axes={"ch": AxisArray.CoordinateAxis(data=np.array(["0", "1"]), dims=["ch"])},
    )
    assert channel_groups_from_field(msg, "missing_axis", "bank") is None


def test_linear_axis_returns_none():
    """A LinearAxis (no `.data`) yields None rather than raising."""
    msg = AxisArray(
        data=np.zeros((3, 2)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=100.0)},
    )
    assert channel_groups_from_field(msg, "time", "bank") is None


def test_length_mismatch_returns_none():
    """Coord length != channel-dim size is treated as unusable metadata."""
    assert channel_groups_from_field(_msg(["A", "A"], n_data=4), "ch", "bank") is None


class TestResolveChannelGroups:
    def test_none_spec(self):
        assert resolve_channel_groups(_msg(["A", "B"]), "ch", None) is None

    def test_explicit_indices(self):
        assert _as_lists(resolve_channel_groups(_msg(["A", "B", "A", "B"]), "ch", [[0, 2], [1, 3]])) == [
            [0, 2],
            [1, 3],
        ]

    def test_field_name(self):
        assert _as_lists(resolve_channel_groups(_msg(["A", "A", "B", "B"]), "ch", "bank")) == [[0, 1], [2, 3]]

    def test_field_sequence(self):
        msg = _multi_msg(["A", "A", "B", "B"], [0, 1, 0, 1])
        assert _as_lists(resolve_channel_groups(msg, "ch", ["array", "bank"])) == [[0], [1], [2], [3]]

    def test_missing_field_resolves_to_none(self):
        """Callers distinguish 'no metadata' from 'one group' and apply their own default."""
        assert resolve_channel_groups(_msg(["A", "B"]), "ch", "region") is None

    def test_callable(self):
        seen = {}

        def spec(message, axis):
            seen["axis"] = axis
            return [[0], [1]]

        assert _as_lists(resolve_channel_groups(_msg(["A", "B"]), "ch", spec)) == [[0], [1]]
        assert seen["axis"] == "ch"

    def test_empty_spec(self):
        assert resolve_channel_groups(_msg(["A", "B"]), "ch", []) == []

    def test_validates(self):
        with pytest.raises(ValueError, match="out-of-range"):
            resolve_channel_groups(_msg(["A", "B"]), "ch", [[0, 5]])
        with pytest.raises(ValueError, match="overlap"):
            resolve_channel_groups(_msg(["A", "B"]), "ch", [[0, 1], [1]])

    def test_mixed_field_and_index_spec_raises(self):
        """The hot-path classifier only looks at the first element, so the full
        homogeneity check lands here -- once, at reset, and loudly."""
        with pytest.raises(ValueError, match="mixes field names with index groups"):
            resolve_channel_groups(_msg(["A", "A", "B", "B"]), "ch", ["bank", [0, 1]])


class TestGroupSpecFields:
    """The classifier that keeps the per-message hash O(1) in spec size."""

    def test_field_specs(self):
        assert group_spec_fields("bank") == ("bank",)
        assert group_spec_fields(["array", "bank"]) == ("array", "bank")
        assert group_spec_fields(np.array(["array", "bank"])) == ("array", "bank")

    def test_specs_that_read_no_metadata(self):
        assert group_spec_fields(None) is None
        assert group_spec_fields([[0, 1], [2, 3]]) is None
        assert group_spec_fields(np.array([[0, 1], [2, 3]])) is None
        assert group_spec_fields([]) is None
        assert group_spec_fields(lambda message, axis: [[0]]) is None

    def test_does_not_scan_the_whole_spec(self):
        """Cost must not grow with the number of index groups: only the first
        element is inspected."""

        class Exploding(list):
            def __getitem__(self, item):
                assert item == 0, "classifier looked past the first element"
                return super().__getitem__(item)

        assert group_spec_fields(Exploding([[0, 1], [2, 3], [4, 5]])) is None


class TestValidateChannelGroups:
    def test_valid_groups_pass(self):
        validate_channel_groups([[0, 1], [2, 3]], 4)

    def test_partial_coverage_passes(self):
        """Omitted channels are the caller's business; only overlap is an error."""
        validate_channel_groups([[0, 1]], 4)

    def test_empty_spec_passes(self):
        validate_channel_groups([], 4)
        validate_channel_groups([[]], 4)
        validate_channel_groups([], 0)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out-of-range"):
            validate_channel_groups([[0, 1, 4]], 4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="out-of-range"):
            validate_channel_groups([[-1, 0]], 4)

    def test_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            validate_channel_groups([[0, 1], [1, 2]], 4)
        with pytest.raises(ValueError, match="overlap"):
            validate_channel_groups([[0, 0]], 4)


class TestGroupSpecFingerprint:
    def test_static_specs_are_empty(self):
        """Nothing about these can change with the message, so they contribute
        nothing to the hash."""
        msg = _msg(["A", "B"])
        assert group_spec_fingerprint(msg, "ch", None) == ()
        assert group_spec_fingerprint(msg, "ch", [[0], [1]]) == ()
        assert group_spec_fingerprint(msg, "ch", lambda message, axis: [[0]]) == ()

    def test_field_presence_is_folded(self):
        present = _msg(["A", "B"])
        absent = AxisArray(data=np.zeros((3, 2)), dims=["time", "ch"])
        assert group_spec_fingerprint(present, "ch", "bank") == (True,)
        assert group_spec_fingerprint(absent, "ch", "bank") == (False,)
        assert group_spec_fingerprint(present, "ch", ["bank", "array"]) == (False,)

    def test_field_values_are_not_folded(self):
        """Only presence, so the hash stays O(1) in channel count."""
        assert group_spec_fingerprint(_msg(["A", "B"]), "ch", "bank") == group_spec_fingerprint(
            _msg(["B", "A"]), "ch", "bank"
        )
