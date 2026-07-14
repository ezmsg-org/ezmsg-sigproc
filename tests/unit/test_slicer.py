import copy

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.slicer import SlicerSettings, SlicerTransformer, parse_slice
from tests.helpers.empty_time import check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg
from tests.helpers.util import assert_messages_equal


def test_parse_slice():
    assert parse_slice("") == (slice(None),)
    assert parse_slice(":") == (slice(None),)
    assert parse_slice("NONE") == (slice(None),)
    assert parse_slice("none") == (slice(None),)
    assert parse_slice("0") == (0,)
    assert parse_slice("10") == (10,)
    assert parse_slice(":-1") == (slice(None, -1),)
    assert parse_slice("0:3") == (slice(0, 3),)
    assert parse_slice("::2") == (slice(None, None, 2),)
    assert parse_slice("0,1") == (0, 1)
    assert parse_slice("4:64, 68:100") == (slice(4, 64), slice(68, 100))


def test_slicer_transformer():
    n_times = 13
    n_chans = 255
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(
        in_dat,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.1),
            "ch": AxisArray.CoordinateAxis(data=np.array([f"Ch{_}" for _ in range(n_chans)]), dims=["ch"]),
        },
        key="test_slicer_transformer",
    )
    backup = [copy.deepcopy(msg_in)]

    xformer = SlicerTransformer(SlicerSettings(selection=":2", axis="ch"))
    msg_out = xformer(msg_in)
    assert_messages_equal([msg_in], backup)
    assert msg_out.data.shape == (n_times, 2)
    assert np.array_equal(msg_out.data, in_dat[:, :2])
    assert np.may_share_memory(msg_out.data, in_dat)

    xformer = SlicerTransformer(SlicerSettings(selection="::3", axis="ch"))
    msg_out = xformer(msg_in)
    assert_messages_equal([msg_in], backup)
    assert msg_out.data.shape == (n_times, n_chans // 3)
    assert np.array_equal(msg_out.data, in_dat[:, ::3])
    assert np.may_share_memory(msg_out.data, in_dat)

    xformer = SlicerTransformer(SlicerSettings(selection="4:64", axis="ch"))
    msg_out = xformer(msg_in)
    assert_messages_equal([msg_in], backup)
    assert msg_out.data.shape == (n_times, 60)
    assert np.array_equal(msg_out.data, in_dat[:, 4:64])
    assert np.may_share_memory(msg_out.data, in_dat)

    # Discontiguous slices leads to a copy
    xformer = SlicerTransformer(SlicerSettings(selection="1, 3:5", axis="ch"))
    msg_out = xformer(msg_in)
    assert_messages_equal([msg_in], backup)
    assert np.array_equal(msg_out.data, msg_in.data[:, [1, 3, 4]])
    assert not np.may_share_memory(msg_out.data, in_dat)


def test_slicer_drop_dim():
    n_times = 50
    n_chans = 10
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(
        in_dat,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.1),
            "ch": AxisArray.CoordinateAxis(data=np.array([f"Ch{_}" for _ in range(n_chans)]), dims=["ch"]),
        },
        key="test_slicer_drop_dim",
    )
    backup = [copy.deepcopy(msg_in)]

    xformer = SlicerTransformer(SlicerSettings(selection="5", axis="ch"))
    msg_out = xformer(msg_in)
    assert_messages_equal([msg_in], backup)
    assert msg_out.data.shape == (n_times,)
    assert np.array_equal(msg_out.data, msg_in.data[:, 5])


@pytest.mark.parametrize("selection", [":3", "0, 1, 2", "Ch0, Ch1, Ch2"])
def test_slicer_label(selection: str):
    """
    We use the monkey-patched AxisArray `labels` field that exists in several other ezmsg
    modules that generate data.
    """
    n_times = 50
    n_chans = 10
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(
        in_dat,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.1),
            "ch": AxisArray.CoordinateAxis(data=np.array([f"Ch{_}" for _ in range(n_chans)]), dims=["ch"]),
        },
        key="test_slicer_label",
    )
    backup = [copy.deepcopy(msg_in)]

    xformer = SlicerTransformer(SlicerSettings(selection=selection, axis="ch"))
    msg_out = xformer(msg_in)
    assert_messages_equal([msg_in], backup)
    assert msg_out.data.shape == (n_times, 3)
    assert np.array_equal(msg_out.data, msg_in.data[:, :3])
    assert np.array_equal(msg_out.axes["ch"].data, msg_in.axes["ch"].data[:3])


def test_slicer_empty_along_ch():
    from ezmsg.sigproc.slicer import SlicerSettings, SlicerTransformer

    proc = SlicerTransformer(SlicerSettings(selection="0:2", axis="ch"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    assert result.data.shape[1] == 2


def test_slicer_empty_along_time():
    from ezmsg.sigproc.slicer import SlicerSettings, SlicerTransformer

    proc = SlicerTransformer(SlicerSettings(selection=":", axis="time"))
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)


def test_slicer_empty_first():
    from ezmsg.sigproc.slicer import SlicerSettings, SlicerTransformer

    proc = SlicerTransformer(SlicerSettings(selection="0:2", axis="ch"))
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_parse_slice_regex():
    ax = AxisArray.CoordinateAxis(data=np.array(["Fp1", "Fp2", "C3", "C4", "Cz", "O1", "O2"]), dims=["ch"])
    # Exact match still wins and returns that single index.
    assert parse_slice("C3", axinfo=ax) == (2,)
    # Regex full-match: all central channels.
    assert parse_slice("C[34z]", axinfo=ax) == (2, 3, 4)
    # Prefix pattern.
    assert parse_slice("Fp.*", axinfo=ax) == (0, 1)
    # Comma-separated mix of exact labels and patterns concatenates in order.
    assert parse_slice("O.*, C3", axinfo=ax) == (5, 6, 2)
    # fullmatch semantics: a bare prefix is not a match.
    with pytest.raises(ValueError, match="matched no labels"):
        parse_slice("Fp", axinfo=ax)
    # Without axinfo, non-numeric tokens are still an error (int parse).
    with pytest.raises(ValueError):
        parse_slice("C[34z]")


def _structured_ch_axis(labels: list[str]) -> AxisArray.CoordinateAxis:
    """Build a ChannelMap-style structured coordinate axis (x/y/label/bank fields)."""
    dt = np.dtype([("x", float), ("y", float), ("label", "U8"), ("bank", int)])
    data = np.array([(float(ix % 2), float(ix // 2), lab, ix // 4) for ix, lab in enumerate(labels)], dtype=dt)
    return AxisArray.CoordinateAxis(data=data, dims=["ch"])


def test_parse_slice_structured_axis():
    ax = _structured_ch_axis(["Fp1", "Fp2", "C3", "C4", "Cz", "O1", "O2"])
    # Labels come from the "label" field: exact match, then regex.
    assert parse_slice("C3", axinfo=ax) == (2,)
    assert parse_slice("C[34z]", axinfo=ax) == (2, 3, 4)
    assert parse_slice("O.*, C3", axinfo=ax) == (5, 6, 2)
    # Integer and slice selections must not trip on the structured dtype.
    assert parse_slice("5", axinfo=ax) == (5,)
    assert parse_slice("1, 3:5", axinfo=ax) == (1, slice(3, 5))
    with pytest.raises(ValueError, match="matched no labels"):
        parse_slice("Fp", axinfo=ax)

    # A structured axis without a "label" field supports only integer/slice selections.
    dt = np.dtype([("x", float), ("y", float)])
    ax_nolabel = AxisArray.CoordinateAxis(data=np.zeros(4, dtype=dt), dims=["ch"])
    assert parse_slice("2", axinfo=ax_nolabel) == (2,)
    with pytest.raises(ValueError):
        parse_slice("C3", axinfo=ax_nolabel)


def test_parse_slice_explicit_field():
    # Banks: [0, 0, 0, 0, 1, 1, 1] (see _structured_ch_axis: ix // 4)
    ax = _structured_ch_axis(["Fp1", "Fp2", "C3", "C4", "Cz", "O1", "O2"])
    # Tokens match field values (stringified ints), not positions.
    assert parse_slice("0", axinfo=ax, field="bank") == (0, 1, 2, 3)
    assert parse_slice("1", axinfo=ax, field="bank") == (4, 5, 6)
    assert parse_slice("1, 0", axinfo=ax, field="bank") == (4, 5, 6, 0, 1, 2, 3)
    # With an explicit field there is no positional-int fallback.
    with pytest.raises(ValueError, match="matched no values in field 'bank'"):
        parse_slice("5", axinfo=ax, field="bank")
    # Positional selection is still available via slice syntax.
    assert parse_slice("3:4", axinfo=ax, field="bank") == (slice(3, 4),)
    # Regex matches against the stringified field values.
    assert parse_slice("[01]", axinfo=ax, field="bank") == (0, 1, 2, 3, 4, 5, 6)
    # A missing field errors and names the available fields.
    with pytest.raises(ValueError, match="no field 'elec'.*'bank'"):
        parse_slice("0", axinfo=ax, field="elec")
    # An unstructured axis errors with an explicit field.
    ax_flat = AxisArray.CoordinateAxis(data=np.array(["Ch0", "Ch1"]), dims=["ch"])
    with pytest.raises(ValueError, match="unstructured data"):
        parse_slice("Ch0", axinfo=ax_flat, field="bank")
    # A missing axis errors too.
    with pytest.raises(ValueError, match="no coordinate data"):
        parse_slice("0", axinfo=None, field="bank")


def test_slicer_field_selection():
    n_times = 20
    labels = ["Fp1", "Fp2", "C3", "C4", "Cz", "O1", "O2"]
    n_chans = len(labels)
    in_dat = np.arange(n_times * n_chans, dtype=float).reshape(n_times, n_chans)
    ch_axis = _structured_ch_axis(labels)
    msg_in = AxisArray(
        in_dat,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
            "ch": ch_axis,
        },
        key="test_slicer_field_selection",
    )
    xformer = SlicerTransformer(SlicerSettings(selection="1", axis="ch", field="bank"))
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, in_dat[:, 4:7])
    assert msg_out.axes["ch"].data.dtype == ch_axis.data.dtype
    assert np.array_equal(msg_out.axes["ch"].data, ch_axis.data[4:7])


def test_slicer_structured_axis():
    n_times = 20
    labels = ["Fp1", "Fp2", "C3", "C4", "Cz", "O1", "O2"]
    n_chans = len(labels)
    in_dat = np.arange(n_times * n_chans, dtype=float).reshape(n_times, n_chans)
    ch_axis = _structured_ch_axis(labels)
    msg_in = AxisArray(
        in_dat,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
            "ch": ch_axis,
        },
        key="test_slicer_structured_axis",
    )
    xformer = SlicerTransformer(SlicerSettings(selection="C.*", axis="ch"))
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, in_dat[:, 2:5])
    # The sliced axis keeps its structured records.
    assert msg_out.axes["ch"].data.dtype == ch_axis.data.dtype
    assert np.array_equal(msg_out.axes["ch"].data, ch_axis.data[2:5])


def test_slicer_regex_selection():
    n_times = 20
    labels = np.array(["Fp1", "Fp2", "C3", "C4", "Cz", "O1", "O2"])
    n_chans = len(labels)
    in_dat = np.arange(n_times * n_chans, dtype=float).reshape(n_times, n_chans)
    msg_in = AxisArray(
        in_dat,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
            "ch": AxisArray.CoordinateAxis(data=labels, dims=["ch"]),
        },
        key="test_slicer_regex_selection",
    )
    xformer = SlicerTransformer(SlicerSettings(selection="C.*", axis="ch"))
    msg_out = xformer(msg_in)
    assert np.array_equal(msg_out.data, in_dat[:, 2:5])
    assert np.array_equal(msg_out.axes["ch"].data, labels[2:5])
