"""Unit tests for ezmsg.sigproc.concat module."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from frozendict import frozendict

from ezmsg.sigproc.concat import (
    ConcatProcessor,
    ConcatSettings,
    _build_merged_coordinate_axis,
    _validate_shared_axes,
)


def _make_msg(
    data: np.ndarray,
    fs: float = 100.0,
    offset: float = 0.0,
    ch_labels: list[str] | None = None,
    ch_axis: CoordinateAxis | None = None,
    key: str = "",
) -> AxisArray:
    """Helper to build a (time, ch) AxisArray."""
    n_ch = data.shape[1] if data.ndim > 1 else 1
    if data.ndim == 1:
        data = data[:, None]
    if ch_axis is None:
        if ch_labels is None:
            ch_labels = [f"Ch{i}" for i in range(n_ch)]
        ch_axis = CoordinateAxis(data=np.array(ch_labels), dims=["ch"], unit="label")
    time_axis = AxisArray.TimeAxis(fs=fs, offset=offset)
    return AxisArray(
        data,
        dims=["time", "ch"],
        axes=frozendict({"time": time_axis, "ch": ch_axis}),
        key=key,
    )


# ---------------------------------------------------------------------------
# Shared helper tests
# ---------------------------------------------------------------------------


class TestBuildMergedCoordinateAxis:
    def test_simple_labels_relabel(self):
        ax_a = CoordinateAxis(data=np.array(["X", "Y"]), dims=["ch"])
        ax_b = CoordinateAxis(data=np.array(["X", "Y", "Z"]), dims=["ch"])
        merged = _build_merged_coordinate_axis(ax_a, ax_b, relabel=True, label_a="_a", label_b="_b")
        assert list(merged.data) == ["X_a", "Y_a", "X_b", "Y_b", "Z_b"]

    def test_simple_labels_no_relabel(self):
        ax_a = CoordinateAxis(data=np.array(["A0", "A1"]), dims=["ch"])
        ax_b = CoordinateAxis(data=np.array(["B0", "B1"]), dims=["ch"])
        merged = _build_merged_coordinate_axis(ax_a, ax_b, relabel=False, label_a="", label_b="")
        assert list(merged.data) == ["A0", "A1", "B0", "B1"]

    def test_struct_preserves_fields(self):
        dt = np.dtype([("label", "U8"), ("x", "f4"), ("y", "f4")])
        data_a = np.array([("ch0", 1.0, 2.0), ("ch1", 3.0, 4.0)], dtype=dt)
        data_b = np.array([("ch2", 5.0, 6.0)], dtype=dt)
        ax_a = CoordinateAxis(data=data_a, dims=["ch"])
        ax_b = CoordinateAxis(data=data_b, dims=["ch"])

        merged = _build_merged_coordinate_axis(ax_a, ax_b, relabel=True, label_a="_L", label_b="_R")
        assert len(merged.data) == 3
        assert merged.data[0]["label"] == "ch0_L"
        assert merged.data[2]["label"] == "ch2_R"
        # Non-label fields preserved.
        np.testing.assert_allclose(merged.data[0]["x"], 1.0)
        np.testing.assert_allclose(merged.data[2]["y"], 6.0)

    def test_struct_no_relabel(self):
        dt = np.dtype([("label", "U8"), ("x", "f4")])
        data_a = np.array([("ch0", 1.0)], dtype=dt)
        data_b = np.array([("ch1", 2.0)], dtype=dt)
        ax_a = CoordinateAxis(data=data_a, dims=["ch"])
        ax_b = CoordinateAxis(data=data_b, dims=["ch"])

        merged = _build_merged_coordinate_axis(ax_a, ax_b, relabel=False, label_a="", label_b="")
        assert merged.data[0]["label"] == "ch0"
        assert merged.data[1]["label"] == "ch1"
        np.testing.assert_allclose(merged.data[0]["x"], 1.0)

    def test_struct_union_fields(self):
        """Inputs with different struct fields produce a union dtype."""
        dt_a = np.dtype([("label", "U8"), ("x", "f4")])
        dt_b = np.dtype([("label", "U8"), ("bank", "i4")])
        data_a = np.array([("ch0", 1.0)], dtype=dt_a)
        data_b = np.array([("ch1", 42)], dtype=dt_b)
        ax_a = CoordinateAxis(data=data_a, dims=["ch"])
        ax_b = CoordinateAxis(data=data_b, dims=["ch"])

        merged = _build_merged_coordinate_axis(ax_a, ax_b, relabel=False, label_a="", label_b="")
        assert "label" in merged.data.dtype.names
        assert "x" in merged.data.dtype.names
        assert "bank" in merged.data.dtype.names
        np.testing.assert_allclose(merged.data[0]["x"], 1.0)
        assert merged.data[1]["bank"] == 42
        # Missing field in B gets default (0.0 for float).
        np.testing.assert_allclose(merged.data[1]["x"], 0.0)

    def test_struct_incompatible_dtypes_raises(self):
        dt_a = np.dtype([("label", "U8"), ("x", "f4")])
        dt_b = np.dtype([("label", "U8"), ("x", "i4")])
        data_a = np.array([("ch0", 1.0)], dtype=dt_a)
        data_b = np.array([("ch1", 2)], dtype=dt_b)
        ax_a = CoordinateAxis(data=data_a, dims=["ch"])
        ax_b = CoordinateAxis(data=data_b, dims=["ch"])

        with pytest.raises(ValueError, match="Incompatible dtypes"):
            _build_merged_coordinate_axis(ax_a, ax_b, relabel=False, label_a="", label_b="")

    def test_struct_label_created_when_absent(self):
        """If relabel=True and no 'label' field exists, one is added."""
        dt = np.dtype([("x", "f4"), ("y", "f4")])
        data_a = np.array([(1.0, 2.0)], dtype=dt)
        data_b = np.array([(3.0, 4.0)], dtype=dt)
        ax_a = CoordinateAxis(data=data_a, dims=["ch"])
        ax_b = CoordinateAxis(data=data_b, dims=["ch"])

        merged = _build_merged_coordinate_axis(ax_a, ax_b, relabel=True, label_a="_L", label_b="_R")
        assert "label" in merged.data.dtype.names
        assert merged.data[0]["label"] == "0_L"
        assert merged.data[1]["label"] == "0_R"


class TestValidateSharedAxes:
    def test_identical_axes_pass(self):
        ch_ax = CoordinateAxis(data=np.array(["C0", "C1"]), dims=["ch"])
        a = AxisArray(np.ones((5, 2)), dims=["time", "ch"], axes={"ch": ch_ax})
        b = AxisArray(np.ones((5, 2)), dims=["time", "ch"], axes={"ch": ch_ax})
        # Should not raise.
        _validate_shared_axes(a, b, concat_dim="feature", align_dim="time", assert_flag=True)

    def test_different_axes_raise(self):
        ax_a = CoordinateAxis(data=np.array(["C0", "C1"]), dims=["ch"])
        ax_b = CoordinateAxis(data=np.array(["X0", "X1"]), dims=["ch"])
        a = AxisArray(np.ones((5, 2)), dims=["time", "ch"], axes={"ch": ax_a})
        b = AxisArray(np.ones((5, 2)), dims=["time", "ch"], axes={"ch": ax_b})
        with pytest.raises(ValueError, match="Shared axis 'ch'"):
            _validate_shared_axes(a, b, concat_dim="feature", align_dim="time", assert_flag=True)

    def test_flag_false_skips(self):
        ax_a = CoordinateAxis(data=np.array(["C0", "C1"]), dims=["ch"])
        ax_b = CoordinateAxis(data=np.array(["X0", "X1"]), dims=["ch"])
        a = AxisArray(np.ones((5, 2)), dims=["time", "ch"], axes={"ch": ax_a})
        b = AxisArray(np.ones((5, 2)), dims=["time", "ch"], axes={"ch": ax_b})
        # Should not raise when flag is False.
        _validate_shared_axes(a, b, concat_dim="feature", align_dim="time", assert_flag=False)


# ---------------------------------------------------------------------------
# ConcatProcessor tests
# ---------------------------------------------------------------------------


class TestBasicConcat:
    def test_concat_along_ch(self):
        settings = ConcatSettings(axis="ch")
        proc = ConcatProcessor(settings)
        n, fs = 10, 100.0

        data_a = np.arange(n * 2, dtype=float).reshape(n, 2)
        data_b = np.arange(n * 3, dtype=float).reshape(n, 3) + 100
        msg_a = _make_msg(data_a, fs=fs, ch_labels=["A0", "A1"])
        msg_b = _make_msg(data_b, fs=fs, ch_labels=["B0", "B1", "B2"])

        proc.push_a(msg_a)
        proc.push_b(msg_b)

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(proc.__acall__())
        assert result.data.shape == (n, 5)
        np.testing.assert_array_equal(result.data[:, :2], data_a)
        np.testing.assert_array_equal(result.data[:, 2:], data_b)

    def test_concat_direct(self):
        """Test _concat directly (synchronous path used by MergeProcessor)."""
        settings = ConcatSettings(axis="ch", relabel_axis=False)
        proc = ConcatProcessor(settings)
        n = 5

        data_a = np.ones((n, 2))
        data_b = np.ones((n, 3)) * 2
        msg_a = _make_msg(data_a, ch_labels=["A0", "A1"])
        msg_b = _make_msg(data_b, ch_labels=["B0", "B1", "B2"])

        result = proc._concat(msg_a, msg_b)
        assert result.data.shape == (n, 5)
        labels = list(result.axes["ch"].data)
        assert labels == ["A0", "A1", "B0", "B1", "B2"]


class TestConcatRelabel:
    def test_default_suffix(self):
        settings = ConcatSettings(axis="ch", label_a="_left", label_b="_right")
        proc = ConcatProcessor(settings)
        n = 5

        msg_a = _make_msg(np.ones((n, 2)), ch_labels=["X", "Y"])
        msg_b = _make_msg(np.ones((n, 3)), ch_labels=["X", "Y", "Z"])

        result = proc._concat(msg_a, msg_b)
        labels = list(result.axes["ch"].data)
        assert labels == ["X_left", "Y_left", "X_right", "Y_right", "Z_right"]

    def test_no_relabel(self):
        settings = ConcatSettings(axis="ch", relabel_axis=False)
        proc = ConcatProcessor(settings)
        n = 5

        msg_a = _make_msg(np.ones((n, 2)), ch_labels=["A0", "A1"])
        msg_b = _make_msg(np.ones((n, 3)), ch_labels=["B0", "B1", "B2"])

        result = proc._concat(msg_a, msg_b)
        labels = list(result.axes["ch"].data)
        assert labels == ["A0", "A1", "B0", "B1", "B2"]


class TestNewAxisConcat:
    def test_new_feature_axis(self):
        settings = ConcatSettings(axis="feature", relabel_axis=False)
        proc = ConcatProcessor(settings)
        n = 5

        data_a = np.ones((n, 3))
        data_b = np.ones((n, 3)) * 2
        msg_a = _make_msg(data_a, ch_labels=["C0", "C1", "C2"])
        msg_b = _make_msg(data_b, ch_labels=["C0", "C1", "C2"])

        result = proc._concat(msg_a, msg_b)
        assert result.data.shape == (n, 3, 2)
        np.testing.assert_array_equal(result.data[:, :, 0], 1.0)
        np.testing.assert_array_equal(result.data[:, :, 1], 2.0)
        assert "feature" in result.dims

    def test_new_axis_dim_mismatch_raises(self):
        settings = ConcatSettings(axis="feature", relabel_axis=False)
        proc = ConcatProcessor(settings)
        n = 5

        msg_a = _make_msg(np.ones((n, 3)), ch_labels=["C0", "C1", "C2"])
        msg_b = _make_msg(np.ones((n, 4)), ch_labels=["C0", "C1", "C2", "C3"])

        with pytest.raises(ValueError, match="Cannot concatenate along new axis"):
            proc._concat(msg_a, msg_b)


class TestAssertIdenticalSharedAxes:
    def test_identical_passes(self):
        settings = ConcatSettings(axis="feature", assert_identical_shared_axes=True)
        proc = ConcatProcessor(settings)
        n = 5

        msg_a = _make_msg(np.ones((n, 2)), ch_labels=["C0", "C1"])
        msg_b = _make_msg(np.ones((n, 2)) * 2, ch_labels=["C0", "C1"])

        result = proc._concat(msg_a, msg_b)
        assert result.data.shape == (n, 2, 2)

    def test_different_raises(self):
        settings = ConcatSettings(axis="feature", assert_identical_shared_axes=True)
        proc = ConcatProcessor(settings)
        n = 5

        msg_a = _make_msg(np.ones((n, 2)), ch_labels=["C0", "C1"])
        msg_b = _make_msg(np.ones((n, 2)) * 2, ch_labels=["X0", "X1"])

        with pytest.raises(ValueError, match="Shared axis 'ch'"):
            proc._concat(msg_a, msg_b)

    def test_flag_false_allows_different(self):
        settings = ConcatSettings(axis="feature", assert_identical_shared_axes=False)
        proc = ConcatProcessor(settings)
        n = 5

        msg_a = _make_msg(np.ones((n, 2)), ch_labels=["C0", "C1"])
        msg_b = _make_msg(np.ones((n, 2)) * 2, ch_labels=["X0", "X1"])

        result = proc._concat(msg_a, msg_b)
        assert result.data.shape == (n, 2, 2)


class TestStructAwareConcat:
    def test_struct_axis_concat(self):
        settings = ConcatSettings(axis="ch", label_a="_L", label_b="_R")
        proc = ConcatProcessor(settings)
        n = 5

        dt = np.dtype([("label", "U8"), ("x", "f4"), ("y", "f4")])
        ch_a = np.array([("ch0", 1.0, 2.0), ("ch1", 3.0, 4.0)], dtype=dt)
        ch_b = np.array([("ch2", 5.0, 6.0)], dtype=dt)
        ax_a = CoordinateAxis(data=ch_a, dims=["ch"])
        ax_b = CoordinateAxis(data=ch_b, dims=["ch"])

        msg_a = _make_msg(np.ones((n, 2)), ch_axis=ax_a)
        msg_b = _make_msg(np.ones((n, 1)) * 2, ch_axis=ax_b)

        result = proc._concat(msg_a, msg_b)
        assert result.data.shape == (n, 3)
        ch_out = result.axes["ch"].data
        assert ch_out[0]["label"] == "ch0_L"
        assert ch_out[2]["label"] == "ch2_R"
        np.testing.assert_allclose(ch_out[0]["x"], 1.0)
        np.testing.assert_allclose(ch_out[2]["y"], 6.0)


class TestCachedAxes:
    def test_cache_reused(self):
        settings = ConcatSettings(axis="ch")
        proc = ConcatProcessor(settings)
        n = 5

        msg_a = _make_msg(np.ones((n, 2)), ch_labels=["A0", "A1"])
        msg_b = _make_msg(np.ones((n, 2)), ch_labels=["B0", "B1"])

        proc._concat(msg_a, msg_b)
        first_cache = proc.state.cached_axes

        proc._concat(msg_a, msg_b)
        assert proc.state.cached_axes is first_cache  # Same dict object.

    def test_cache_invalidated_on_shape_change(self):
        settings = ConcatSettings(axis="ch", relabel_axis=False)
        proc = ConcatProcessor(settings)
        n = 5

        msg_a1 = _make_msg(np.ones((n, 2)), ch_labels=["A0", "A1"])
        msg_b1 = _make_msg(np.ones((n, 3)), ch_labels=["B0", "B1", "B2"])
        proc._concat(msg_a1, msg_b1)
        first_cache = proc.state.cached_axes

        # Change A to 4 channels.
        msg_a2 = _make_msg(np.ones((n, 4)), ch_labels=["A0", "A1", "A2", "A3"])
        proc._concat(msg_a2, msg_b1)
        assert proc.state.cached_axes is not first_cache
