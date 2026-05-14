"""Unit tests for ezmsg.sigproc.flatten."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.sigproc.flatten import FlattenSettings, FlattenTransformer
from tests.helpers.util import requires_mlx


def _make_3d(
    fs: float = 50.0,
    n_time: int = 5,
    ch_labels=("c1", "c2", "c3"),
    feature_labels=("spk", "sbp"),
):
    """Build a (time, ch, feature) AxisArray with both flatten axes labeled."""
    data = np.arange(n_time * len(ch_labels) * len(feature_labels), dtype=float).reshape(
        n_time, len(ch_labels), len(feature_labels)
    )
    return AxisArray(
        data=data,
        dims=["time", "ch", "feature"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=0.0),
            "ch": CoordinateAxis(data=np.array(ch_labels), dims=["ch"]),
            "feature": CoordinateAxis(data=np.array(feature_labels), dims=["feature"]),
        },
        key="features",
    )


class TestCartesianProductLabels:
    def test_default_flatten_3d_to_2d(self):
        """Default flatten_axes = all-non-preserve, in input order.

        For ``(time, ch, feature)`` the 'feature' axis varies fastest in
        the merged axis (matching numpy's C-order reshape).  Output is a
        structured axis with one field per source axis plus a canonical
        ``"label"`` field carrying the cartesian-product join.
        """
        msg = _make_3d()
        proc = FlattenTransformer(FlattenSettings())
        out = proc(msg)
        assert out.dims == ["time", "ch"]
        assert out.data.shape == (5, 6)
        # data[0,0,0]=0, data[0,0,1]=1, data[0,1,0]=2, data[0,1,1]=3, ...
        np.testing.assert_array_equal(out.data[0], [0, 1, 2, 3, 4, 5])

        ch_axis = out.axes["ch"]
        assert ch_axis.data.dtype.names == ("ch", "feature", "label")
        assert list(ch_axis.data["ch"]) == ["c1", "c1", "c2", "c2", "c3", "c3"]
        assert list(ch_axis.data["feature"]) == ["spk", "sbp", "spk", "sbp", "spk", "sbp"]
        assert list(ch_axis.data["label"]) == [
            "c1/spk",
            "c1/sbp",
            "c2/spk",
            "c2/sbp",
            "c3/spk",
            "c3/sbp",
        ]

    def test_label_separator_override(self):
        msg = _make_3d()
        proc = FlattenTransformer(FlattenSettings(label_separator="__"))
        ch_axis = proc(msg).axes["ch"]
        assert ch_axis.data["label"][0] == "c1__spk"
        assert ch_axis.data["label"][5] == "c3__sbp"

    def test_explicit_flatten_axes_order_changes_label_grouping(self):
        """``flatten_axes=("feature", "ch")`` → labels group by feature first."""
        msg = _make_3d()
        proc = FlattenTransformer(FlattenSettings(flatten_axes=("feature", "ch")))
        ch_axis = proc(msg).axes["ch"]
        # Now feature is slowest-varying, ch is fastest.
        assert list(ch_axis.data["label"]) == [
            "spk/c1",
            "spk/c2",
            "spk/c3",
            "sbp/c1",
            "sbp/c2",
            "sbp/c3",
        ]
        assert list(ch_axis.data["feature"]) == ["spk"] * 3 + ["sbp"] * 3
        assert list(ch_axis.data["ch"]) == ["c1", "c2", "c3"] * 2

    def test_falls_back_to_int_labels_when_axes_unlabeled(self):
        """Axes without CoordinateAxis → 1-based uint32 indices per axis."""
        data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        msg = AxisArray(
            data=data,
            dims=["time", "ch", "feature"],
            axes={"time": AxisArray.TimeAxis(fs=50.0, offset=0.0)},
            key="raw",
        )
        proc = FlattenTransformer(FlattenSettings())
        ch_axis = proc(msg).axes["ch"]
        # ch is slowest (size 3), feature is fastest (size 4).
        np.testing.assert_array_equal(
            ch_axis.data["ch"],
            np.asarray([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            ch_axis.data["feature"],
            np.asarray([1, 2, 3, 4] * 3, dtype=np.uint32),
        )
        assert list(ch_axis.data["label"]) == [
            "1/1",
            "1/2",
            "1/3",
            "1/4",
            "2/1",
            "2/2",
            "2/3",
            "2/4",
            "3/1",
            "3/2",
            "3/3",
            "3/4",
        ]

    def test_preserve_axis_kept_through(self):
        msg = _make_3d(fs=200.0)
        proc = FlattenTransformer(FlattenSettings())
        out = proc(msg)
        time_ax = out.axes["time"]
        assert time_ax.gain == pytest.approx(1.0 / 200.0)


class TestStructFieldPassthrough:
    def test_struct_axis_propagates_all_fields(self):
        """A structured ``ch`` axis contributes every field to the merged struct."""
        ch_dtype = np.dtype(
            [
                ("x", "f4"),
                ("y", "f4"),
                ("label", "U16"),
                ("device", "U8"),
            ]
        )
        ch_data = np.zeros(2, dtype=ch_dtype)
        ch_data["x"] = [1.0, 2.0]
        ch_data["y"] = [3.0, 4.0]
        ch_data["label"] = ["e1", "e2"]
        ch_data["device"] = ["HUB1", "HUB2"]
        msg = AxisArray(
            data=np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2),
            dims=["time", "ch", "feature"],
            axes={
                "time": AxisArray.TimeAxis(fs=50.0, offset=0.0),
                "ch": CoordinateAxis(data=ch_data, dims=["ch"]),
                "feature": CoordinateAxis(data=np.array(("spk", "sbp")), dims=["feature"]),
            },
            key="features",
        )
        proc = FlattenTransformer(FlattenSettings())
        out_ax = proc(msg).axes["ch"]

        # All struct fields from ``ch`` survive + the ``feature`` axis
        # contributes its own field + ``label`` is the cartesian join.
        assert out_ax.data.dtype.names == ("x", "y", "label", "device", "feature")
        np.testing.assert_array_equal(out_ax.data["x"], [1.0, 1.0, 2.0, 2.0])
        np.testing.assert_array_equal(out_ax.data["y"], [3.0, 3.0, 4.0, 4.0])
        assert list(out_ax.data["device"]) == ["HUB1", "HUB1", "HUB2", "HUB2"]
        assert list(out_ax.data["feature"]) == ["spk", "sbp", "spk", "sbp"]
        # Canonical label overwrites the inherited per-channel label.
        assert list(out_ax.data["label"]) == ["e1/spk", "e1/sbp", "e2/spk", "e2/sbp"]


class TestSampleAxisRename:
    def test_sample_axis_renames_preserve(self):
        """``sample_axis`` renames preserve_axis on the output (e.g. win→time)."""
        data = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
        msg = AxisArray(
            data=data,
            dims=["win", "time", "ch"],
            axes={
                "win": AxisArray.TimeAxis(fs=10.0, offset=0.2),
                "time": AxisArray.TimeAxis(fs=100.0, offset=-0.02),
                "ch": CoordinateAxis(data=np.array(["c0", "c1"]), dims=["ch"]),
            },
            key="signal",
        )
        proc = FlattenTransformer(FlattenSettings(preserve_axis="win", sample_axis="time"))
        out = proc(msg)
        assert out.dims == ["time", "ch"]
        assert out.data.shape == (2, 6)
        assert out.axes["time"].gain == pytest.approx(1.0 / 10.0)


class TestSettingsValidation:
    def test_unknown_preserve_axis_raises(self):
        msg = _make_3d()
        proc = FlattenTransformer(FlattenSettings(preserve_axis="not_there"))
        with pytest.raises(ValueError, match="preserve_axis"):
            proc(msg)

    def test_unknown_flatten_axis_raises(self):
        msg = _make_3d()
        proc = FlattenTransformer(FlattenSettings(flatten_axes=("ch", "not_there")))
        with pytest.raises(ValueError, match="flatten_axes"):
            proc(msg)

    def test_preserve_in_flatten_axes_raises(self):
        msg = _make_3d()
        proc = FlattenTransformer(FlattenSettings(flatten_axes=("time", "ch")))
        with pytest.raises(ValueError, match="cannot also be in"):
            proc(msg)


# ============== MLX Tests ==============
@requires_mlx
class TestMLXBackend:
    """FlattenTransformer should work end-to-end on mlx.core arrays."""

    def _make_3d_mlx(self, np_data):
        import mlx.core as mx

        return AxisArray(
            data=mx.array(np_data),
            dims=["time", "ch", "feature"],
            axes={
                "time": AxisArray.TimeAxis(fs=50.0, offset=0.0),
                "ch": CoordinateAxis(data=np.array(["c1", "c2", "c3"]), dims=["ch"]),
                "feature": CoordinateAxis(data=np.array(["spk", "sbp"]), dims=["feature"]),
            },
            key="features",
        )

    def test_default_flatten_mlx_matches_numpy(self):
        """No-permute path: dims already in (preserve, *flatten) order."""
        import mlx.core as mx

        np_data = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
        msg_np = AxisArray(
            data=np_data,
            dims=["time", "ch", "feature"],
            axes={
                "time": AxisArray.TimeAxis(fs=50.0, offset=0.0),
                "ch": CoordinateAxis(data=np.array(["c1", "c2", "c3"]), dims=["ch"]),
                "feature": CoordinateAxis(data=np.array(["spk", "sbp"]), dims=["feature"]),
            },
            key="features",
        )
        msg_mx = self._make_3d_mlx(np_data)

        proc_np = FlattenTransformer(FlattenSettings())
        proc_mx = FlattenTransformer(FlattenSettings())

        out_np = proc_np(msg_np)
        out_mx = proc_mx(msg_mx)

        assert isinstance(out_mx.data, mx.array), f"Expected mx.array, got {type(out_mx.data)}"
        assert out_mx.data.shape == out_np.data.shape == (5, 6)
        assert out_mx.dims == out_np.dims == ["time", "ch"]
        np.testing.assert_allclose(np.asarray(out_mx.data), out_np.data)
        # Cartesian-product label field is backend-independent.
        np.testing.assert_array_equal(out_mx.axes["ch"].data["label"], out_np.axes["ch"].data["label"])

    def test_permute_path_mlx_matches_numpy(self):
        """Permute path: explicit flatten_axes reorders the axes."""
        import mlx.core as mx

        np_data = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
        axes = {
            "time": AxisArray.TimeAxis(fs=50.0, offset=0.0),
            "ch": CoordinateAxis(data=np.array(["c1", "c2", "c3"]), dims=["ch"]),
            "feature": CoordinateAxis(data=np.array(["spk", "sbp"]), dims=["feature"]),
        }
        msg_np = AxisArray(data=np_data, dims=["time", "ch", "feature"], axes=axes, key="features")
        msg_mx = AxisArray(
            data=mx.array(np_data),
            dims=["time", "ch", "feature"],
            axes=axes,
            key="features",
        )

        settings = FlattenSettings(flatten_axes=("feature", "ch"))
        out_np = FlattenTransformer(settings)(msg_np)
        out_mx = FlattenTransformer(settings)(msg_mx)

        assert isinstance(out_mx.data, mx.array)
        assert out_mx.data.shape == out_np.data.shape == (5, 6)
        np.testing.assert_allclose(np.asarray(out_mx.data), out_np.data)

    def test_state_caching_across_messages_mlx(self):
        """Second message should hit the cached state and still produce
        correct MLX output (covers _hash_message + cached output_axis_obj)."""
        import mlx.core as mx

        proc = FlattenTransformer(FlattenSettings())
        np_data1 = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
        np_data2 = np_data1 + 100.0

        out1 = proc(self._make_3d_mlx(np_data1))
        out2 = proc(self._make_3d_mlx(np_data2))

        assert isinstance(out1.data, mx.array)
        assert isinstance(out2.data, mx.array)
        np.testing.assert_allclose(
            np.asarray(out2.data) - np.asarray(out1.data),
            np.full((5, 6), 100.0, dtype=np.float32),
        )
        # Same cached output_axis_obj is reused across messages.
        assert out1.axes["ch"] is out2.axes["ch"]
