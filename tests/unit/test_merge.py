"""Unit tests for ezmsg.sigproc.merge module."""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from frozendict import frozendict

from ezmsg.sigproc.merge import MergeProcessor, MergeSettings


def _make_msg(
    data: np.ndarray,
    fs: float = 100.0,
    offset: float = 0.0,
    ch_labels: list[str] | None = None,
    key: str = "",
) -> AxisArray:
    """Helper to build a (time, ch) AxisArray."""
    n_ch = data.shape[1] if data.ndim > 1 else 1
    if data.ndim == 1:
        data = data[:, None]
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


class TestAlignedMerge:
    """Both streams perfectly aligned — verify output shape and data."""

    def test_basic(self):
        settings = MergeSettings(axis="ch")
        proc = MergeProcessor(settings)

        fs = 100.0
        n_samples = 10
        n_ch_a, n_ch_b = 2, 3
        data_a = np.arange(n_samples * n_ch_a, dtype=float).reshape(n_samples, n_ch_a)
        data_b = np.arange(n_samples * n_ch_b, dtype=float).reshape(n_samples, n_ch_b) + 100

        msg_a = _make_msg(data_a, fs=fs, offset=0.0, ch_labels=["A0", "A1"])
        msg_b = _make_msg(data_b, fs=fs, offset=0.0, ch_labels=["B0", "B1", "B2"])

        result_a = proc(msg_a)
        # Only one side present so far — no merge yet.
        assert result_a is None

        merged = proc.push_b(msg_b)
        assert merged is not None
        assert merged.data.shape == (n_samples, n_ch_a + n_ch_b)
        np.testing.assert_array_equal(merged.data[:, :n_ch_a], data_a)
        np.testing.assert_array_equal(merged.data[:, n_ch_a:], data_b)

    def test_multiple_chunks(self):
        """Feed several aligned chunks and verify each produces output."""
        settings = MergeSettings(axis="ch")
        proc = MergeProcessor(settings)
        fs = 100.0
        chunk_size = 5

        for i in range(4):
            offset = i * chunk_size / fs
            data_a = np.ones((chunk_size, 2)) * i
            data_b = np.ones((chunk_size, 3)) * (i + 10)
            msg_a = _make_msg(data_a, fs=fs, offset=offset, ch_labels=["A0", "A1"])
            msg_b = _make_msg(data_b, fs=fs, offset=offset, ch_labels=["B0", "B1", "B2"])

            proc(msg_a)
            merged = proc.push_b(msg_b)
            assert merged is not None
            assert merged.data.shape == (chunk_size, 5)


class TestOffsetStagger:
    """Stream A arrives first (several chunks), then B catches up."""

    def test_staggered_arrival(self):
        settings = MergeSettings(axis="ch")
        proc = MergeProcessor(settings)
        fs = 100.0
        chunk_size = 10

        # Push 3 chunks to A with no B present.
        for i in range(3):
            offset = i * chunk_size / fs
            data_a = np.ones((chunk_size, 2)) * i
            msg_a = _make_msg(data_a, fs=fs, offset=offset, ch_labels=["A0", "A1"])
            assert proc(msg_a) is None

        # Now push B starting at chunk 1's offset.
        b_offset = 1 * chunk_size / fs
        data_b = np.ones((chunk_size, 3)) * 99
        msg_b = _make_msg(data_b, fs=fs, offset=b_offset, ch_labels=["B0", "B1", "B2"])
        merged = proc.push_b(msg_b)

        assert merged is not None
        # Should align to chunk 1's time range.
        assert merged.data.shape[0] == chunk_size
        assert merged.data.shape[1] == 5
        # A data at chunk 1 was all 1s.
        np.testing.assert_allclose(merged.data[:, :2], 1.0)
        np.testing.assert_allclose(merged.data[:, 2:], 99.0)


class TestFloatingPointOffset:
    """Offsets that differ by a tiny float epsilon should merge successfully."""

    def test_tiny_offset_difference(self):
        settings = MergeSettings(axis="ch")
        proc = MergeProcessor(settings)
        fs = 1000.0
        n = 20
        offset = 1.0

        data_a = np.ones((n, 2))
        data_b = np.ones((n, 2)) * 2

        # Introduce a tiny floating-point mismatch.
        eps = 1e-14
        msg_a = _make_msg(data_a, fs=fs, offset=offset, ch_labels=["A0", "A1"])
        msg_b = _make_msg(data_b, fs=fs, offset=offset + eps, ch_labels=["B0", "B1"])

        proc(msg_a)
        merged = proc.push_b(msg_b)
        assert merged is not None
        assert merged.data.shape == (n, 4)


class TestGainMismatch:
    """Gain mismatch triggers a full reset, not an error."""

    def test_b_mismatched_gain_resets(self):
        """B has a different gain than A — triggers full reset, no error."""
        settings = MergeSettings(axis="ch")
        proc = MergeProcessor(settings)
        n = 10

        msg_a = _make_msg(np.ones((n, 2)), fs=100.0, ch_labels=["A0", "A1"])
        msg_b_bad = _make_msg(np.ones((n, 3)), fs=200.0, ch_labels=["B0", "B1", "B2"])

        proc(msg_a)
        # B at different gain — full reset, no merge, no error.
        result = proc.push_b(msg_b_bad)
        assert result is None

        # A at 200 Hz arrives — merges with buffered B from the reset.
        msg_a2 = _make_msg(np.ones((n, 2)) * 2, fs=200.0, offset=0.0, ch_labels=["A0", "A1"])
        merged = proc(msg_a2)
        assert merged is not None
        assert merged.data.shape == (n, 5)

    def test_a_gain_change_resets(self):
        """A changes gain mid-stream — state resets, merge resumes."""
        settings = MergeSettings(axis="ch")
        proc = MergeProcessor(settings)
        n = 5

        # First pair at 100 Hz.
        msg_a1 = _make_msg(np.ones((n, 2)), fs=100.0, ch_labels=["A0", "A1"])
        msg_b1 = _make_msg(np.ones((n, 3)), fs=100.0, ch_labels=["B0", "B1", "B2"])
        proc(msg_a1)
        merged = proc.push_b(msg_b1)
        assert merged is not None

        # A switches to 200 Hz — triggers _reset_state.
        msg_a2 = _make_msg(np.ones((n, 2)) * 2, fs=200.0, offset=0.5, ch_labels=["A0", "A1"])
        result = proc(msg_a2)
        # B buffer was reset, so no merge yet.
        assert result is None

        # B at 200 Hz arrives — merge succeeds.
        msg_b2 = _make_msg(np.ones((n, 3)) * 3, fs=200.0, offset=0.5, ch_labels=["B0", "B1", "B2"])
        merged = proc.push_b(msg_b2)
        assert merged is not None
        assert merged.data.shape == (n, 5)

    def test_b_gain_change_resets(self):
        """B changes gain mid-stream — full reset, merge resumes."""
        settings = MergeSettings(axis="ch")
        proc = MergeProcessor(settings)
        n = 5

        # First pair at 100 Hz.
        msg_a1 = _make_msg(np.ones((n, 2)), fs=100.0, ch_labels=["A0", "A1"])
        msg_b1 = _make_msg(np.ones((n, 3)), fs=100.0, ch_labels=["B0", "B1", "B2"])
        proc(msg_a1)
        merged = proc.push_b(msg_b1)
        assert merged is not None

        # B switches to 200 Hz — full reset. B is buffered.
        msg_b2 = _make_msg(np.ones((n, 3)) * 2, fs=200.0, offset=0.5, ch_labels=["B0", "B1", "B2"])
        result = proc.push_b(msg_b2)
        assert result is None

        # A at 200 Hz arrives — merges with buffered B.
        msg_a2 = _make_msg(np.ones((n, 2)) * 3, fs=200.0, offset=0.5, ch_labels=["A0", "A1"])
        merged = proc(msg_a2)
        assert merged is not None
        assert merged.data.shape == (n, 5)
        np.testing.assert_allclose(merged.data[:, :2], 3.0)
        np.testing.assert_allclose(merged.data[:, 2:], 2.0)


class TestConcatDimChange:
    """Concat-axis dimensionality change resets per-input + common state."""

    def test_a_concat_dim_change(self):
        """A changes channel count — A buffer reset, B preserved."""
        settings = MergeSettings(axis="ch", relabel_axis=False)
        proc = MergeProcessor(settings)
        fs = 100.0
        n = 5

        # First pair: A(ch=2), B(ch=3).
        offset_0 = 0.0
        msg_a1 = _make_msg(np.ones((n, 2)), fs=fs, offset=offset_0, ch_labels=["A0", "A1"])
        msg_b1 = _make_msg(np.ones((n, 3)), fs=fs, offset=offset_0, ch_labels=["B0", "B1", "B2"])
        proc(msg_a1)
        merged = proc.push_b(msg_b1)
        assert merged is not None
        assert merged.data.shape == (n, 5)

        # Push B at next offset (ch=3 still).
        offset_1 = n / fs
        msg_b2 = _make_msg(
            np.ones((n, 3)) * 2,
            fs=fs,
            offset=offset_1,
            ch_labels=["B0", "B1", "B2"],
        )
        assert proc.push_b(msg_b2) is None  # No A data yet.

        # A changes to ch=4 at same offset — partial reset + re-alignment.
        msg_a2 = _make_msg(
            np.ones((n, 4)) * 3,
            fs=fs,
            offset=offset_1,
            ch_labels=["A0", "A1", "A2", "A3"],
        )
        merged = proc(msg_a2)
        assert merged is not None
        assert merged.data.shape == (n, 7)  # 4 + 3
        np.testing.assert_allclose(merged.data[:, :4], 3.0)
        np.testing.assert_allclose(merged.data[:, 4:], 2.0)

    def test_b_concat_dim_change(self):
        """B changes channel count — B buffer reset, A preserved."""
        settings = MergeSettings(axis="ch", relabel_axis=False)
        proc = MergeProcessor(settings)
        fs = 100.0
        n = 5

        # First pair: A(ch=2), B(ch=3).
        offset_0 = 0.0
        msg_a1 = _make_msg(np.ones((n, 2)), fs=fs, offset=offset_0, ch_labels=["A0", "A1"])
        msg_b1 = _make_msg(np.ones((n, 3)), fs=fs, offset=offset_0, ch_labels=["B0", "B1", "B2"])
        proc(msg_a1)
        merged = proc.push_b(msg_b1)
        assert merged is not None
        assert merged.data.shape == (n, 5)

        # Push A at next offset (ch=2 still).
        offset_1 = n / fs
        msg_a2 = _make_msg(np.ones((n, 2)) * 2, fs=fs, offset=offset_1, ch_labels=["A0", "A1"])
        assert proc(msg_a2) is None  # No B data yet.

        # B changes to ch=5 at same offset — partial reset + re-alignment.
        msg_b2 = _make_msg(
            np.ones((n, 5)) * 3,
            fs=fs,
            offset=offset_1,
            ch_labels=["B0", "B1", "B2", "B3", "B4"],
        )
        merged = proc.push_b(msg_b2)
        assert merged is not None
        assert merged.data.shape == (n, 7)  # 2 + 5
        np.testing.assert_allclose(merged.data[:, :2], 2.0)
        np.testing.assert_allclose(merged.data[:, 2:], 3.0)


class TestRelabelAxis:
    """Verify that coordinate axis labels are suffixed correctly."""

    def test_default_relabel(self):
        settings = MergeSettings(axis="ch", label_a="_left", label_b="_right")
        proc = MergeProcessor(settings)
        fs = 100.0
        n = 5

        data_a = np.ones((n, 2))
        data_b = np.ones((n, 3))
        msg_a = _make_msg(data_a, fs=fs, ch_labels=["X", "Y"])
        msg_b = _make_msg(data_b, fs=fs, ch_labels=["X", "Y", "Z"])

        proc(msg_a)
        merged = proc.push_b(msg_b)
        assert merged is not None
        ch_ax = merged.axes["ch"]
        labels = list(ch_ax.data)
        assert labels == ["X_left", "Y_left", "X_right", "Y_right", "Z_right"]

    def test_cached_axis_reused(self):
        """Verify the merged concat axis is built once and reused."""
        settings = MergeSettings(axis="ch")
        proc = MergeProcessor(settings)
        fs = 100.0
        n = 5

        for i in range(3):
            offset = i * n / fs
            msg_a = _make_msg(np.ones((n, 2)), fs=fs, offset=offset, ch_labels=["A0", "A1"])
            msg_b = _make_msg(np.ones((n, 2)), fs=fs, offset=offset, ch_labels=["B0", "B1"])
            proc(msg_a)
            proc.push_b(msg_b)

        # The cached axis should have been built once and reused.
        assert proc.state.merged_concat_axis is not None


class TestNoRelabel:
    """relabel_axis=False preserves original labels."""

    def test_no_relabel(self):
        settings = MergeSettings(axis="ch", relabel_axis=False)
        proc = MergeProcessor(settings)
        fs = 100.0
        n = 5

        data_a = np.ones((n, 2))
        data_b = np.ones((n, 3))
        msg_a = _make_msg(data_a, fs=fs, ch_labels=["A0", "A1"])
        msg_b = _make_msg(data_b, fs=fs, ch_labels=["B0", "B1", "B2"])

        proc(msg_a)
        merged = proc.push_b(msg_b)
        assert merged is not None
        ch_ax = merged.axes["ch"]
        labels = list(ch_ax.data)
        assert labels == ["A0", "A1", "B0", "B1", "B2"]
