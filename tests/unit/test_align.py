"""Unit tests for ezmsg.sigproc.align module."""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from frozendict import frozendict

from ezmsg.sigproc.align import AlignAlongAxisProcessor, AlignAlongAxisSettings


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


class TestAlignedPairs:
    """Both streams perfectly aligned — verify paired output."""

    def test_basic(self):
        settings = AlignAlongAxisSettings(axis="time")
        proc = AlignAlongAxisProcessor(settings)
        fs = 100.0
        n = 10

        data_a = np.arange(n * 2, dtype=float).reshape(n, 2)
        data_b = np.arange(n * 3, dtype=float).reshape(n, 3) + 100

        msg_a = _make_msg(data_a, fs=fs, offset=0.0, ch_labels=["A0", "A1"])
        msg_b = _make_msg(data_b, fs=fs, offset=0.0, ch_labels=["B0", "B1", "B2"])

        pair_from_a = proc(msg_a)
        assert pair_from_a is None  # Only one side present.

        pair = proc.push_b(msg_b)
        assert pair is not None
        aa_a, aa_b = pair
        assert aa_a.data.shape == (n, 2)
        assert aa_b.data.shape == (n, 3)
        np.testing.assert_array_equal(aa_a.data, data_a)
        np.testing.assert_array_equal(aa_b.data, data_b)

    def test_multiple_chunks(self):
        settings = AlignAlongAxisSettings(axis="time")
        proc = AlignAlongAxisProcessor(settings)
        fs = 100.0
        chunk = 5

        for i in range(4):
            offset = i * chunk / fs
            data_a = np.ones((chunk, 2)) * i
            data_b = np.ones((chunk, 3)) * (i + 10)
            msg_a = _make_msg(data_a, fs=fs, offset=offset, ch_labels=["A0", "A1"])
            msg_b = _make_msg(data_b, fs=fs, offset=offset, ch_labels=["B0", "B1", "B2"])
            proc(msg_a)
            pair = proc.push_b(msg_b)
            assert pair is not None
            assert pair[0].data.shape == (chunk, 2)
            assert pair[1].data.shape == (chunk, 3)


class TestStaggeredArrival:
    def test_a_arrives_first(self):
        settings = AlignAlongAxisSettings(axis="time")
        proc = AlignAlongAxisProcessor(settings)
        fs = 100.0
        chunk = 10

        for i in range(3):
            offset = i * chunk / fs
            msg_a = _make_msg(np.ones((chunk, 2)) * i, fs=fs, offset=offset)
            assert proc(msg_a) is None

        b_offset = 1 * chunk / fs
        msg_b = _make_msg(np.ones((chunk, 3)) * 99, fs=fs, offset=b_offset)
        pair = proc.push_b(msg_b)
        assert pair is not None
        aa_a, aa_b = pair
        assert aa_a.data.shape[0] == chunk
        assert aa_b.data.shape[0] == chunk
        np.testing.assert_allclose(aa_a.data, 1.0)
        np.testing.assert_allclose(aa_b.data, 99.0)


class TestFloatingPointOffset:
    def test_tiny_epsilon(self):
        settings = AlignAlongAxisSettings(axis="time")
        proc = AlignAlongAxisProcessor(settings)
        fs = 1000.0
        n = 20
        offset = 1.0
        eps = 1e-14

        msg_a = _make_msg(np.ones((n, 2)), fs=fs, offset=offset)
        msg_b = _make_msg(np.ones((n, 2)) * 2, fs=fs, offset=offset + eps)
        proc(msg_a)
        pair = proc.push_b(msg_b)
        assert pair is not None
        assert pair[0].data.shape == (n, 2)


class TestGainMismatch:
    def test_b_different_gain_resets(self):
        settings = AlignAlongAxisSettings(axis="time")
        proc = AlignAlongAxisProcessor(settings)
        n = 10

        msg_a = _make_msg(np.ones((n, 2)), fs=100.0)
        msg_b_bad = _make_msg(np.ones((n, 3)), fs=200.0)
        proc(msg_a)
        assert proc.push_b(msg_b_bad) is None

        msg_a2 = _make_msg(np.ones((n, 2)) * 2, fs=200.0, offset=0.0)
        pair = proc(msg_a2)
        assert pair is not None
        assert pair[0].data.shape == (n, 2)
        assert pair[1].data.shape == (n, 3)

    def test_a_gain_change(self):
        settings = AlignAlongAxisSettings(axis="time")
        proc = AlignAlongAxisProcessor(settings)
        n = 5

        msg_a1 = _make_msg(np.ones((n, 2)), fs=100.0)
        msg_b1 = _make_msg(np.ones((n, 3)), fs=100.0)
        proc(msg_a1)
        assert proc.push_b(msg_b1) is not None

        # A switches to 200 Hz — triggers reset.
        msg_a2 = _make_msg(np.ones((n, 2)) * 2, fs=200.0, offset=0.5)
        assert proc(msg_a2) is None

        msg_b2 = _make_msg(np.ones((n, 3)) * 3, fs=200.0, offset=0.5)
        pair = proc.push_b(msg_b2)
        assert pair is not None

    def test_b_gain_change(self):
        settings = AlignAlongAxisSettings(axis="time")
        proc = AlignAlongAxisProcessor(settings)
        n = 5

        msg_a1 = _make_msg(np.ones((n, 2)), fs=100.0)
        msg_b1 = _make_msg(np.ones((n, 3)), fs=100.0)
        proc(msg_a1)
        assert proc.push_b(msg_b1) is not None

        msg_b2 = _make_msg(np.ones((n, 3)) * 2, fs=200.0, offset=0.5)
        assert proc.push_b(msg_b2) is None

        msg_a2 = _make_msg(np.ones((n, 2)) * 3, fs=200.0, offset=0.5)
        pair = proc(msg_a2)
        assert pair is not None


class TestShapeChange:
    def test_a_shape_change_resets_a(self):
        settings = AlignAlongAxisSettings(axis="time")
        proc = AlignAlongAxisProcessor(settings)
        fs = 100.0
        n = 5

        msg_a1 = _make_msg(np.ones((n, 2)), fs=fs, offset=0.0)
        msg_b1 = _make_msg(np.ones((n, 3)), fs=fs, offset=0.0)
        proc(msg_a1)
        pair = proc.push_b(msg_b1)
        assert pair is not None

        # Push B at next offset.
        offset_1 = n / fs
        msg_b2 = _make_msg(np.ones((n, 3)) * 2, fs=fs, offset=offset_1)
        assert proc.push_b(msg_b2) is None

        # A changes to 4 channels at same offset — shape change, re-alignment.
        msg_a2 = _make_msg(np.ones((n, 4)) * 3, fs=fs, offset=offset_1, ch_labels=["A0", "A1", "A2", "A3"])
        pair = proc(msg_a2)
        assert pair is not None
        assert pair[0].data.shape == (n, 4)
        assert pair[1].data.shape == (n, 3)
