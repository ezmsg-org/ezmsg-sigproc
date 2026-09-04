import copy

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from frozendict import frozendict

from ezmsg.sigproc.downsample import DownsampleSettings, DownsampleTransformer
from tests.helpers.empty_time import check_empty_result, check_state_not_corrupted, make_empty_msg, make_msg
from tests.helpers.util import assert_messages_equal, requires_mlx


@pytest.mark.parametrize("block_size", [1, 5, 10, 20])
@pytest.mark.parametrize("target_rate", [19.0, 9.5, 6.3])
@pytest.mark.parametrize("factor", [None, 1, 2])
def test_downsample_core(block_size: int, target_rate: float, factor: int | None):
    in_fs = 19.0
    test_dur = 4.0
    n_channels = 2
    n_features = 3
    num_samps = int(np.ceil(test_dur * in_fs))
    num_msgs = int(np.ceil(num_samps / block_size))
    sig = np.arange(num_samps * n_channels * n_features).reshape(num_samps, n_channels, n_features)
    # tvec = np.arange(num_samps) / in_fs

    def msg_generator():
        for msg_ix in range(num_msgs):
            msg_sig = sig[msg_ix * block_size : (msg_ix + 1) * block_size]
            msg_idx: float = msg_sig[0, 0, 0] / (n_channels * n_features)
            msg_offs = msg_idx / in_fs
            msg = AxisArray(
                data=msg_sig,
                dims=["time", "ch", "feat"],
                axes=frozendict(
                    {
                        "time": AxisArray.TimeAxis(fs=in_fs, offset=msg_offs),
                        "ch": AxisArray.CoordinateAxis(data=np.arange(n_channels).astype(str), dims=["ch"]),
                        "feat": AxisArray.CoordinateAxis(
                            data=np.array([f"Feat{_ + 1}" for _ in range(n_features)]),
                            dims=["feat"],
                        ),
                    }
                ),
                key="test_downsample_core",
            )
            yield msg

    in_msgs = list(msg_generator())
    backup = [copy.deepcopy(msg) for msg in in_msgs]

    proc = DownsampleTransformer(target_rate=target_rate, factor=factor)
    out_msgs = []
    for msg in in_msgs:
        res = proc(msg)
        if res.data.size:
            out_msgs.append(res)

    assert_messages_equal(in_msgs, backup)

    # Assert correctness of gain
    expected_factor: int = int(in_fs // target_rate) if factor is None else factor
    assert all(msg.axes["time"].gain == expected_factor / in_fs for msg in out_msgs)

    # Assert messages have the correct timestamps
    expected_offsets = np.cumsum([0] + [_.data.shape[0] for _ in out_msgs]) * expected_factor / in_fs
    actual_offsets = np.array([_.axes["time"].offset for _ in out_msgs])
    assert np.allclose(actual_offsets, expected_offsets[:-1])

    # Compare returned values to expected values.
    allres_msg = AxisArray.concatenate(*out_msgs, dim="time")
    assert np.array_equal(allres_msg.data, sig[::expected_factor])


def test_downsample_empty_after_init():
    from ezmsg.sigproc.downsample import DownsampleTransformer

    proc = DownsampleTransformer(factor=2)
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_downsample_empty_target_rate():
    from ezmsg.sigproc.downsample import DownsampleTransformer

    proc = DownsampleTransformer(target_rate=25.0)
    normal = make_msg()
    empty = make_empty_msg()
    _ = proc(normal)
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


def test_downsample_empty_first():
    from ezmsg.sigproc.downsample import DownsampleTransformer

    proc = DownsampleTransformer(factor=2)
    empty = make_empty_msg()
    normal = make_msg()
    result = proc(empty)
    check_empty_result(result)
    check_state_not_corrupted(proc, normal)


@requires_mlx
@pytest.mark.parametrize("factor", [1, 2, 4])
@pytest.mark.parametrize("block_size", [7, 13, 30])
def test_downsample_mlx_matches_numpy(factor: int, block_size: int):
    """Downsampling must work on MLX arrays and agree sample-for-sample with NumPy.

    The selection used to be an integer index array, which MLX rejects outright
    ("Cannot index mlx array using the given type"), so this path was entirely
    broken on MLX. It is now a strided slice.
    """
    import mlx.core as mx

    fs = 100.0
    n_samples, n_ch = 97, 4
    rng = np.random.default_rng(0)
    src = rng.standard_normal((n_samples, n_ch)).astype(np.float32)

    def stream(data):
        proc = DownsampleTransformer(factor=factor)
        kept = []
        for start in range(0, n_samples, block_size):
            chunk = data[start : start + block_size]
            msg = AxisArray(
                chunk,
                dims=["time", "ch"],
                axes=frozendict({"time": AxisArray.TimeAxis(fs=fs, offset=start / fs)}),
                key="ds",
            )
            out = proc(msg)
            if out.data.shape[0]:
                kept.append((np.asarray(out.data), out.axes["time"].offset))
        return kept

    mlx_kept = stream(mx.array(src))
    np_kept = stream(src)

    assert [d.shape for d, _ in mlx_kept] == [d.shape for d, _ in np_kept]
    for (mlx_d, mlx_off), (np_d, np_off) in zip(mlx_kept, np_kept, strict=True):
        assert np.array_equal(mlx_d, np_d)
        assert mlx_off == pytest.approx(np_off)
    # And the concatenation is exactly every `factor`-th input sample.
    assert np.array_equal(np.concatenate([d for d, _ in mlx_kept]), src[::factor])


class TestTheDimensionIsAlwaysTheChunkDimension:
    """``s_idx`` carries across messages so the kept samples form one arithmetic
    sequence over the whole stream rather than restarting per chunk. That is the
    point along the accumulating dimension and meaningless along any other, so
    there is no setting to get wrong -- the dimension comes from the message.
    """

    @staticmethod
    def _msg(i, chunk_dim="time", n_freq=5, n_time=4):
        data = np.arange(n_freq, dtype=float)[None, :] + i * 100
        kwargs = {"chunk_dim": chunk_dim} if chunk_dim else {}
        return AxisArray(
            np.tile(data, (n_time, 1)),
            dims=["time", "freq"],
            axes={
                "time": AxisArray.TimeAxis(fs=100.0, offset=i * n_time / 100.0),
                "freq": AxisArray.LinearAxis(gain=2.0, offset=0.0),
            },
            key="dev",
            **kwargs,
        )

    def test_settings_no_longer_carry_an_axis(self):
        """Removed rather than validated: a rejected setting is still a setting
        someone has to understand before they can not use it."""
        assert "axis" not in DownsampleSettings.__dataclass_fields__
        with pytest.raises(TypeError):
            DownsampleSettings(axis="freq", factor=2)

    def test_it_follows_the_declared_chunk_dim(self):
        proc = DownsampleTransformer(DownsampleSettings(factor=2))
        out = proc(self._msg(0))
        assert proc.state.axis == "time"
        assert out.data.shape == (2, 5)

    def test_the_static_axis_is_left_untouched(self):
        proc = DownsampleTransformer(DownsampleSettings(factor=2))
        out = proc(self._msg(0))
        assert out.data.shape[1] == 5, "freq must pass through whole"
        assert out.axes["freq"].gain == 2.0

    def test_what_a_configurable_axis_would_have_allowed(self):
        """Decimating a static axis rotates the selection between messages,
        because the phase counter carries across chunk boundaries. Slicer is the
        tool for that job and selects the same elements every time."""
        from ezmsg.sigproc.slicer import SlicerSettings, SlicerTransformer

        sliced = SlicerTransformer(SlicerSettings(selection="::2", axis="freq"))
        kept = [(sliced(self._msg(i)).data[0] - i * 100).astype(int).tolist() for i in range(4)]
        assert kept == [[0, 2, 4]] * 4

    def test_an_undeclared_chunk_dim_falls_back_to_streaming_dims(self):
        proc = DownsampleTransformer(DownsampleSettings(factor=2))
        proc(self._msg(0, chunk_dim=None))
        assert proc.state.axis == DownsampleTransformer.STREAMING_DIMS[0] == "time"


class TestTheDimensionFollowsTheMessage:
    def test_it_follows_a_windowing_stage_onto_win(self):
        """The payoff for removing the setting: a Downsample after a Window
        decimates *windows* with no reconfiguration. Left at the old ``"time"``
        default it would have decimated within each window instead."""
        from ezmsg.sigproc.window import WindowSettings, WindowTransformer

        windowed = WindowTransformer(WindowSettings(axis="time", newaxis="win", window_dur=0.02, window_shift=0.01))(
            AxisArray(
                np.zeros((40, 3), np.float32),
                dims=["time", "ch"],
                axes={"time": AxisArray.TimeAxis(fs=100.0)},
                key="dev",
                chunk_dim="time",
            )
        )
        assert windowed.chunk_dim == "win"

        proc = DownsampleTransformer(DownsampleSettings(factor=2))
        out = proc(windowed)
        assert proc.state.axis == "win"
        assert out.data.shape[1:] == windowed.data.shape[1:], "the window contents must be untouched"

    def test_it_uses_the_declaration_not_the_first_dimension(self):
        """``dims[0]`` is a tempting stand-in and a wrong one. A transposed
        stream is ``(ch, time)``: the first dimension is static and the one that
        accumulates is second."""
        msg = AxisArray(
            np.arange(24, dtype=float).reshape(3, 8),
            dims=["ch", "time"],
            axes={
                "ch": CoordinateAxis(data=np.array(["a", "b", "c"]), dims=["ch"]),
                "time": AxisArray.TimeAxis(fs=100.0),
            },
            key="dev",
            chunk_dim="time",
        )
        assert msg.dims[0] != msg.chunk_dim, "the fixture must distinguish the two"

        proc = DownsampleTransformer(DownsampleSettings(factor=2))
        out = proc(msg)
        assert proc.state.axis == "time"
        assert out.data.shape == (3, 4)
