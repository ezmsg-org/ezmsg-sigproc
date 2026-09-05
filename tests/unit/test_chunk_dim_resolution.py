"""The axis a processor operates on, when the settings do not name one.

Every stage used to guess positionally -- ``dims[0]`` for "the streaming axis",
``dims[-1]`` for "the channel axis". Both are positions rather than meanings,
and ``AxisArray.chunk_dim`` is the producer's declaration of which dimension
messages actually accumulate along. These tests pin the three resolution rules
(:mod:`ezmsg.sigproc.util.message`) and then check that the stages that carry
state between messages really follow the declaration.

The fixtures are deliberately transposed or windowed, because that is the only
place the old guess and the new rule disagree.
"""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.sigproc.util.message import (
    resolve_chunk_dim,
    resolve_configured_chunk_dim,
    resolve_feature_dim,
    resolve_transform_dim,
)

FS = 100.0


def _ch_axis(n):
    return CoordinateAxis(data=np.array([f"ch{i}" for i in range(n)]), dims=["ch"])


def transposed(n_ch=3, n_time=8, chunk_dim="time"):
    """``(ch, time)``: the first dim is static, the accumulating one is second."""
    kwargs = {"chunk_dim": chunk_dim} if chunk_dim else {}
    return AxisArray(
        np.arange(n_ch * n_time, dtype=float).reshape(n_ch, n_time),
        dims=["ch", "time"],
        axes={"ch": _ch_axis(n_ch), "time": AxisArray.TimeAxis(fs=FS)},
        key="dev",
        **kwargs,
    )


def windowed(n_win=4, n_time=8, n_ch=3):
    """``(win, time, ch)``: ``win`` accumulates, ``time`` is within-window."""
    return AxisArray(
        np.zeros((n_win, n_time, n_ch), dtype=float),
        dims=["win", "time", "ch"],
        axes={
            "win": AxisArray.TimeAxis(fs=FS / n_time),
            "time": AxisArray.TimeAxis(fs=FS),
            "ch": _ch_axis(n_ch),
        },
        key="dev",
        chunk_dim="win",
    )


def raw(n_time=8, n_ch=3, chunk_dim="time"):
    kwargs = {"chunk_dim": chunk_dim} if chunk_dim else {}
    return AxisArray(
        np.zeros((n_time, n_ch), dtype=float),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=FS), "ch": _ch_axis(n_ch)},
        key="dev",
        **kwargs,
    )


class TestResolveChunkDim:
    def test_the_declaration_wins_over_position(self):
        msg = transposed()
        assert msg.dims[0] != msg.chunk_dim, "the fixture must distinguish the two"
        assert resolve_chunk_dim(msg) == "time"

    def test_it_follows_a_windowing_stage_onto_win(self):
        assert resolve_chunk_dim(windowed()) == "win"

    def test_an_undeclared_chunk_dim_falls_back_to_streaming_dims(self):
        assert resolve_chunk_dim(transposed(chunk_dim=None)) == "time"

    def test_the_streaming_dims_fallback_is_configurable(self):
        msg = windowed()
        object.__setattr__(msg, "chunk_dim", None)
        assert resolve_chunk_dim(msg, ("win",)) == "win"

    def test_dims_zero_is_the_last_resort_only(self):
        """Nothing declared and nothing recognised: the position is all there is."""
        msg = AxisArray(np.zeros((4, 2)), dims=["a", "b"], key="dev")
        assert resolve_chunk_dim(msg) == "a"


class TestResolveFeatureDim:
    def test_it_skips_the_chunk_dim_on_a_transposed_stream(self):
        """``dims[-1]`` here is ``time``. A slicer defaulting to it would drop
        samples, and an affine transform would matmul across time."""
        msg = transposed()
        assert msg.dims[-1] == msg.chunk_dim, "the fixture must make the naive guess wrong"
        assert resolve_feature_dim(msg) == "ch"

    def test_it_is_unchanged_on_a_conventional_stream(self):
        assert resolve_feature_dim(raw()) == "ch"

    def test_position_zero_skips_the_chunk_dim_too(self):
        """RangedAggregate's case: ``dims[0]`` is usually the chunk dim, which is
        the worst possible default for an axis that must carry band values."""
        assert resolve_feature_dim(windowed(), 0) == "time"

    def test_a_chunk_only_message_falls_back_rather_than_raising(self):
        msg = AxisArray(np.zeros(8), dims=["time"], axes={"time": AxisArray.TimeAxis(fs=FS)}, chunk_dim="time")
        assert resolve_feature_dim(msg) == "time"


class TestResolveTransformDim:
    def test_windowed_input_transforms_within_the_window(self):
        """``win`` accumulates, but each window's spectrum is over ``time``."""
        assert resolve_transform_dim(windowed()) == "time"

    def test_raw_input_falls_through_to_the_chunk_dim(self):
        """``ch`` carries a CoordinateAxis, not a LinearAxis, so it is not a
        candidate and the rule lands back on ``time``."""
        assert resolve_transform_dim(raw()) == "time"

    def test_it_holds_under_transposition(self):
        assert resolve_transform_dim(transposed()) == "time"


class TestResolveConfiguredChunkDim:
    def test_an_explicit_axis_still_wins(self):
        """The escape hatch stays open: an explicit axis is an instruction."""

        class Proc:
            STREAMING_DIMS = ("time",)

        assert resolve_configured_chunk_dim(Proc(), windowed(), "time") == "time"

    def test_a_disagreement_warns_once(self, caplog):
        class Proc:
            STREAMING_DIMS = ("time",)

        proc = Proc()
        with caplog.at_level("WARNING"):
            for _ in range(3):
                resolve_configured_chunk_dim(proc, windowed(), "time")
        assert sum("chunk_dim" in r.message for r in caplog.records) == 1

    def test_agreement_is_silent(self, caplog):
        class Proc:
            STREAMING_DIMS = ("time",)

        with caplog.at_level("WARNING"):
            resolve_configured_chunk_dim(Proc(), raw(), "time")
        assert not caplog.records

    def test_a_mere_guess_never_warns(self, caplog):
        """Warning against STREAMING_DIMS rather than a declaration would fire on
        every correctly-configured windowed pipeline whose producer is silent."""

        class Proc:
            STREAMING_DIMS = ("time",)

        msg = windowed()
        object.__setattr__(msg, "chunk_dim", None)
        with caplog.at_level("WARNING"):
            resolve_configured_chunk_dim(Proc(), msg, "win")
        assert not caplog.records


class TestStagesFollowTheDeclaration:
    """The point of the exercise: state carried between messages must be carried
    along the dimension that actually grows."""

    def test_window_buffers_along_the_declared_dim(self):
        from ezmsg.sigproc.window import WindowSettings, WindowTransformer

        proc = WindowTransformer(WindowSettings(window_dur=0.04, window_shift=0.02))
        out = proc(transposed(n_ch=3, n_time=8))
        # 8 samples of a 4-sample window shifting by 2 -> windows along `win`,
        # each holding 4 time samples and all 3 channels.
        assert "win" in out.dims
        assert out.data.shape[out.get_axis_idx("time")] == 4
        assert out.data.shape[out.get_axis_idx("ch")] == 3

    def test_scaler_accumulates_along_the_declared_dim(self):
        from ezmsg.sigproc.scaler import (
            AdaptiveStandardScalerSettings,
            AdaptiveStandardScalerTransformer,
        )

        proc = AdaptiveStandardScalerTransformer(AdaptiveStandardScalerSettings(time_constant=1.0))
        out = proc(transposed(n_ch=3, n_time=8))
        assert out.data.shape == (3, 8)
        assert out.dims == ["ch", "time"]

    def test_scaler_looks_up_the_axis_index_rather_than_assuming_zero(self):
        """The old code hardcoded ``axis_idx = 0`` whenever ``axis`` was unset,
        which silently transposed the data on a ``(ch, time)`` stream."""
        from ezmsg.sigproc.scaler import (
            RiverAdaptiveStandardScalerSettings,
            RiverAdaptiveStandardScalerTransformer,
        )

        pytest.importorskip("river")
        proc = RiverAdaptiveStandardScalerTransformer(RiverAdaptiveStandardScalerSettings(time_constant=1.0))
        proc(transposed(n_ch=3, n_time=8))
        assert proc.state.axis == "time"
        assert proc.state.axis_idx == 1

    def test_filter_carries_zi_along_the_declared_dim(self):
        from ezmsg.sigproc.butterworthfilter import (
            ButterworthFilterSettings,
            ButterworthFilterTransformer,
        )

        proc = ButterworthFilterTransformer(ButterworthFilterSettings(order=2, cuton=None, cutoff=20.0, coef_type="ba"))
        first = proc(transposed(n_ch=3, n_time=8))
        second = proc(transposed(n_ch=3, n_time=8))
        assert first.data.shape == second.data.shape == (3, 8)
        # zi is per-channel, so it has one entry per channel and the two chunks
        # differ: continuity was carried across the message boundary.
        assert not np.allclose(first.data, second.data)

    def test_diff_carries_the_previous_sample_along_the_declared_dim(self):
        from ezmsg.sigproc.diff import DiffSettings, DiffTransformer

        proc = DiffTransformer(DiffSettings())
        out = proc(transposed(n_ch=3, n_time=8))
        assert out.data.shape == (3, 8), "diff prepends the carried sample, preserving length"
        # Row-major arange over (3, 8): consecutive samples along `time` differ
        # by 1, so every within-row difference is 1.
        assert np.allclose(out.data[:, 1:], 1.0)


class TestSpectrumPicksTheTransformAxis:
    def test_windowed_input_transforms_time_not_win(self):
        """Previously ``dims[0]`` picked ``win`` here, FFT-ing across windows."""
        from ezmsg.sigproc.spectrum import SpectrumSettings, SpectrumTransformer

        out = SpectrumTransformer(SpectrumSettings())(windowed(n_win=4, n_time=8, n_ch=3))
        assert "freq" in out.dims
        assert out.dims.index("freq") == 1, "the freq axis replaces `time`, not `win`"
        assert out.data.shape[0] == 4, "the win dimension survives"

    def test_windowed_output_keeps_accumulating_along_win(self):
        from ezmsg.sigproc.spectrum import SpectrumSettings, SpectrumTransformer

        out = SpectrumTransformer(SpectrumSettings())(windowed())
        assert out.chunk_dim == "win"

    def test_raw_input_still_transforms_time(self):
        from ezmsg.sigproc.spectrum import SpectrumSettings, SpectrumTransformer

        out = SpectrumTransformer(SpectrumSettings())(raw(n_time=16, n_ch=3))
        assert "freq" in out.dims
        assert "time" not in out.dims
        assert out.chunk_dim is None, "the transformed axis is consumed"


class TestFeatureStagesSkipTheChunkDim:
    def test_slicer_defaults_to_channels_on_a_transposed_stream(self):
        from ezmsg.sigproc.slicer import SlicerSettings, SlicerTransformer

        out = SlicerTransformer(SlicerSettings(selection="0:2"))(transposed(n_ch=3, n_time=8))
        assert out.data.shape == (2, 8), "channels sliced, samples untouched"

    def test_affine_transform_defaults_to_channels_on_a_transposed_stream(self):
        from ezmsg.sigproc.affinetransform import (
            AffineTransformSettings,
            AffineTransformTransformer,
        )

        weights = np.eye(3) * 2.0
        out = AffineTransformTransformer(AffineTransformSettings(weights=weights))(transposed(n_ch=3, n_time=8))
        assert out.data.shape == (3, 8)
        assert np.allclose(out.data, transposed(n_ch=3, n_time=8).data * 2.0)
