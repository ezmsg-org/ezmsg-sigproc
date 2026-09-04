"""When a stateful processor resets, now that the base class decides by default.

Most processors no longer implement ``_hash_message`` at all: the base class
folds in the message key, the dims, the length of every dimension except the one
the stream is chunked along, the coordinate values on those dimensions and the
gain and offset of any linear axis among them. These tests pin the behaviour
that fell out of removing those overrides, and the ``chunk_dim`` bookkeeping the
default depends on.
"""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.sigproc.aggregate import AggregateSettings, AggregateTransformer, AggregationFunction
from ezmsg.sigproc.butterworthfilter import ButterworthFilterSettings, ButterworthFilterTransformer
from ezmsg.sigproc.flatten import FlattenSettings, FlattenTransformer
from ezmsg.sigproc.spectrum import SpectrumSettings, SpectrumTransformer
from ezmsg.sigproc.window import WindowSettings, WindowTransformer

FS = 100.0


def _msg(data, labels, fs=FS, key="dev", chunk_dim="time"):
    return AxisArray(
        data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key=key,
        chunk_dim=chunk_dim,
    )


class TestPerChannelStateFollowsTheChannels:
    """A filter's `zi` belongs to the channels it was built for.

    This is what the sweep was for. A device reconfigured mid-session sends the
    same number of channels under different labels; keying the state on shape
    alone left the new channels being filtered through the old ones' history.
    """

    @staticmethod
    def _filter():
        return ButterworthFilterTransformer(ButterworthFilterSettings(axis="time", order=2, cuton=1.0, cutoff=20.0))

    def test_relabel_at_fixed_channel_count_resets(self):
        rng = np.random.default_rng(0)
        warmup = rng.standard_normal((50, 2)) * 10.0
        fresh = rng.standard_normal((20, 2)) * 0.01

        carried = self._filter()
        carried(_msg(warmup, ["armA-1", "armA-2"]))
        got = carried(_msg(fresh, ["armB-1", "armB-2"]))

        # What a correctly reset filter produces for the same input.
        want = self._filter()(_msg(fresh, ["armB-1", "armB-2"]))
        np.testing.assert_allclose(got.data, want.data)

    def test_chunk_size_jitter_does_not_reset(self):
        """The filter must carry state across chunks, or it would ring forever."""
        rng = np.random.default_rng(1)
        proc = self._filter()
        proc(_msg(rng.standard_normal((50, 2)), ["a", "b"]))
        # FilterByDesign delegates to an inner FilterTransformer that owns `zi`.
        state_before = proc._state.filter._state.zi.copy()
        proc(_msg(rng.standard_normal((17, 2)), ["a", "b"]))
        assert not np.array_equal(proc._state.filter._state.zi, state_before), (
            "the filter should have advanced its state across the chunk, not reset it"
        )

    def test_channel_count_change_resets(self):
        rng = np.random.default_rng(2)
        proc = self._filter()
        proc(_msg(rng.standard_normal((50, 2)), ["a", "b"]))
        out = proc(_msg(rng.standard_normal((50, 3)), ["a", "b", "c"]))
        assert out.data.shape[1] == 3


class TestChunkDimBookkeeping:
    """Every operation that renames, consumes or invents the chunked dimension
    has to say so, or the base class excludes the wrong one."""

    @staticmethod
    def _stream(n_time=64, n_ch=2):
        rng = np.random.default_rng(3)
        return _msg(rng.standard_normal((n_time, n_ch)), [f"c{i}" for i in range(n_ch)])

    def test_window_declares_the_new_axis(self):
        """After windowing, messages append along `win`, not `time`."""
        proc = WindowTransformer(WindowSettings(axis="time", newaxis="win", window_dur=0.2, window_shift=0.1))
        out = proc(self._stream())
        assert out.dims[:2] == ["win", "time"]
        assert out.chunk_dim == "win"

    def test_batcher_mode_keeps_the_incoming_chunk_dim(self):
        """Batcher mode adds no axis: windows tile the target axis."""
        proc = WindowTransformer(
            WindowSettings(axis="time", newaxis=None, window_dur=0.2, window_shift=0.2, batch_windows=True)
        )
        out = proc(self._stream())
        assert "win" not in out.dims
        assert out.chunk_dim == "time"

    def test_spectrum_clears_it_when_it_consumes_it(self):
        """`time` becomes `freq`: each output is one spectrum, nothing appends."""
        out = SpectrumTransformer(SpectrumSettings(axis="time"))(self._stream())
        assert "time" not in out.dims
        assert out.chunk_dim is None

    def test_spectrum_keeps_a_windowed_chunk_dim(self):
        """Windowed input still appends along `win` after the transform."""
        win = WindowTransformer(WindowSettings(axis="time", newaxis="win", window_dur=0.2, window_shift=0.1))(
            self._stream()
        )
        out = SpectrumTransformer(SpectrumSettings(axis="time"))(win)
        assert out.chunk_dim == "win"
        assert "win" in out.dims

    def test_aggregate_clears_it_when_reducing_it(self):
        out = AggregateTransformer(AggregateSettings(axis="time", operation=AggregationFunction.MEAN))(self._stream())
        assert "time" not in out.dims
        assert out.chunk_dim is None

    def test_flatten_follows_a_renamed_preserve_axis(self):
        proc = FlattenTransformer(
            FlattenSettings(preserve_axis="time", sample_axis="sample", flatten_axes=("ch",), output_axis="ch")
        )
        out = proc(self._stream())
        assert out.chunk_dim == "sample"

    @pytest.mark.parametrize("declared", ["time", None])
    def test_a_declaration_is_optional(self, declared):
        """Undeclared messages still work -- the base class falls back."""
        rng = np.random.default_rng(4)
        msg = _msg(rng.standard_normal((32, 2)), ["a", "b"], chunk_dim=declared)
        out = ButterworthFilterTransformer(ButterworthFilterSettings(axis="time", order=2, cuton=1.0, cutoff=20.0))(msg)
        assert out.chunk_dim == declared


class TestOverridesThatRemain:
    """The three processors that still need something the default cannot know."""

    @staticmethod
    def _msg(dtype):
        rng = np.random.default_rng(5)
        return _msg(rng.standard_normal((64, 2)).astype(dtype), ["a", "b"])

    def test_spectrum_reacts_to_a_dtype_change(self):
        """A complex input takes a different branch and a different freq axis."""
        proc = SpectrumTransformer(SpectrumSettings(axis="time"))
        proc(self._msg(np.float32))
        before = proc._hash
        proc(self._msg(np.complex64))
        assert proc._hash != before

    def test_spectrum_reacts_to_a_transform_length_change(self):
        """The FFT is sized by the chunk dimension -- the one length the default
        ignores -- so Spectrum has to fold it back in."""
        proc = SpectrumTransformer(SpectrumSettings(axis="time"))
        rng = np.random.default_rng(6)
        proc(_msg(rng.standard_normal((64, 2)), ["a", "b"]))
        before = proc._hash
        proc(_msg(rng.standard_normal((128, 2)), ["a", "b"]))
        assert proc._hash != before
