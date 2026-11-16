"""When a stateful processor resets, now that the base class decides by default.

Most processors no longer implement ``_hash_message`` at all: the base class
folds in the message key, the dims, the length of every dimension except the one
the stream is chunked along, the coordinate values on those dimensions and the
gain and offset of any linear axis among them. These tests pin the behaviour
that fell out of removing those overrides, and the ``stream_dim`` bookkeeping the
default depends on.
"""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, compute_config_id

from ezmsg.sigproc.aggregate import AggregateSettings, AggregateTransformer, AggregationFunction
from ezmsg.sigproc.butterworthfilter import ButterworthFilterSettings, ButterworthFilterTransformer
from ezmsg.sigproc.filterbank import FilterbankSettings, FilterbankTransformer
from ezmsg.sigproc.flatten import FlattenSettings, FlattenTransformer
from ezmsg.sigproc.spectrum import SpectrumSettings, SpectrumTransformer
from ezmsg.sigproc.wavelets import CWTSettings, CWTTransformer
from ezmsg.sigproc.window import WindowSettings, WindowTransformer

FS = 100.0


def _msg(data, labels, fs=FS, key="dev", stream_dim="time"):
    return AxisArray(
        data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key=key,
        stream_dim=stream_dim,
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


class TestStreamDimBookkeeping:
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
        assert out.stream_dim == "win"

    def test_batcher_mode_keeps_the_incoming_stream_dim(self):
        """Batcher mode adds no axis: windows tile the target axis."""
        proc = WindowTransformer(
            WindowSettings(axis="time", newaxis=None, window_dur=0.2, window_shift=0.2, batch_windows=True)
        )
        out = proc(self._stream())
        assert "win" not in out.dims
        assert out.stream_dim == "time"

    def test_spectrum_clears_it_when_it_consumes_it(self):
        """`time` becomes `freq`: each output is one spectrum, nothing appends."""
        out = SpectrumTransformer(SpectrumSettings(axis="time"))(self._stream())
        assert "time" not in out.dims
        assert out.stream_dim is None

    def test_spectrum_keeps_a_windowed_stream_dim(self):
        """Windowed input still appends along `win` after the transform."""
        win = WindowTransformer(WindowSettings(axis="time", newaxis="win", window_dur=0.2, window_shift=0.1))(
            self._stream()
        )
        out = SpectrumTransformer(SpectrumSettings(axis="time"))(win)
        assert out.stream_dim == "win"
        assert "win" in out.dims

    def test_aggregate_clears_it_when_reducing_it(self):
        out = AggregateTransformer(AggregateSettings(axis="time", operation=AggregationFunction.MEAN))(self._stream())
        assert "time" not in out.dims
        assert out.stream_dim is None

    def test_flatten_follows_a_renamed_preserve_axis(self):
        proc = FlattenTransformer(
            FlattenSettings(preserve_axis="time", sample_axis="sample", flatten_axes=("ch",), output_axis="ch")
        )
        out = proc(self._stream())
        assert out.stream_dim == "sample"

    @pytest.mark.parametrize("declared", ["time", None])
    def test_a_declaration_is_optional(self, declared):
        """Undeclared messages still work -- the base class falls back."""
        rng = np.random.default_rng(4)
        msg = _msg(rng.standard_normal((32, 2)), ["a", "b"], stream_dim=declared)
        out = ButterworthFilterTransformer(ButterworthFilterSettings(axis="time", order=2, cuton=1.0, cutoff=20.0))(msg)
        assert out.stream_dim == declared


class TestChangesTheDefaultNowCovers:
    """Spectrum, CWT and Filterbank no longer override ``_hash_message``.

    A dtype change on a declared stream changes ``config_id``, which the default
    resets on; a transform-length change is handled in ``Spectrum._process``.
    """

    @staticmethod
    def _declared(data, labels=("a", "b")):
        m = _msg(data, list(labels))
        m.config_id = compute_config_id(m)
        return m

    def test_spectrum_resets_on_a_real_to_complex_change(self):
        """A complex input takes a different branch and a different freq axis."""
        rng = np.random.default_rng(5)
        real = self._declared(rng.standard_normal((64, 2)))
        cplx = self._declared(rng.standard_normal((64, 2)) + 1j * rng.standard_normal((64, 2)))
        assert real.config_id != cplx.config_id

        proc = SpectrumTransformer(SpectrumSettings(axis="time"))
        proc(real)
        before = proc._hash
        got = proc(cplx)
        assert proc._hash != before

        want = SpectrumTransformer(SpectrumSettings(axis="time"))(cplx)
        np.testing.assert_allclose(got.data, want.data)
        np.testing.assert_allclose(
            got.axes["freq"].value(np.arange(got.data.shape[0])), want.axes["freq"].value(np.arange(want.data.shape[0]))
        )

    def test_spectrum_follows_a_transform_length_change(self):
        """The FFT is sized by the stream dimension, which config_id leaves out."""
        rng = np.random.default_rng(6)
        proc = SpectrumTransformer(SpectrumSettings(axis="time"))
        short = self._declared(rng.standard_normal((64, 2)))
        long = self._declared(rng.standard_normal((128, 2)))
        assert short.config_id == long.config_id
        proc(short)
        got = proc(long)
        want = SpectrumTransformer(SpectrumSettings(axis="time"))(long)
        assert got.data.shape[got.get_axis_idx("freq")] == want.data.shape[want.get_axis_idx("freq")]
        np.testing.assert_allclose(got.data, want.data)

    def test_cwt_resets_on_a_real_to_complex_change(self):
        rng = np.random.default_rng(7)
        real = self._declared(rng.standard_normal((64, 2)))
        cplx = self._declared(rng.standard_normal((64, 2)) + 1j * rng.standard_normal((64, 2)))

        def make():
            return CWTTransformer(CWTSettings(frequencies=np.array([10.0, 20.0]), wavelet="morl", axis="time"))

        proc = make()
        proc(real)
        got = proc(cplx)
        want = make()(cplx)
        assert np.iscomplexobj(got.data)
        np.testing.assert_allclose(got.data, want.data)

    def test_filterbank_resets_on_a_real_to_complex_change(self):
        rng = np.random.default_rng(8)
        real = self._declared(rng.standard_normal((64, 2)))
        cplx = self._declared(rng.standard_normal((64, 2)) + 1j * rng.standard_normal((64, 2)))
        kernels = [np.ones(5) / 5, np.array([1.0, -1.0])]

        def make():
            return FilterbankTransformer(FilterbankSettings(kernels=kernels, axis="time"))

        proc = make()
        proc(real)
        got = proc(cplx)
        want = make()(cplx)
        assert np.iscomplexobj(got.data)
        np.testing.assert_allclose(got.data, want.data)
