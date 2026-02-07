import copy
import platform
import time

import numpy as np
import pytest
import scipy.fft as sp_fft
import scipy.signal as sps
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis

from ezmsg.sigproc.spectrum import (
    SpectralOutput,
    SpectralTransform,
    SpectrumSettings,
    SpectrumTransformer,
    WindowFunction,
)
from tests.helpers.empty_time import FS, N_CH
from tests.helpers.util import (
    assert_messages_equal,
    create_messages_with_periodic_signal,
)


def _debug_plot_welch(raw: AxisArray, result: AxisArray, welch_db: bool = True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1)

    t_ax = raw.axes["time"]
    t_vec = t_ax.value(np.arange(raw.data.shape[raw.get_axis_idx("time")]))
    ch0_raw = raw.data[..., :, 0]
    if ch0_raw.ndim > 1:
        # For multi-win inputs
        ch0_raw = ch0_raw[0]
    ax[0].plot(t_vec, ch0_raw)
    ax[0].set_xlabel("Time (s)")

    f_ax = result.axes["freq"]
    f_vec = f_ax.value(np.arange(result.data.shape[result.get_axis_idx("freq")]))
    ch0_spec = result.data[..., :, 0]
    if ch0_spec.ndim > 1:
        ch0_spec = ch0_spec[0]
    ax[1].plot(f_vec, ch0_spec, label="calculated", linewidth=2.0)
    ax[1].set_xlabel("Frequency (Hz)")

    f, Pxx = sps.welch(ch0_raw, fs=1 / raw.axes["time"].gain, window="hamming", nperseg=len(ch0_raw))
    if welch_db:
        Pxx = 10 * np.log10(Pxx)
    ax[1].plot(f, Pxx, label="welch", color="tab:orange", linestyle="--")
    ax[1].set_ylabel("dB" if welch_db else "V**2/Hz")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


@pytest.mark.parametrize("window", [WindowFunction.HANNING, WindowFunction.HAMMING])
@pytest.mark.parametrize("transform", [SpectralTransform.REL_DB, SpectralTransform.REL_POWER])
@pytest.mark.parametrize("output", [SpectralOutput.POSITIVE, SpectralOutput.NEGATIVE, SpectralOutput.FULL])
def test_spectrum_gen_multiwin(window: WindowFunction, transform: SpectralTransform, output: SpectralOutput):
    win_dur = 1.0
    win_step_dur = 0.5
    fs = 1000.0
    sin_params = [
        {"a": 1.0, "f": 10.0, "p": 0.0, "dur": 20.0},
        {"a": 0.5, "f": 20.0, "p": np.pi / 7, "dur": 20.0},
        {"a": 0.2, "f": 200.0, "p": np.pi / 11, "dur": 20.0},
    ]
    win_len = int(win_dur * fs)

    messages = create_messages_with_periodic_signal(
        sin_params=sin_params, fs=fs, msg_dur=win_dur, win_step_dur=win_step_dur
    )
    input_multiwin = AxisArray.concatenate(*messages, dim="win")
    input_multiwin.axes["win"] = AxisArray.TimeAxis(offset=0, fs=1 / win_step_dur)

    proc = SpectrumTransformer(SpectrumSettings(axis="time", window=window, transform=transform, output=output))
    result = proc(input_multiwin)
    # _debug_plot_welch(input_multiwin, result, welch_db=True)
    assert isinstance(result, AxisArray)
    assert "time" not in result.dims
    assert "time" not in result.axes
    assert "ch" in result.dims
    assert "win" in result.dims
    assert "ch" in result.axes  # We will not check anything else about axes["ch"].
    assert "freq" in result.axes
    assert result.axes["freq"].gain == 1 / win_dur
    assert "freq" in result.dims
    fax_ix = result.get_axis_idx("freq")
    f_len = win_len if output == SpectralOutput.FULL else (win_len // 2 + 1 - (win_len % 2))
    assert result.data.shape[fax_ix] == f_len
    f_vec = result.axes["freq"].value(np.arange(f_len))
    if output == SpectralOutput.NEGATIVE:
        f_vec = np.abs(f_vec)
    for s_p in sin_params:
        f_ix = np.argmin(np.abs(f_vec - s_p["f"]))
        peak_inds = np.argmax(
            slice_along_axis(result.data, slice(f_ix - 3, f_ix + 3), axis=fax_ix),
            axis=fax_ix,
        )
        assert np.all(peak_inds == 3)


@pytest.mark.parametrize("window", [WindowFunction.HANNING, WindowFunction.HAMMING])
@pytest.mark.parametrize("transform", [SpectralTransform.REL_DB, SpectralTransform.REL_POWER])
@pytest.mark.parametrize("output", [SpectralOutput.POSITIVE, SpectralOutput.NEGATIVE, SpectralOutput.FULL])
def test_spectrum_gen(window: WindowFunction, transform: SpectralTransform, output: SpectralOutput):
    win_dur = 1.0
    win_step_dur = 0.5
    fs = 1000.0
    sin_params = [
        {"a": 1.0, "f": 10.0, "p": 0.0, "dur": 20.0},
        {"a": 0.5, "f": 20.0, "p": np.pi / 7, "dur": 20.0},
        {"a": 0.2, "f": 200.0, "p": np.pi / 11, "dur": 20.0},
    ]
    messages = create_messages_with_periodic_signal(
        sin_params=sin_params, fs=fs, msg_dur=win_dur, win_step_dur=win_step_dur
    )
    backup = [copy.deepcopy(_) for _ in messages]

    proc = SpectrumTransformer(SpectrumSettings(axis="time", window=window, transform=transform, output=output))
    results = [proc(msg) for msg in messages]

    assert_messages_equal(messages, backup)

    assert "freq" in results[0].dims
    assert "ch" in results[0].dims
    assert "win" not in results[0].dims
    # _debug_plot_welch(messages[0], results[0], welch_db=True)


@pytest.mark.parametrize("complex", [False, True])
def test_spectrum_vs_sps_fft(complex: bool):
    # spectrum uses np.fft. Here we compare the output of spectrum against scipy.fft.fftn
    win_dur = 1.0
    win_step_dur = 0.5
    fs = 1000.0
    sin_params = [
        {"a": 1.0, "f": 10.0, "p": 0.0, "dur": 20.0},
        {"a": 0.5, "f": 20.0, "p": np.pi / 7, "dur": 20.0},
        {"a": 0.2, "f": 200.0, "p": np.pi / 11, "dur": 20.0},
    ]
    messages = create_messages_with_periodic_signal(
        sin_params=sin_params, fs=fs, msg_dur=win_dur, win_step_dur=win_step_dur
    )
    nfft = 1 << (messages[0].data.shape[0] - 1).bit_length()  # nextpow2

    proc = SpectrumTransformer(
        SpectrumSettings(
            axis="time",
            window=WindowFunction.NONE,
            transform=SpectralTransform.RAW_COMPLEX if complex else SpectralTransform.REAL,
            output=SpectralOutput.FULL if complex else SpectralOutput.POSITIVE,
            norm="backward",
            do_fftshift=False,
            nfft=nfft,
        )
    )
    results = [proc(msg) for msg in messages]
    test_spec = results[0].data
    if complex:
        sp_res = sp_fft.fft(messages[0].data, n=nfft, axis=0)
    else:
        sp_res = sp_fft.rfft(messages[0].data, n=nfft, axis=0).real
    assert np.allclose(test_spec, sp_res)


def test_spectrum_empty_after_init():
    """After processing normal windowed data, an empty-win message should pass through."""
    from frozendict import frozendict

    from ezmsg.sigproc.spectrum import SpectrumSettings, SpectrumTransformer

    window_samples = 10
    proc = SpectrumTransformer(SpectrumSettings(axis="time"))
    normal = AxisArray(
        data=np.random.randn(5, window_samples, N_CH).astype(np.float64),
        dims=["win", "time", "ch"],
        axes=frozendict(
            {
                "win": AxisArray.LinearAxis(gain=0.05, offset=0.0),
                "time": AxisArray.TimeAxis(fs=FS),
                "ch": AxisArray.CoordinateAxis(data=np.arange(N_CH).astype(str), dims=["ch"]),
            }
        ),
    )
    empty = AxisArray(
        data=np.random.randn(0, window_samples, N_CH).astype(np.float64),
        dims=["win", "time", "ch"],
        axes=frozendict(
            {
                "win": AxisArray.LinearAxis(gain=0.05, offset=0.0),
                "time": AxisArray.TimeAxis(fs=FS),
                "ch": AxisArray.CoordinateAxis(data=np.arange(N_CH).astype(str), dims=["ch"]),
            }
        ),
    )
    result1 = proc(normal)
    assert result1.data.shape[0] == 5
    result = proc(empty)
    assert result.data.shape[0] == 0
    assert result.data.shape[2] == N_CH
    result3 = proc(normal)
    assert result3.data.shape[0] == 5
    assert np.all(np.isfinite(result3.data))


def test_spectrum_empty_first():
    """Empty-win message as first input."""
    from frozendict import frozendict

    from ezmsg.sigproc.spectrum import SpectrumSettings, SpectrumTransformer

    window_samples = 10
    proc = SpectrumTransformer(SpectrumSettings(axis="time"))
    empty = AxisArray(
        data=np.random.randn(0, window_samples, N_CH).astype(np.float64),
        dims=["win", "time", "ch"],
        axes=frozendict(
            {
                "win": AxisArray.LinearAxis(gain=0.05, offset=0.0),
                "time": AxisArray.TimeAxis(fs=FS),
                "ch": AxisArray.CoordinateAxis(data=np.arange(N_CH).astype(str), dims=["ch"]),
            }
        ),
    )
    normal = AxisArray(
        data=np.random.randn(5, window_samples, N_CH).astype(np.float64),
        dims=["win", "time", "ch"],
        axes=frozendict(
            {
                "win": AxisArray.LinearAxis(gain=0.05, offset=0.0),
                "time": AxisArray.TimeAxis(fs=FS),
                "ch": AxisArray.CoordinateAxis(data=np.arange(N_CH).astype(str), dims=["ch"]),
            }
        ),
    )
    result = proc(empty)
    assert result.data.shape[0] == 0
    result2 = proc(normal)
    assert result2.data.shape[0] == 5
    assert np.all(np.isfinite(result2.data))


requires_apple_silicon = pytest.mark.skipif(
    platform.machine() != "arm64" or platform.system() != "Darwin",
    reason="Requires Apple Silicon for MLX",
)


@requires_apple_silicon
@pytest.mark.parametrize("transform", [SpectralTransform.REL_DB, SpectralTransform.REL_POWER])
@pytest.mark.parametrize("output", [SpectralOutput.POSITIVE, SpectralOutput.NEGATIVE, SpectralOutput.FULL])
def test_spectrum_mlx(transform: SpectralTransform, output: SpectralOutput):
    """SpectrumTransformer on MLX arrays should produce correct spectral peaks."""
    import mlx.core as mx

    win_dur = 1.0
    fs = 1000.0
    sin_params = [
        {"a": 1.0, "f": 10.0, "p": 0.0, "dur": 5.0},
        {"a": 0.5, "f": 20.0, "p": np.pi / 7, "dur": 5.0},
        {"a": 0.2, "f": 200.0, "p": np.pi / 11, "dur": 5.0},
    ]
    win_len = int(win_dur * fs)
    messages_np = create_messages_with_periodic_signal(sin_params=sin_params, fs=fs, msg_dur=win_dur, win_step_dur=None)
    msg_np_orig = messages_np[0]

    # Build an MLX message
    msg_mx = AxisArray(
        data=mx.array(msg_np_orig.data.astype(np.float32)),
        dims=msg_np_orig.dims,
        axes=msg_np_orig.axes,
    )

    settings = SpectrumSettings(axis="time", window=WindowFunction.HAMMING, transform=transform, output=output)

    proc_mx = SpectrumTransformer(settings)
    result = proc_mx(msg_mx)

    assert isinstance(result.data, mx.array), f"Expected mx.array, got {type(result.data)}"
    assert "freq" in result.dims
    assert "freq" in result.axes
    assert result.axes["freq"].gain == 1 / win_dur

    # Check correct spectral shape
    fax_ix = result.get_axis_idx("freq")
    f_len = win_len if output == SpectralOutput.FULL else (win_len // 2 + 1 - (win_len % 2))
    assert result.data.shape[fax_ix] == f_len

    # Verify peaks are at the expected frequencies
    result_np = np.asarray(result.data)
    f_vec = result.axes["freq"].value(np.arange(f_len))
    if output == SpectralOutput.NEGATIVE:
        f_vec = np.abs(f_vec)
    for s_p in sin_params:
        f_ix = np.argmin(np.abs(f_vec - s_p["f"]))
        peak_inds = np.argmax(
            slice_along_axis(result_np, slice(f_ix - 3, f_ix + 3), axis=fax_ix),
            axis=fax_ix,
        )
        assert np.all(peak_inds == 3), f"Peak not at expected freq {s_p['f']} Hz"


@requires_apple_silicon
def test_spectrum_mlx_benchmark():
    """Benchmark SpectrumTransformer: numpy vs MLX on multi-window input."""
    import mlx.core as mx

    fs = 1000.0
    win_dur = 1.0
    n_windows = 200
    n_channels = 256
    win_samples = int(win_dur * fs)

    rng = np.random.default_rng(42)
    np_data = rng.standard_normal((n_windows, win_samples, n_channels)).astype(np.float32)

    settings = SpectrumSettings(axis="time", window=WindowFunction.HAMMING, transform=SpectralTransform.REL_DB)

    msg_np = AxisArray(
        data=np_data,
        dims=["win", "time", "ch"],
        axes={
            "win": AxisArray.LinearAxis(gain=win_dur, offset=0.0),
            "time": AxisArray.TimeAxis(fs=fs),
        },
    )
    msg_mx = AxisArray(
        data=mx.array(np_data),
        dims=["win", "time", "ch"],
        axes=msg_np.axes,
    )

    # --- Numpy ---
    proc_np = SpectrumTransformer(settings)
    proc_np(msg_np)  # Warmup

    t0 = time.perf_counter()
    _ = proc_np(msg_np)
    t_numpy = time.perf_counter() - t0

    # --- MLX ---
    proc_mx = SpectrumTransformer(settings)
    warmup = proc_mx(msg_mx)
    mx.eval(warmup.data)  # Force compilation

    t0 = time.perf_counter()
    result_mx = proc_mx(msg_mx)
    mx.eval(result_mx.data)
    t_mlx = time.perf_counter() - t0

    # Correctness — compare REL_POWER on float32 to avoid dB noise-floor amplification
    proc_np_pow = SpectrumTransformer(
        SpectrumSettings(axis="time", window=WindowFunction.HAMMING, transform=SpectralTransform.REL_POWER)
    )
    proc_mx_pow = SpectrumTransformer(
        SpectrumSettings(axis="time", window=WindowFunction.HAMMING, transform=SpectralTransform.REL_POWER)
    )
    rp_np = proc_np_pow(msg_np)
    rp_mx = proc_mx_pow(msg_mx)
    np.testing.assert_allclose(np.asarray(rp_mx.data), rp_np.data.astype(np.float32), rtol=5e-3, atol=1e-10)

    assert isinstance(result_mx.data, mx.array)

    print(
        f"\n  Spectrum benchmark ({n_windows} windows, {win_samples} samples × {n_channels} ch):"
        f"\n    numpy: {t_numpy:.4f}s"
        f"\n    mlx:   {t_mlx:.4f}s"
        f"\n    ratio (mlx/numpy): {t_mlx / t_numpy:.2f}x"
    )
