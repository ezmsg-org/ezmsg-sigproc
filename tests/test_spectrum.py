import copy
from dataclasses import field
import os
import pytest
import typing

import numpy as np
import scipy.signal as sps
import scipy.fft as sp_fft
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.sigproc.spectrum import (
    spectrum,
    SpectralTransform,
    SpectralOutput,
    WindowFunction,
    Spectrum,
    SpectrumSettings,
)
from ezmsg.sigproc.window import Window, WindowSettings
from ezmsg.sigproc.synth import EEGSynth, EEGSynthSettings
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messagecodec import message_log
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings
from util import (
    get_test_fn,
    create_messages_with_periodic_signal,
    assert_messages_equal,
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

    f, Pxx = sps.welch(
        ch0_raw, fs=1 / raw.axes["time"].gain, window="hamming", nperseg=len(ch0_raw)
    )
    if welch_db:
        Pxx = 10 * np.log10(Pxx)
    ax[1].plot(f, Pxx, label="welch", color="tab:orange", linestyle="--")
    ax[1].set_ylabel("dB" if welch_db else "V**2/Hz")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


@pytest.mark.parametrize("window", [WindowFunction.HANNING, WindowFunction.HAMMING])
@pytest.mark.parametrize(
    "transform", [SpectralTransform.REL_DB, SpectralTransform.REL_POWER]
)
@pytest.mark.parametrize(
    "output", [SpectralOutput.POSITIVE, SpectralOutput.NEGATIVE, SpectralOutput.FULL]
)
def test_spectrum_gen_multiwin(
    window: WindowFunction, transform: SpectralTransform, output: SpectralOutput
):
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

    gen = spectrum(axis="time", window=window, transform=transform, output=output)
    result = gen.send(input_multiwin)
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
    f_len = (
        win_len if output == SpectralOutput.FULL else (win_len // 2 + 1 - (win_len % 2))
    )
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
@pytest.mark.parametrize(
    "transform", [SpectralTransform.REL_DB, SpectralTransform.REL_POWER]
)
@pytest.mark.parametrize(
    "output", [SpectralOutput.POSITIVE, SpectralOutput.NEGATIVE, SpectralOutput.FULL]
)
def test_spectrum_gen(
    window: WindowFunction, transform: SpectralTransform, output: SpectralOutput
):
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

    gen = spectrum(axis="time", window=window, transform=transform, output=output)
    results = [gen.send(msg) for msg in messages]

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

    gen = spectrum(
        axis="time",
        window=WindowFunction.NONE,
        transform=SpectralTransform.RAW_COMPLEX if complex else SpectralTransform.REAL,
        output=SpectralOutput.FULL if complex else SpectralOutput.POSITIVE,
        norm="backward",
        do_fftshift=False,
        nfft=nfft,
    )
    results = [gen.send(msg) for msg in messages]
    test_spec = results[0].data
    if complex:
        sp_res = sp_fft.fft(messages[0].data, n=nfft, axis=0)
    else:
        sp_res = sp_fft.rfft(messages[0].data, n=nfft, axis=0).real
    assert np.allclose(test_spec, sp_res)


class SpectrumSettingsTest(ez.Settings):
    synth_settings: EEGSynthSettings
    window_settings: WindowSettings
    spectrum_settings: SpectrumSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(
        default_factory=TerminateOnTotalSettings
    )


class SpectrumIntegrationTest(ez.Collection):
    SETTINGS = SpectrumSettingsTest

    SOURCE = EEGSynth()
    WIN = Window()
    SPEC = Spectrum()
    SINK = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.SOURCE.apply_settings(self.SETTINGS.synth_settings)
        self.WIN.apply_settings(self.SETTINGS.window_settings)
        self.SPEC.apply_settings(self.SETTINGS.spectrum_settings)
        self.SINK.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.SOURCE.OUTPUT_SIGNAL, self.WIN.INPUT_SIGNAL),
            (self.WIN.OUTPUT_SIGNAL, self.SPEC.INPUT_SIGNAL),
            (self.SPEC.OUTPUT_SIGNAL, self.SINK.INPUT_MESSAGE),
            (self.SINK.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
        )


def test_spectrum_system(
    test_name: typing.Optional[str] = None,
):
    fs = 500.0
    n_time = 100  # samples per block. dispatch_rate = fs / n_time
    target_dur = 2.0
    window_dur = 1.0
    window_shift = 0.2
    n_ch = 8
    target_messages = int((target_dur - window_dur) / window_shift + 1)
    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    settings = SpectrumSettingsTest(
        synth_settings=EEGSynthSettings(
            fs=fs,
            n_time=n_time,
            alpha_freq=10.5,
            n_ch=n_ch,
        ),
        window_settings=WindowSettings(
            axis="time",
            window_dur=window_dur,
            window_shift=window_shift,
        ),
        spectrum_settings=SpectrumSettings(
            axis="time",
            window=WindowFunction.HAMMING,
            transform=SpectralTransform.REL_DB,
            output=SpectralOutput.POSITIVE,
        ),
        log_settings=MessageLoggerSettings(
            output=test_filename,
        ),
        term_settings=TerminateOnTotalSettings(
            total=target_messages,
        ),
    )
    system = SpectrumIntegrationTest(settings)
    ez.run(SYSTEM=system)

    messages: typing.List[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)
    agg = AxisArray.concatenate(*messages, dim="time")
    # Spectral length is half window length because we output only POSITIVE frequencies.
    win_len = window_dur * fs
    assert agg.data.shape == (target_messages, win_len // 2 + 1 - (win_len % 2), n_ch)
