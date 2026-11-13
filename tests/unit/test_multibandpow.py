import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.multibandpow import (
    EnvelopePowerOutput,
    MultiBandEnvelopePowerSettings,
    MultiBandEnvelopePowerTransformer,
)
from ezmsg.sigproc.util.minphase import MinPhaseMode

FS = 1000.0
BANDS = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 70.0), (70.0, 150.0)]


def _msg(x: np.ndarray, offset: float = 0.0) -> AxisArray:
    x = x.reshape(x.shape[0], -1)
    return AxisArray(
        x,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=FS, offset=offset),
            "ch": AxisArray.CoordinateAxis(np.arange(x.shape[1]).astype(str), dims=["ch"]),
        },
        key="test",
    )


def _times(out: AxisArray) -> np.ndarray:
    return out.axes["time"].value(np.arange(out.data.shape[0]))


def _common_delay(proc: MultiBandEnvelopePowerTransformer) -> float:
    return proc._procs["filterbank"].state.filterbank.state.delay


def _burst(t, t0, sigma, freqs):
    return sum(np.exp(-0.5 * ((t - t0) / sigma) ** 2) * np.cos(2 * np.pi * f * (t - t0)) for f in freqs)


@pytest.mark.parametrize(
    "output, expect, rtol",
    [
        (EnvelopePowerOutput.POWER, lambda a: a**2, 1e-2),
        (EnvelopePowerOutput.AMPLITUDE, lambda a: a, 5e-3),
        (EnvelopePowerOutput.LOG_POWER, lambda a: np.log10(a**2), None),
    ],
)
def test_sinusoid_in_each_band(output, expect, rtol):
    # Channel i carries a sinusoid at band i's centre, amplitude i + 1.
    t = np.arange(int(4 * FS)) / FS
    amps = np.arange(1.0, len(BANDS) + 1)
    x = np.stack([a * np.cos(2 * np.pi * (lo + hi) / 2 * t + 0.3) for a, (lo, hi) in zip(amps, BANDS)], axis=1)
    proc = MultiBandEnvelopePowerTransformer(MultiBandEnvelopePowerSettings(bands=BANDS, output=output))
    out = proc(_msg(x))
    assert out.dims == ["time", "ch", "band"]
    assert out.stream_dim == "time"
    steady = out.data[_times(out) >= 1.0]  # past the start-up transient
    assert steady.shape[0] > 50
    diag = steady[:, np.arange(len(BANDS)), np.arange(len(BANDS))]
    if rtol is None:  # log power: within 1 % in power
        np.testing.assert_allclose(diag, np.broadcast_to(expect(amps), diag.shape), atol=np.log10(1.01))
    else:
        np.testing.assert_allclose(diag, np.broadcast_to(expect(amps), diag.shape), rtol=rtol)
        off = steady.copy()
        off[:, np.arange(len(BANDS)), np.arange(len(BANDS))] = 0
        # Other bands: below 1 % of the in-band power (10 % in amplitude). The worst case, 0.54 %,
        # is 10.5 Hz leaking into 13-30 Hz, whose 8.5 Hz transition reaches down to 8.75 Hz;
        # every non-adjacent pair is below 1e-6.
        lim = 1e-2 if output == EnvelopePowerOutput.POWER else 1e-1
        assert np.all(off.max(axis=(0, 2)) < lim * expect(amps))


@pytest.mark.parametrize(
    "band, fc, fm, bin_duration", [((15.0, 25.0), 20.0, 2.0, 0.01), ((70.0, 150.0), 110.0, 10.0, 0.01)]
)
def test_am_carrier_follows_envelope_aligned(band, fc, fm, bin_duration):
    t = np.arange(int(5 * FS)) / FS

    def env(tt):
        return 1.0 + 0.5 * np.cos(2 * np.pi * fm * tt)

    x = env(t) * np.cos(2 * np.pi * fc * t)
    proc = MultiBandEnvelopePowerTransformer(MultiBandEnvelopePowerSettings(bands=[band], bin_duration=bin_duration))
    out = proc(_msg(x))
    tt = _times(out)
    keep = tt >= 1.0
    # Expected: mean of a(t)^2 over each bin's own (delay-corrected) sample times.
    spb = round(bin_duration * FS)
    expected = np.array([np.mean(env(t0 + np.arange(spb) / FS) ** 2) for t0 in tt[keep]])
    got = out.data[keep, 0, 0]
    # A 1 ms misalignment would already give > 1 % error at these modulation rates,
    # and a residual 2*fc ripple (as in band-pass -> square) would be ~50 %.
    np.testing.assert_allclose(got, expected, rtol=1e-2)


def test_bands_share_one_time_base():
    bands = [(4.0, 8.0), (13.0, 30.0), (70.0, 150.0)]
    t = np.arange(int(5 * FS)) / FS
    t0 = 2.51  # centre of a 20 ms bin
    x = _burst(t, t0, 0.1, [(lo + hi) / 2 for lo, hi in bands])
    proc = MultiBandEnvelopePowerTransformer(MultiBandEnvelopePowerSettings(bands=bands, bin_duration=0.02))
    out = proc(_msg(x))
    tt = _times(out)
    peaks = tt[np.argmax(out.data[:, 0, :], axis=0)]
    assert np.all(peaks == peaks[0])
    assert peaks[0] <= t0 < peaks[0] + 0.02  # and it is the bin that holds t0


def test_chunk_invariance():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3000, 3))
    settings = MultiBandEnvelopePowerSettings(bands=BANDS, bin_duration=0.013)
    whole = MultiBandEnvelopePowerTransformer(settings)(_msg(x, offset=10.0))

    proc = MultiBandEnvelopePowerTransformer(settings)
    sizes = rng.integers(1, 50, size=200)
    edges = np.concatenate([[0], np.cumsum(sizes)])
    edges = edges[edges < x.shape[0]].tolist() + [x.shape[0]]
    outs = [proc(_msg(x[a:b], offset=10.0 + a / FS)) for a, b in zip(edges[:-1], edges[1:])]
    outs = [o for o in outs if o.data.shape[0]]
    data = np.concatenate([o.data for o in outs])
    times = np.concatenate([_times(o) for o in outs])
    assert data.shape == whole.data.shape
    np.testing.assert_allclose(data, whole.data, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(times, _times(whole), rtol=0, atol=1e-9)


def test_min_phase_lowers_delay_keeps_alignment():
    bands = [(4.0, 8.0), (13.0, 30.0), (70.0, 150.0)]
    t = np.arange(int(5 * FS)) / FS
    t0 = 2.5
    x = _burst(t, t0, 0.1, [(lo + hi) / 2 for lo, hi in bands])

    delays, peaks = {}, {}
    for mode in (MinPhaseMode.NONE, MinPhaseMode.HOMOMORPHICFULL):
        proc = MultiBandEnvelopePowerTransformer(
            MultiBandEnvelopePowerSettings(bands=bands, bin_duration=0.001, min_phase=mode)
        )
        out = proc(_msg(x))
        delays[mode] = _common_delay(proc)
        peaks[mode] = _times(out)[np.argmax(out.data[:, 0, :], axis=0)]

    # Only the 4-8 Hz kernel sets the delay; converting it more than halves the latency.
    assert delays[MinPhaseMode.HOMOMORPHICFULL] < 0.5 * delays[MinPhaseMode.NONE]
    np.testing.assert_allclose(peaks[MinPhaseMode.NONE], t0, atol=1e-3)
    # Approximate alignment: every band's burst peak within 10 ms of the true time (measured: 6 ms).
    np.testing.assert_allclose(peaks[MinPhaseMode.HOMOMORPHICFULL], t0, atol=0.01)


def test_band_axis_and_shape():
    proc = MultiBandEnvelopePowerTransformer(MultiBandEnvelopePowerSettings(bands=BANDS, bin_duration=0.05))
    out = proc(_msg(np.zeros((1000, 4)), offset=3.0))
    assert out.dims == ["time", "ch", "band"]
    assert out.data.shape == (20, 4, len(BANDS))
    assert out.axes["time"].gain == pytest.approx(0.05)
    band_ax = out.axes["band"]
    assert band_ax.dims == ["band"] and band_ax.unit == "Hz"
    np.testing.assert_array_equal(band_ax.data["lo"], [b[0] for b in BANDS])
    np.testing.assert_array_equal(band_ax.data["hi"], [b[1] for b in BANDS])
    assert "_fingerprint" in band_ax.__dict__  # primed at construction
    np.testing.assert_array_equal(out.axes["ch"].data, ["0", "1", "2", "3"])


@pytest.mark.parametrize(
    "bands, kwargs",
    [
        ([(8.0, 4.0)], {}),  # lo >= hi
        ([(0.0, 10.0)], {}),  # lo at 0
        ([(400.0, 500.0)], {}),  # hi at Nyquist
        ([(2.0, 30.0)], {}),  # transition reaches 0 Hz
        ([(400.0, 490.0)], {}),  # transition reaches Nyquist
        ([(4.0, 8.0)], {"transition_ratio": 0.0}),
        ([(4.0, 8.0)], {"min_phase": MinPhaseMode.HOMOMORPHIC}),
    ],
)
def test_invalid_settings_raise(bands, kwargs):
    proc = MultiBandEnvelopePowerTransformer(MultiBandEnvelopePowerSettings(bands=bands, **kwargs))
    with pytest.raises(ValueError):
        proc(_msg(np.zeros((100, 1))))


def test_startup_constant_keeps_dc_offset_out_of_the_bands():
    from ezmsg.sigproc.util.startup import StartupMode

    x = np.full((2000, 1), 1000.0)

    def run(startup, chunk=100):
        proc = MultiBandEnvelopePowerTransformer(MultiBandEnvelopePowerSettings(bands=BANDS, startup=startup))
        outs = [proc(_msg(x[i : i + chunk], offset=i / FS)) for i in range(0, len(x), chunk)]
        return np.concatenate([o.data for o in outs if o.data.size], axis=0)

    # Stop-band leakage of the offset (-60 dB of 1e6) against the step a zero pre-history rings with.
    assert run(StartupMode.CONSTANT).max() < 10.0
    assert run(StartupMode.ZERO).max() > 1e4
    assert run(StartupMode.BUFFERED).max() < 10.0  # withheld start-up passes through the later stages
