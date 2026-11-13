"""Correctness lock for the adaptive line-noise canceller.

The :class:`AdaptiveLNCTransformer` is a *stateful, chunked* LMS line-noise
filter with an optional frequency-locked loop (FLL). This test pins two layers:

* **The fixed-frequency core** (FLL disabled, ``freq_time_constant=None``) against an
  independent, deliberately slow sample-by-sample reference
  (:func:`_reference_lnc`). With the loop frozen the NCO is a constant-frequency
  oscillator, so streaming in arbitrary chunks must reproduce the single-pass
  reference and must cancel a stationary line.
* **The frequency tracker** (FLL enabled) against a signal whose line sits at a
  deliberately *offset* normalised frequency (simulating an off/drifting device
  clock or mains): ``omega`` must converge to the true frequency and cancel it
  far better than the frozen filter can.

The reference here is intentionally minimal and self-contained.
"""

from __future__ import annotations

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis
from ezmsg.util.messages.util import replace

from ezmsg.sigproc.adaptive_lnc import (
    AdaptiveLNCSettings,
    AdaptiveLNCTransformer,
)

FS = 30000.0
LINE_FREQ = 60.0


# --------------------------------------------------------------------------- #
# Reference implementation (correct, not fast)                                #
# --------------------------------------------------------------------------- #
def _reference_lnc(
    x: np.ndarray,
    adapt_rate: float,
    control: float = 1.0,
    line_freq: float = LINE_FREQ,
    fs: float = FS,
) -> np.ndarray:
    """Sample-by-sample fixed-frequency LMS line-noise canceller, one channel.

    Quadrature references sin/cos at ``line_freq`` come from a continuous-phase
    oscillator; a 2-tap weight adapts via LMS to reconstruct the line, which is
    gated by ``control`` and subtracted::

        nr   = w_sin*sin + w_cos*cos
        err  = nr - x[n]
        w   -= err * ref * adapt_rate          # for each of sin, cos
        y[n] = x[n] - control * nr
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    omega = 2.0 * np.pi * line_freq / fs
    phases = omega * np.arange(n)
    ref_sin = np.sin(phases).astype(np.float32)
    ref_cos = np.cos(phases).astype(np.float32)

    corrected = np.empty(n, dtype=np.float32)
    w_sin = np.float32(0.0)
    w_cos = np.float32(0.0)
    mu = np.float32(adapt_rate)
    ctrl = np.float32(control)

    for i in range(n):
        rs = ref_sin[i]
        rc = ref_cos[i]
        nr = w_sin * rs + w_cos * rc
        err = nr - x[i]
        w_sin = w_sin - err * (rs * mu)
        w_cos = w_cos - err * (rc * mu)
        corrected[i] = x[i] - nr * ctrl
    return corrected


def _reference_lnc_multichannel(X: np.ndarray, adapt_rate: float, control: float = 1.0, **kw) -> np.ndarray:
    """Apply :func:`_reference_lnc` per channel. ``X`` is (n_samples, n_ch)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    out = np.empty_like(X)
    for c in range(X.shape[1]):
        out[:, c] = _reference_lnc(X[:, c], adapt_rate, control, **kw)
    return out


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _make_axisarray(data: np.ndarray, offset: float = 0.0) -> AxisArray:
    """Wrap (n_samples, n_ch) data as a time/ch AxisArray at FS."""
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": LinearAxis(offset=offset, gain=1.0 / FS)},
        key="test_lnc",
    )


def _stream_in_chunks(proc: AdaptiveLNCTransformer, data: np.ndarray, chunk_sizes: list[int]) -> np.ndarray:
    """Feed ``data`` (n_samples, n_ch) through ``proc`` in the given chunks."""
    outputs = []
    start = 0
    for size in chunk_sizes:
        chunk = data[start : start + size]
        msg = _make_axisarray(chunk, offset=start / FS)
        outputs.append(proc(msg).data)
        start += size
    assert start == data.shape[0], "chunk sizes must cover the whole signal"
    return np.concatenate(outputs, axis=0)


def _line_signal(n: int, n_ch: int, omega: float, amp: float, seed: int) -> np.ndarray:
    """(n, n_ch): 10 Hz 'neural' signal + a line at angular freq ``omega``
    (rad/sample) with per-channel phase + broadband noise."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    t = k / FS
    out = np.zeros((n, n_ch), dtype=np.float32)
    for c in range(n_ch):
        sig = 50.0 * np.sin(2 * np.pi * 10 * t + 0.3 * c)
        noise = amp * np.sin(omega * k + 0.7 + 0.5 * c)
        out[:, c] = sig + noise + rng.normal(0, 5, n)
    return out


def _synthetic_mixture(n: int, n_ch: int = 1, seed: int = 0) -> np.ndarray:
    """Mixture with the line exactly at the nominal LINE_FREQ/FS frequency."""
    omega = 2.0 * np.pi * LINE_FREQ / FS
    return _line_signal(n, n_ch, omega, amp=200.0, seed=seed)


def _band_magnitude(sig_col: np.ndarray, omega: float) -> float:
    """Magnitude of ``sig_col`` at angular frequency ``omega`` over its 2nd
    half (after convergence), via a windowed DFT bin at that exact frequency."""
    half = sig_col.shape[0] // 2
    seg = sig_col[half:]
    win = np.hanning(seg.shape[0])
    k = np.arange(seg.shape[0])
    return float(np.abs(np.sum(seg * win * np.exp(-1j * omega * k))))


def adaptive_lnc(
    line_freq: float = 60.0,
    num_harmonics: int = 1,
    adapt_time_constant: float = 0.1,
    freq_time_constant: float | None = 0.5,
    max_freq_deviation: float | None = 2.0,
    cancel_method: str = "notch",
    axis: str = "time",
    **kwargs,
) -> AdaptiveLNCTransformer:
    """Construct an :class:`AdaptiveLNCTransformer` with the given parameters."""
    return AdaptiveLNCTransformer(
        settings=AdaptiveLNCSettings(
            line_freq=line_freq,
            num_harmonics=num_harmonics,
            adapt_time_constant=adapt_time_constant,
            freq_time_constant=freq_time_constant,
            max_freq_deviation=max_freq_deviation,
            cancel_method=cancel_method,
            axis=axis,
            **kwargs,
        )
    )


# --------------------------------------------------------------------------- #
# Fixed-frequency core (FLL disabled)                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "chunk_sizes",
    [
        [6000],  # single chunk
        [1] * 6000,  # one sample at a time
        [30] * 200,  # uniform small chunks
        [333, 1, 1000, 2666, 1000, 1000],  # irregular chunks
    ],
    ids=["whole", "single-sample", "uniform-30", "irregular"],
)
def test_streaming_matches_reference(chunk_sizes):
    """With the FLL frozen, chunked streaming reproduces the single-pass
    reference to floating-point noise, for any chunking."""
    n = int(sum(chunk_sizes))
    mixture = _synthetic_mixture(n, n_ch=4)
    adapt_rate = 1e-3  # raw LMS step for the reference

    expected = _reference_lnc_multichannel(mixture, adapt_rate, control=1.0)

    # Transformer takes a time constant; mu = 2 / (tau * fs), so tau here
    # reproduces the reference's mu exactly. Tracking frozen.
    proc = adaptive_lnc(
        line_freq=LINE_FREQ,
        adapt_time_constant=2.0 / (adapt_rate * FS),
        freq_time_constant=None,
    )
    got = _stream_in_chunks(proc, mixture, chunk_sizes)

    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-2)


def test_chunking_is_invariant():
    """With the FLL frozen, different chunkings yield the same output."""
    n = 6000
    mixture = _synthetic_mixture(n, n_ch=3, seed=1)

    out_whole = _stream_in_chunks(adaptive_lnc(freq_time_constant=None), mixture, [n])
    out_split = _stream_in_chunks(adaptive_lnc(freq_time_constant=None), mixture, [7, 13, 480] + [500] * 11)
    np.testing.assert_allclose(out_whole, out_split, rtol=1e-4, atol=1e-2)


def test_passthrough_emits_input_unchanged():
    """cancel_method="passthrough" does no LNC: output equals input exactly."""
    n = 3000
    mixture = _synthetic_mixture(n, n_ch=2, seed=2)
    proc = adaptive_lnc(cancel_method="passthrough", freq_time_constant=None)
    got = _stream_in_chunks(proc, mixture, [100] * 30)
    np.testing.assert_array_equal(got, mixture)


def test_cancels_stationary_line():
    """A line exactly at the nominal frequency is cancelled by the frozen
    filter (>20 dB after convergence)."""
    n = int(FS * 1.0)
    mixture = _synthetic_mixture(n, n_ch=1, seed=3)
    omega = 2.0 * np.pi * LINE_FREQ / FS
    proc = adaptive_lnc(line_freq=LINE_FREQ, freq_time_constant=None)
    corrected = _stream_in_chunks(proc, mixture, [300] * (n // 300))

    before = _band_magnitude(mixture[:, 0], omega)
    after = _band_magnitude(corrected[:, 0], omega)
    reduction_db = 20 * np.log10(after / before)
    assert reduction_db < -20.0, f"only {reduction_db:.1f} dB of rejection"


def test_line_freq_change_reseeds_nco():
    """Changing line_freq reseeds the NCO frequency (state reset)."""
    n = 1000
    mixture = _synthetic_mixture(n, n_ch=1, seed=4)
    proc = adaptive_lnc(line_freq=60.0, freq_time_constant=None)
    _stream_in_chunks(proc, mixture, [n])
    omega_60 = proc._state.omega

    proc.update_settings(AdaptiveLNCSettings(line_freq=50.0, freq_time_constant=None))
    _stream_in_chunks(proc, mixture, [n])
    omega_50 = proc._state.omega

    assert omega_60 == pytest.approx(2 * np.pi * 60.0 / FS)
    assert omega_50 == pytest.approx(2 * np.pi * 50.0 / FS)


def test_frequency_tracking_is_bounded_around_nominal():
    max_deviation = 0.25
    proc = adaptive_lnc(
        line_freq=LINE_FREQ,
        freq_time_constant=0.1,
        max_freq_deviation=max_deviation,
    )
    proc(_make_axisarray(np.zeros((1, 1), dtype=np.float32)))

    # A quarter-cycle phasor rotation would request a many-Hz correction in a
    # single FLL step. Both directions must clamp at the configured bounds.
    proc._state.z_phasor_prev = np.ones(1, dtype=np.complex128)
    proc._fll_step(np.full(1, 1j, dtype=np.complex128), beta=1.0)
    freq = proc._state.omega * FS / (2.0 * np.pi)
    assert freq == pytest.approx(LINE_FREQ + max_deviation)

    proc._state.z_phasor_prev = np.ones(1, dtype=np.complex128)
    proc._fll_step(np.full(1, -1j, dtype=np.complex128), beta=1.0)
    freq = proc._state.omega * FS / (2.0 * np.pi)
    assert freq == pytest.approx(LINE_FREQ - max_deviation)


# --------------------------------------------------------------------------- #
# Frequency tracking (FLL enabled)                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_ch", [1, 16], ids=["single", "pooled-16ch"])
def test_tracks_offset_frequency(n_ch):
    """When the real line sits off the nominal frequency, the FLL converges
    omega to it and cancels far better than the frozen filter."""
    n = int(FS * 2.0)  # 2 s
    omega_nominal = 2.0 * np.pi * LINE_FREQ / FS
    omega_true = omega_nominal * 1.01  # ~0.6 Hz mains / clock offset
    mixture = _line_signal(n, n_ch, omega_true, amp=200.0, seed=7)
    chunks = [200] * (n // 200)

    tracking = adaptive_lnc(line_freq=LINE_FREQ, adapt_time_constant=0.03, freq_time_constant=0.1)
    out_track = _stream_in_chunks(tracking, mixture, chunks)

    frozen = adaptive_lnc(line_freq=LINE_FREQ, freq_time_constant=None)
    out_frozen = _stream_in_chunks(frozen, mixture, chunks)

    # omega converged close to the true line frequency.
    assert tracking._state.omega == pytest.approx(omega_true, rel=2e-3)

    # ...and the residual at the true frequency is far smaller than frozen.
    res_track = _band_magnitude(out_track[:, 0], omega_true)
    res_frozen = _band_magnitude(out_frozen[:, 0], omega_true)
    assert res_track < 0.2 * res_frozen


def test_single_shot_tracks_within_one_chunk():
    """A single offline buffer cleans itself: the FLL walks it in blocks and
    converges omega to the offset line frequency using only that buffer."""
    n = int(FS * 2.0)
    omega_nominal = 2.0 * np.pi * LINE_FREQ / FS
    omega_true = omega_nominal * 1.01
    mixture = _line_signal(n, n_ch=8, omega=omega_true, amp=200.0, seed=9)

    proc = adaptive_lnc(line_freq=LINE_FREQ, adapt_time_constant=0.03, freq_time_constant=0.1)
    _stream_in_chunks(proc, mixture, [n])  # one shot
    assert proc._state.omega == pytest.approx(omega_true, rel=2e-3)


def test_tracking_is_chunk_invariant():
    """With the global block grid, tracking-on output no longer depends on how
    the stream is chunked (matches to floating-point noise)."""
    n = int(FS * 1.0)
    omega_true = (2.0 * np.pi * LINE_FREQ / FS) * 1.01
    mixture = _line_signal(n, n_ch=4, omega=omega_true, amp=200.0, seed=10)

    split = [37, 211, 752] + [500] * 58
    assert sum(split) == n
    out_whole = _stream_in_chunks(adaptive_lnc(freq_time_constant=0.1), mixture, [n])
    out_split = _stream_in_chunks(adaptive_lnc(freq_time_constant=0.1), mixture, split)
    np.testing.assert_allclose(out_whole, out_split, rtol=1e-3, atol=1e-1)


def test_pooling_does_not_require_phase_alignment():
    """The pooled estimator must track even when channels carry the line at
    very different phases (it uses a static-phase-invariant cross-product)."""
    n = int(FS * 2.0)
    omega_true = (2.0 * np.pi * LINE_FREQ / FS) * 0.99
    # Wildly different per-channel phases (0.5*c radians across 16 ch).
    mixture = _line_signal(n, 16, omega_true, amp=200.0, seed=11)
    proc = adaptive_lnc(line_freq=LINE_FREQ, adapt_time_constant=0.03, freq_time_constant=0.1)
    _stream_in_chunks(proc, mixture, [200] * (n // 200))
    assert proc._state.omega == pytest.approx(omega_true, rel=3e-3)


# --------------------------------------------------------------------------- #
# Harmonics                                                                   #
# --------------------------------------------------------------------------- #
def test_num_harmonics_cancels_harmonic_content():
    """A non-sinusoidal line (fundamental + 3rd harmonic): num_harmonics=1
    leaves the harmonic, num_harmonics=3 removes it too."""
    n = int(FS * 2.0)
    k = np.arange(n)
    omega1 = 2.0 * np.pi * LINE_FREQ / FS
    rng = np.random.default_rng(21)
    # fundamental + strong 3rd harmonic (like the real 'bio' recording)
    line = 200.0 * np.sin(omega1 * k) + 120.0 * np.sin(3 * omega1 * k + 0.5)
    sig = 50.0 * np.sin(2 * np.pi * 10 * k / FS)
    mixture = (sig + line + rng.normal(0, 5, n)).astype(np.float32)[:, None]

    chunks = [200] * (n // 200)
    fund = _stream_in_chunks(adaptive_lnc(LINE_FREQ, num_harmonics=1, adapt_time_constant=0.03), mixture, chunks)
    harm = _stream_in_chunks(adaptive_lnc(LINE_FREQ, num_harmonics=3, adapt_time_constant=0.03), mixture, chunks)

    def red(y, hz):
        return 20 * np.log10(
            _band_magnitude(y[:, 0], 2 * np.pi * hz / FS) / _band_magnitude(mixture[:, 0], 2 * np.pi * hz / FS)
        )

    # Both kill the fundamental.
    assert red(fund, 60) < -20.0
    assert red(harm, 60) < -20.0
    # Only the harmonic-aware filter kills the 3rd harmonic.
    assert red(fund, 180) > -3.0  # essentially untouched
    assert red(harm, 180) < -20.0  # strongly suppressed


# --------------------------------------------------------------------------- #
# Common-mode reconstruct-and-subtract (cancel_method="subtract")             #
# --------------------------------------------------------------------------- #
def _common_mode_scene(n, nch, seed):
    """(x, line, sig): a common-mode 60 Hz line (sub-degree tau_c phases,
    per-channel amplitude) plus an *independent* per-channel 60 Hz burst to
    preserve. (Sampling-delay phases are tiny at 60 Hz; alignment, if needed,
    is a separate upstream stage.)"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    slot = np.arange(nch) % 32
    tau = slot * (64.0 / 66.0e6)
    w = 2 * np.pi * LINE_FREQ
    line = (200.0 * (1 + 0.2 * rng.standard_normal(nch)))[None, :] * np.cos(w * t[:, None] + 0.7 + w * tau[None, :])
    psi = rng.uniform(0, 2 * np.pi, nch)
    env = np.exp(-(((t - n / FS / 2) / 0.2) ** 2))[:, None]  # mid-record burst
    sig = (25.0 * np.cos(w * t[:, None] + psi[None, :])) * env
    x = (line + sig + 5.0 * rng.standard_normal((n, nch))).astype(np.float32)
    return x, line.astype(np.float32), sig.astype(np.float32)


def _fraction(y, ref):
    """Mean per-channel projection of ``y`` onto a known component ``ref``."""
    return float(np.mean([(y[:, c] @ ref[:, c]) / (ref[:, c] @ ref[:, c]) for c in range(ref.shape[1])]))


def test_cancel_method_defaults_to_notch():
    assert adaptive_lnc().settings.cancel_method == "notch"


def test_subtract_preserves_inband_signal_that_notch_destroys():
    """Both remove the common-mode line, but subtract keeps the independent
    in-band signal the notch erases. Each mode uses its natural
    adapt_time_constant: a notch wide enough to catch the line, and a slow
    line-estimate for subtract (so it ignores the transient signal)."""
    n, nch = int(FS * 2.0), 16
    x, line, sig = _common_mode_scene(n, nch, seed=7)
    chunks = [250] * (n // 250)

    def run(method, atc):
        proc = adaptive_lnc(
            LINE_FREQ,
            adapt_time_constant=atc,
            freq_time_constant=None,
            cancel_method=method,
        )
        return _stream_in_chunks(proc, x, chunks)

    y_notch = run("notch", 0.1)  # ~3 Hz notch — covers the burst
    y_sub = run("subtract", 0.5)  # slow line estimate — ignores the burst

    # Both strongly remove the (ground-truth) common-mode line.
    assert _fraction(y_notch, line) < 0.2
    assert _fraction(y_sub, line) < 0.2

    # The discriminator: notch erases the in-band signal; subtract preserves it.
    assert _fraction(y_notch, sig) < 0.25
    assert _fraction(y_sub, sig) > 0.6


# --------------------------------------------------------------------------- #
# Array-API / MLX backend                                                     #
# --------------------------------------------------------------------------- #
def test_mlx_backend_matches_numpy():
    """The MLX (GPU) backend reproduces the numpy result: same cancellation and
    the FLL converges to the same frequency."""
    mx = pytest.importorskip("mlx.core")
    n = int(FS * 1.0)
    omega_true = (2.0 * np.pi * LINE_FREQ / FS) * 1.01
    mixture = _line_signal(n, 8, omega_true, amp=200.0, seed=12).astype(np.float32)
    mixture += (100.0 * np.sin(3 * omega_true * np.arange(n))[:, None]).astype(np.float32)
    chunks = [200] * (n // 200)

    def make():
        return adaptive_lnc(LINE_FREQ, num_harmonics=3, adapt_time_constant=0.05, freq_time_constant=0.1)

    p_np = make()
    y_np = _stream_in_chunks(p_np, mixture, chunks)

    p_mx = make()
    outs, start, last = [], 0, None
    for size in chunks:
        d = mx.array(mixture[start : start + size])
        msg = AxisArray(
            data=d,
            dims=["time", "ch"],
            axes={"time": LinearAxis(offset=start / FS, gain=1.0 / FS)},
            key="test_lnc",
        )
        last = p_mx(msg).data
        outs.append(np.array(last))
        start += size
    y_mx = np.concatenate(outs, axis=0)

    # Output stays on the MLX backend (same array type and dtype as the input).
    assert isinstance(last, mx.array), f"expected mlx output, got {type(last)}"
    assert last.dtype == mx.float32

    # float32 (MLX) vs float64 (scipy) sosfilt on a near-unit-circle notch:
    # compare at signal scale rather than element-wise near the cancelled zeros.
    rel = np.max(np.abs(y_mx - y_np)) / np.max(np.abs(mixture))
    assert rel < 1e-2, f"MLX vs numpy rel diff {rel:.2e}"
    assert p_mx._state.omega == pytest.approx(p_np._state.omega, rel=1e-3)


# --------------------------------------------------------------------------- #
# Warm start and keeping the tracked frequency across a reset                 #
# --------------------------------------------------------------------------- #
WARM_N = int(0.1 * FS)  # first 100 ms (6 line periods, 1 period of the 10 Hz 'neural' tone)


def _line_amp(y: np.ndarray, omega: float, start: int, stop: int) -> float:
    """Mean per-channel amplitude of ``y`` at ``omega`` over ``[start, stop)``
    (rectangular DFT bin; windows here span whole periods)."""
    k = np.arange(start, stop)
    z = np.sum(y[start:stop] * np.exp(-1j * omega * k)[:, None], axis=0)
    return float(np.mean(np.abs(z)) * 2 / (stop - start))


def _messages(proc: AdaptiveLNCTransformer, data: np.ndarray, chunk_sizes: list[int]) -> list[AxisArray]:
    """Like :func:`_stream_in_chunks` but return the output messages."""
    outs, start = [], 0
    for size in chunk_sizes:
        outs.append(proc(_make_axisarray(data[start : start + size], offset=start / FS)))
        start += size
    return outs


def test_warm_start_seed_is_steady_state():
    """The closed-form notch seed is the exact steady state of the fitted line:
    a pure line (3 harmonics, nominal frequency) is cancelled from sample 0."""
    n = WARM_N
    k = np.arange(n)
    w = 2.0 * np.pi * LINE_FREQ / FS
    x = np.stack([200 * np.sin(w * k + c) + 80 * np.cos(3 * w * k + 0.3 * c) for c in range(3)], axis=1)
    y = _stream_in_chunks(adaptive_lnc(num_harmonics=3, freq_time_constant=None, warm_start="first_chunk"), x, [n])
    assert np.max(np.abs(y)) < 1e-6 * 200


def test_warm_start_cancels_from_first_sample():
    """Line + noise at the nominal frequency: residual line over the first 100 ms
    drops from ~the full line (cold) to near the converged level (warm)."""
    n = int(FS * 1.5)
    w = 2.0 * np.pi * LINE_FREQ / FS
    x = _line_signal(n, 4, w, amp=200.0, seed=30)
    chunks = [300] * (n // 300)
    cold = _stream_in_chunks(adaptive_lnc(), x, chunks)
    warm = _stream_in_chunks(adaptive_lnc(warm_start="buffered"), x, chunks)
    first = _stream_in_chunks(adaptive_lnc(warm_start="first_chunk"), x, [WARM_N] + chunks[10:])

    cold_early = _line_amp(cold, w, 0, WARM_N)
    warm_early = _line_amp(warm, w, 0, WARM_N)
    converged = _line_amp(cold, w, n - WARM_N, n)
    print(f"cold {cold_early:.3g}, warm {warm_early:.3g}, converged {converged:.3g}")
    assert warm_early < 3 * converged
    assert warm_early < 0.01 * cold_early
    # first_chunk with a 100 ms first chunk fits the same window as buffered.
    np.testing.assert_allclose(first, warm, rtol=1e-6, atol=1e-6)


def test_warm_start_offset_frequency_no_worse_than_cold():
    """Line at 60.5 Hz (inside max_freq_deviation): warm start is far better over
    the first 100 ms, no worse integrated over the run, and converges the same."""
    n = int(FS * 3.0)
    w = 2.0 * np.pi * 60.5 / FS
    x = _line_signal(n, 4, w, amp=200.0, seed=31)
    chunks = [300] * (n // 300)
    p_cold, p_warm = adaptive_lnc(), adaptive_lnc(warm_start="buffered")
    cold = _stream_in_chunks(p_cold, x, chunks)
    warm = _stream_in_chunks(p_warm, x, chunks)

    def wins(y):
        return np.array([_line_amp(y, w, i, i + WARM_N) for i in range(0, n, WARM_N)])

    wc, ww = wins(cold), wins(warm)
    print(f"first 100 ms cold {wc[0]:.3g} warm {ww[0]:.3g}")
    print(f"mean power cold {np.mean(wc**2):.4g} warm {np.mean(ww**2):.4g}")
    assert ww[0] < 0.1 * wc[0]
    assert np.mean(ww**2) <= np.mean(wc**2)
    assert p_warm.state.omega == pytest.approx(w, rel=2e-3)
    assert ww[-1] <= 1.1 * wc[-1]


def test_warm_start_without_line_matches_cold():
    """No line: the fit is ~zero (noise only), so warm output equals cold."""
    n = int(FS * 0.5)
    x = _line_signal(n, 4, 2.0 * np.pi * LINE_FREQ / FS, amp=0.0, seed=32)
    chunks = [300] * (n // 300)
    cold = _stream_in_chunks(adaptive_lnc(), x, chunks)
    warm = _stream_in_chunks(adaptive_lnc(warm_start="buffered"), x, chunks)
    diff = np.max(np.abs(warm - cold))
    print(f"max |warm - cold| = {diff:.3g} (signal peak {np.max(np.abs(x)):.3g})")
    assert diff < 0.5


def test_warm_start_harmonics():
    """Fundamental + 3rd harmonic with num_harmonics=3: both are cancelled over
    the first 100 ms with warm start, not with cold."""
    n = int(FS * 0.5)
    k = np.arange(n)
    w = 2.0 * np.pi * LINE_FREQ / FS
    rng = np.random.default_rng(33)
    x = (200 * np.sin(w * k) + 120 * np.sin(3 * w * k + 0.5) + rng.normal(0, 5, n)).astype(np.float32)[:, None]
    chunks = [250] * (n // 250)
    cold = _stream_in_chunks(adaptive_lnc(num_harmonics=3), x, chunks)
    warm = _stream_in_chunks(adaptive_lnc(num_harmonics=3, warm_start="buffered"), x, chunks)
    for h, amp in [(1, 200.0), (3, 120.0)]:
        c, wa = _line_amp(cold, h * w, 0, WARM_N), _line_amp(warm, h * w, 0, WARM_N)
        print(f"harmonic {h}: cold {c:.3g} warm {wa:.3g}")
        assert wa < 0.01 * amp
        assert wa < 0.05 * c


def test_warm_start_subtract():
    """SUBTRACT seeds its line estimate from the fit: the line is removed in the
    first window instead of after the estimate has converged."""
    n, nch = int(FS * 0.5), 16
    x, line, _ = _common_mode_scene(n, nch, seed=34)
    chunks = [250] * (n // 250)

    def run(**kw):
        p = adaptive_lnc(adapt_time_constant=0.5, freq_time_constant=None, cancel_method="subtract", **kw)
        return _stream_in_chunks(p, x, chunks)

    period = int(FS / LINE_FREQ)  # the cold estimate is first committed after one period
    cold, warm = run(), run(warm_start="buffered")
    fc, fw = _fraction(cold[:period], line[:period]), _fraction(warm[:period], line[:period])
    fc100, fw100 = _fraction(cold[:WARM_N], line[:WARM_N]), _fraction(warm[:WARM_N], line[:WARM_N])
    print(f"line fraction, first period: cold {fc:.3g} warm {fw:.3g}; first 100 ms: cold {fc100:.3g} warm {fw100:.3g}")
    assert fc > 0.9
    assert abs(fw) < 0.05
    assert abs(fw100) < abs(fc100)


def test_warm_start_buffered_holds_then_releases():
    """buffered: zero-length output until the window fills, then every held
    sample at its original time; identical output for any chunking."""
    n = int(FS * 0.5)
    x = _line_signal(n, 3, 2.0 * np.pi * 60.4 / FS, amp=200.0, seed=35)

    msgs = _messages(adaptive_lnc(warm_start="buffered"), x, [700] * 4 + [n - 2800])
    for m in msgs[:3]:
        assert m.data.shape == (0, 3) and m.data.dtype == x.dtype
        assert m.axes["time"].offset == 0.0 and m.axes["time"].gain == 1.0 / FS
    assert msgs[3].data.shape == (0, 3)  # 2800 < 3000
    assert msgs[4].data.shape == (n, 3) and msgs[4].axes["time"].offset == 0.0

    def run(chunks):
        out = _messages(adaptive_lnc(warm_start="buffered"), x, chunks)
        return np.concatenate([m.data for m in out]), out

    ref, _ = run([n])
    for chunks in ([1] * 3200 + [n - 3200], [37, 211, 2751, 2, 999] + [1000] * 11):
        got, out = run(chunks)
        assert got.shape == ref.shape  # nothing lost or duplicated
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-1)
        # After the release each message keeps its own original timestamp.
        starts = np.cumsum([0] + chunks[:-1])
        nonempty = [(m, s) for m, s in zip(out, starts) if m.data.shape[0]]
        assert nonempty[0][0].axes["time"].offset == 0.0
        for m, s in nonempty[1:]:
            assert m.axes["time"].offset == pytest.approx(s / FS)


def test_keep_frequency_on_reset_reconverges_faster():
    """After a gap (reset), keeping the tracked off-nominal frequency cancels
    better than restarting from line_freq; with warm start too, the fit uses
    the kept frequency and the restart is clean from the first sample."""
    n, gap = int(FS * 3.0), int(FS * 0.37)
    w = (2.0 * np.pi * LINE_FREQ / FS) * 1.01
    sig = _line_signal(n + gap + int(FS * 1.0), 4, w, amp=200.0, seed=36)
    before, after = sig[:n], sig[n + gap :]

    def restart(keep, warm_start="off"):
        p = adaptive_lnc(freq_time_constant=0.2, keep_frequency_on_reset=keep, warm_start=warm_start)
        _stream_in_chunks(p, before, [300] * (n // 300))
        assert p.state.omega == pytest.approx(w, rel=2e-3)
        p._request_reset()
        return _stream_in_chunks(p, after, [300] * 100)

    y_keep, y_cold = restart(True), restart(False)
    rk, rc = _line_amp(y_keep, w, 6000, 15000), _line_amp(y_cold, w, 6000, 15000)
    y_kw, y_w = restart(True, "buffered"), restart(False, "buffered")
    rkw, rw = (max(_line_amp(y, w, i, i + WARM_N) for i in range(0, 15000, WARM_N)) for y in (y_kw, y_w))
    print(f"0.2-0.5 s after restart: kept {rk:.3g} cold {rc:.3g}")
    print(f"worst 100 ms over 0.5 s: kept+warm {rkw:.3g} warm {rw:.3g}")
    assert rk < 0.6 * rc
    assert rkw < 3.0
    assert rkw < 0.1 * rw


def test_keep_frequency_on_reset_fallbacks():
    """The kept omega survives a plain reset, but reseeds to nominal when fs
    changes or the kept value is outside the new bounds."""
    w = (2.0 * np.pi * LINE_FREQ / FS) * 1.01  # 60.6 Hz
    x = _line_signal(int(FS * 2.0), 2, w, amp=200.0, seed=38)
    p = adaptive_lnc(freq_time_constant=0.2, keep_frequency_on_reset=True)
    _stream_in_chunks(p, x, [300] * 200)
    kept = p.state.omega
    assert kept == pytest.approx(w, rel=2e-3)

    p._request_reset()
    p(_make_axisarray(x[:100]))  # shorter than an FLL window: omega untouched
    assert p.state.omega == kept

    # Sample rate change -> nominal for the new fs.
    half = AxisArray(x[:100], dims=["time", "ch"], axes={"time": LinearAxis(gain=2.0 / FS)}, key="test_lnc")
    p(half)
    assert p.state.omega == pytest.approx(2.0 * np.pi * LINE_FREQ / (FS / 2))

    # Back at FS (omega no longer comparable) and then out of bounds -> nominal.
    p = adaptive_lnc(freq_time_constant=0.2, keep_frequency_on_reset=True)
    _stream_in_chunks(p, x, [300] * 200)
    p.update_settings(replace(p.settings, max_freq_deviation=0.25))  # forces a reset
    p(_make_axisarray(x[:100]))
    assert p.state.omega == pytest.approx(2.0 * np.pi * LINE_FREQ / FS)


@pytest.mark.parametrize("cancel_method", ["notch", "subtract"])
def test_warm_start_mlx_matches_numpy(cancel_method):
    """Warm start (buffered) on MLX reproduces numpy, incl. the zero-length hold."""
    mx = pytest.importorskip("mlx.core")
    n = int(FS * 0.5)
    w = 2.0 * np.pi * LINE_FREQ / FS
    k = np.arange(n)
    line = 200.0 * np.sin(w * k + 0.7) + 80.0 * np.sin(3 * w * k + 0.4)  # common-mode, as SUBTRACT assumes
    x = (_line_signal(n, 8, w, amp=0.0, seed=39) + line[:, None]).astype(np.float32)
    chunks = [1000] * (n // 1000)

    def make():
        return adaptive_lnc(num_harmonics=3, cancel_method=cancel_method, warm_start="buffered")

    y_np = _stream_in_chunks(make(), x, chunks)
    p_mx, outs, start = make(), [], 0
    for size in chunks:
        msg = AxisArray(
            mx.array(x[start : start + size]),
            dims=["time", "ch"],
            axes={"time": LinearAxis(offset=start / FS, gain=1.0 / FS)},
            key="test_lnc",
        )
        out = p_mx(msg)
        assert isinstance(out.data, mx.array)
        outs.append(np.array(out.data))
        start += size
    assert outs[0].shape == (0, 8) and outs[2].shape == (3000, 8)
    y_mx = np.concatenate(outs, axis=0)
    assert np.max(np.abs(y_mx - y_np)) / np.max(np.abs(x)) < 1e-2
    assert _line_amp(y_mx, w, 0, WARM_N) < 2.0
