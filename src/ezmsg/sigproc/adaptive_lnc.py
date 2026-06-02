"""Adaptive line-noise cancellation (LNC).

An LMS adaptive filter that estimates and subtracts a line-frequency (e.g.
50/60 Hz mains) interferer, and optionally its harmonics, from a multichannel
signal. Quadrature references (sin/cos) at the line frequency -- and at each
harmonic ``k`` x the fundamental, phase-locked to it -- drive per-channel
adaptive weights that reconstruct the line noise; the summed reconstruction is
subtracted from the signal (Widrow & Stearns, *Adaptive Signal Processing*, LMS
noise cancellation with quadrature references).

Because we observe the signal only in *samples*, the line shows up at a
normalised frequency of ``f_mains / fs`` cycles/sample. Both the device clock
(``fs``) and the mains frequency are imprecise and drift, so that ratio is
unknown and time-varying. The transformer therefore tracks it adaptively:

  * A numerically-controlled oscillator (NCO) generates the reference at an
    angular frequency ``omega`` (rad/sample); harmonics are generated at
    ``k * omega``, so a single oscillator serves every harmonic.
  * A per-channel LMS adapts amplitude and phase at each harmonic frequency.
  * A frequency-locked loop (FLL) nudges ``omega`` from the rotation of the
    fundamental's adaptive weights, **pooled across all channels** for
    robustness -- the
    line frequency is a single global quantity, so every channel constrains
    the same estimate. The pooled estimator (a power-weighted cross-product)
    is invariant to static per-channel phase, so it needs no assumption about
    inter-channel phase relationships.

Both loops are parameterised by a **time constant in seconds**, so their
behaviour is independent of the (possibly wildly variable) chunk size:

  * ``adapt_time_constant`` is the settling time of the per-channel LMS;
    internally ``mu = 2 / (adapt_time_constant * fs)``.
  * ``freq_time_constant`` is the settling time of the FLL; the per-update gain
    is derived from the elapsed time of each update interval,
    ``beta = 1 - exp(-dt / freq_time_constant)`` (the same tau<->alpha mapping
    EWMA uses), so chunking never changes the dynamics. It should exceed
    ``adapt_time_constant`` so the two loops do not fight.

The frequency estimate accumulates over an internal window of one mains period
before each update -- not a fixed chunk size, but a measurement window that
fills across however the caller chunks the stream (4, 30, 60, mixed samples).
Output is emitted for every chunk immediately; only the frequency *update*
waits for a full window, which keeps the rotation estimate clean even when
chunks are only a handful of samples. As a result the loop behaves identically
for a single offline buffer (it converges within that one buffer) and for a
fast sequence of tiny chunks, and tracking-on output is chunk-invariant.

The sequential A/D also gives each channel a known sub-degree per-channel phase
offset, but at the line frequency it is negligible and it cancels in frequency
tracking, so it is *not* handled here -- the dedicated
``SamplingDelayAlignmentTransformer`` owns that correction broadband, upstream,
for every cross-channel step.

Statefulness: expensive setup happens once in :meth:`_reset_state`; each chunk
runs only the per-sample recursion plus one pooled frequency update. With the
FLL disabled (``freq_time_constant`` None/<=0) the transformer reduces to a
fixed-frequency canceller whose output is independent of chunking.
"""

import enum
from typing import Any

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import scipy.signal
from array_api_compat import array_namespace
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

# Optional Apple-Silicon GPU backend. The canceller is an LTI SOS notch
# cascade (see `design_lnc_sos`), so on MLX arrays we dispatch to the Metal
# `sosfilt` kernel; everything else runs through scipy on the array's own
# namespace (numpy, cupy, ...).
try:  # pragma: no cover - exercised only when mlx is installed
    import mlx.core as _mx

    from ezmsg.sigproc.util.sosfilt_mlx_metal import sosfilt_mlx_metal as _sosfilt_mlx
except Exception:  # pragma: no cover
    _mx = None
    _sosfilt_mlx = None


def _is_mlx(arr: object) -> bool:
    """True if ``arr`` is an MLX array (and the MLX backend is available)."""
    return _mx is not None and isinstance(arr, _mx.array)


def _namespace(arr: object) -> tuple[Any, bool]:
    """Return ``(xp, is_mlx)`` for ``arr``: the MLX module for MLX arrays, else
    the array's Array-API namespace (numpy, cupy, ...)."""
    if _is_mlx(arr):
        return _mx, True
    return array_namespace(arr), False


def _to_numpy(arr: object) -> np.ndarray:
    """Bring a small backend array to host numpy (for the scalar FLL math)."""
    return np.asarray(arr)


class CancelMethod(str, enum.Enum):
    """How :class:`AdaptiveLNCTransformer` removes the line."""

    NOTCH = "notch"
    """Apply the SOS notch cascade -- a perfect null that also removes any
    signal at the line frequency."""

    SUBTRACT = "subtract"
    """Estimate the common-mode line (pooled global phase + per-channel
    amplitude) once per window and subtract *only* it, preserving signal that
    is independent across channels at the line frequency. Assumes a 1-D channel
    axis."""

    PASSTHROUGH = "passthrough"
    """Emit the input unchanged -- no cancellation and no frequency tracking."""


class AdaptiveLNCSettings(ez.Settings):
    """Settings for :class:`AdaptiveLNCTransformer`."""

    line_freq: float = 60.0
    """Nominal line (mains) frequency in Hz (60 in N. America, 50 in EU). Seeds
    the NCO; the true normalised frequency is then tracked by the FLL."""

    num_harmonics: int = 1
    """Number of line-frequency harmonics to cancel, including the fundamental.
    1 = fundamental only. Harmonic ``k`` is generated phase-locked at ``k`` x
    the single tracked fundamental, so one NCO/FLL serves them all; each
    harmonic gets its own per-channel LMS weight. e.g. 5 also cancels
    120/180/240/300 Hz. Absent harmonics simply drive their weights to ~0."""

    adapt_time_constant: float = 0.1
    """Settling time (seconds) of the amplitude/phase canceller (the LMS).
    Smaller = faster tracking of amplitude/phase changes and a wider, noisier
    notch; larger = a narrower, cleaner notch that adapts more slowly. Converted
    internally to the LMS step size ``mu = 2 / (adapt_time_constant * fs)``.
    Read live each chunk."""

    freq_time_constant: float | None = 0.5
    """Settling time (seconds) of the frequency tracker (the FLL). ``None`` (or
    <= 0) freezes the frequency at the nominal ``line_freq`` -- a fixed-
    reference LMS. Should be larger than ``adapt_time_constant`` so the two
    loops do not fight. Independent of chunk size (the per-update gain is
    derived from elapsed time). Read live each chunk."""

    cancel_method: CancelMethod = CancelMethod.NOTCH
    """How to remove the line; see :class:`CancelMethod`. ``NOTCH`` (default)
    applies the SOS notch cascade -- a perfect null that also removes any signal
    at the line frequency. ``SUBTRACT`` estimates the common-mode line (pooled
    global phase + per-channel amplitude) once per window and subtracts *only*
    it, preserving signal that is independent across channels at the line
    frequency; it assumes a 1-D channel axis. ``PASSTHROUGH`` emits the input
    unchanged. (Per-channel sampling-delay alignment is handled separately,
    upstream, by ``SamplingDelayAlignmentTransformer``.)"""

    axis: str = "time"
    """Name of the axis to filter along."""


@processor_state
class AdaptiveLNCState:
    """State for :class:`AdaptiveLNCTransformer`."""

    omega: float = 0.0
    """Current NCO angular frequency in rad/sample (tracked by the FLL)."""

    phase: float = 0.0
    """NCO phase (rad) for the first sample of the next chunk; accumulates so
    the demodulation reference is continuous across chunk boundaries."""

    block_len: int = 0
    """FLL update window in samples (one mains period)."""

    samples_in_block: int = 0
    """Samples accumulated toward the next FLL update; carried across chunks so
    the update grid is global, not per-chunk."""

    zi: npt.NDArray | None = None
    """SOS biquad filter state, carried across chunks. Backend-native layout:
    ``(n_harm, 2, *sample_shape)`` for scipy, ``(n_harm, *sample_shape, 2)`` for
    the MLX kernel."""

    sos: npt.NDArray | None = None
    """Cached notch coefficients, already on the working backend. Rebuilt only
    when ``omega`` or ``mu`` changes (i.e. once per FLL window, not per chunk),
    so streaming avoids redundant host->device transfers."""

    sos_key: tuple | None = None
    """``(omega, mu, n_harm)`` the cached ``sos`` was built for."""

    z_acc_real: npt.NDArray | None = None
    """Demodulation accumulator ``sum(removed * cos(phase))`` over the current
    window (per channel); the line phasor's real part."""

    z_acc_imag: npt.NDArray | None = None
    """Demodulation accumulator ``sum(removed * sin(phase))`` over the current
    window (per channel)."""

    z_phasor_prev: npt.NDArray | None = None
    """Pooled per-channel line phasor (numpy complex) from the previous window;
    the FLL measures rotation against it."""

    line_phasor: complex = 0j
    """Observable: pooled global line phasor at the last window end."""

    # --- "subtract" mode state (per-harmonic; lists so MLX immutability is fine)
    sub_z_real: list | None = None
    """Per-harmonic demod accumulators ``sum(x * cos(k*phase))`` this window."""

    sub_z_imag: list | None = None
    """Per-harmonic demod accumulators ``sum(x * sin(k*phase))`` this window."""

    sub_amp: list | None = None
    """Committed per-harmonic per-channel line amplitude (backend arrays);
    ``None`` until the first window completes."""

    sub_phase_off: list | None = None
    """Committed per-harmonic global phase ``theta_k`` (scalar floats), for
    reconstruction."""

    sub_yc_smooth: list | None = None
    """Per-harmonic EWMA-smoothed per-channel line phasor (numpy complex).
    Smoothing (time constant ``adapt_time_constant``) keeps the line estimate
    from chasing transient in-band signal."""


class AdaptiveLNCTransformer(
    BaseStatefulTransformer[
        AdaptiveLNCSettings,
        AxisArray,
        AxisArray,
        AdaptiveLNCState,
    ]
):
    """Quadrature-LMS line-noise canceller, implemented as its exact LTI
    equivalent (an SOS notch cascade) with frequency tracking.

    **Cancellation.** A fixed-frequency quadrature-LMS canceller is exactly a
    2nd-order notch (Glover 1977; see :func:`design_lnc_sos`). Each harmonic is
    one biquad notch at ``k * omega``; the cascade is applied with ``sosfilt``
    carrying state ``zi`` across chunks. The output is::

        y_notch = sosfilt(design_lnc_sos(omega, mu, num_harmonics), x)
        y = y_notch                                    # the line removed

    This replaces the former per-sample LMS recursion: it is vectorised, scales
    with channel count, and dispatches to a GPU ``sosfilt`` (MLX/Metal) on MLX
    arrays -- so the whole transformer is Array-API compatible.

    **Frequency tracking (FLL).** Once per window (one mains period) the NCO
    frequency is nudged by the pooled rotation of the *demodulated line phasor*
    between windows::

        Z_c = sum_n removed_{c,n} * exp(-j * phase_n)  # per-channel, per window
        cross = sum_c Z_c * conj(Z_c_prev)             # power-weighted, pooled
        omega += beta * angle(cross) / window_len      # rad/sample

    Demodulating the removed line gives the same frequency-error signal the old
    weight-rotation detector did, without materialising adaptive weights. Static
    per-channel phase cancels in ``cross``, so no inter-channel phase assumption
    is needed. The window fills on a global grid carried in state, so the loop
    is chunk-size independent: a single offline buffer converges within itself
    and streamed output is chunk-invariant. ``mu = 2/(adapt_time_constant*fs)``
    and ``beta = 1 - exp(-window_dt/freq_time_constant)``.
    """

    # These are read fresh every chunk inside `_process`, so changing them must
    # NOT reset the filter state. line_freq seeds omega, axis/delay model and
    # num_harmonics shape the state, so those do force a reset.
    NONRESET_SETTINGS_FIELDS = frozenset({"adapt_time_constant", "freq_time_constant"})

    def _hash_message(self, message: AxisArray) -> int:
        ax_idx = message.get_axis_idx(self.settings.axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        return hash((message.key, message.axes[self.settings.axis].gain, sample_shape))

    def _reset_state(self, message: AxisArray) -> None:
        ax_idx = message.get_axis_idx(self.settings.axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        xp, is_mlx = _namespace(message.data)

        fs = 1.0 / message.axes[self.settings.axis].gain
        # Seed the NCO at the nominal normalised frequency; the FLL refines it.
        self._state.omega = 2.0 * np.pi * self.settings.line_freq / fs
        self._state.phase = 0.0

        # Frequency-update window: one mains period, inferred from line_freq/fs.
        self._state.block_len = max(1, int(round(fs / self.settings.line_freq)))
        self._state.samples_in_block = 0

        # SOS filter state: one biquad per harmonic. The two backends use
        # different delay layouts, so `zi` is created per backend; the demod
        # accumulators share their shape and go through `xp`.
        n_harm = max(1, int(self.settings.num_harmonics))
        if is_mlx:
            self._state.zi = _mx.zeros((n_harm,) + sample_shape + (2,))
        else:
            self._state.zi = np.zeros((n_harm, 2) + sample_shape)
        self._state.z_acc_real = xp.zeros(sample_shape)
        self._state.z_acc_imag = xp.zeros(sample_shape)
        self._state.z_phasor_prev = None
        self._state.sos = None  # built lazily on the working backend
        self._state.sos_key = None
        self._state.line_phasor = 0j

        # "subtract" mode: per-harmonic demod accumulators; estimate not yet known.
        self._state.sub_z_real = [xp.zeros(sample_shape) for _ in range(n_harm)]
        self._state.sub_z_imag = [xp.zeros(sample_shape) for _ in range(n_harm)]
        self._state.sub_amp = None
        self._state.sub_phase_off = None
        self._state.sub_yc_smooth = None

    @staticmethod
    def _cat(pieces: list, is_mlx: bool, xp: Any) -> npt.NDArray:
        """Concatenate segment outputs along the time axis (backend-portable)."""
        if len(pieces) == 1:
            return pieces[0]
        return _mx.concatenate(pieces, axis=0) if is_mlx else xp.concat(pieces, axis=0)

    def _ensure_sos(self, mu: float, n_harm: int, is_mlx: bool) -> npt.NDArray:
        """Return the notch coefficients on the working backend, rebuilding only
        when ``(omega, mu, n_harm)`` changes -- i.e. once per FLL window, not per
        chunk. Avoids re-running the design and (on MLX) re-transferring the
        coefficients to the device for every segment within a window."""
        st = self._state
        key = (st.omega, mu, n_harm)
        if st.sos is None or st.sos_key != key:
            sos = design_lnc_sos(st.omega, mu, n_harm)  # numpy float64
            st.sos = _mx.array(sos.astype(np.float32)) if is_mlx else sos
            st.sos_key = key
        return st.sos

    def _sosfilt(self, sos: npt.NDArray, x_seg: npt.NDArray, is_mlx: bool) -> npt.NDArray:
        """Apply the notch cascade over one segment (time on axis 0), carrying
        ``self._state.zi`` across calls. Dispatches to the MLX Metal kernel for
        MLX arrays, else scipy."""
        if is_mlx:
            x_t = _mx.moveaxis(x_seg, 0, x_seg.ndim - 1)  # kernel wants time last
            y_t, self._state.zi = _sosfilt_mlx(sos, x_t, zi=self._state.zi)
            return _mx.moveaxis(y_t, y_t.ndim - 1, 0)
        y, self._state.zi = scipy.signal.sosfilt(sos, x_seg, axis=0, zi=self._state.zi)
        return y

    def _accumulate_demod(self, removed: npt.NDArray, seg_len: int, xp: Any) -> None:
        """Demodulate the removed line against the fundamental NCO and add to
        the current window's phasor accumulators (real/imag, per channel)."""
        st = self._state
        ph = st.phase + st.omega * xp.arange(seg_len)
        bshape = (seg_len,) + (1,) * (removed.ndim - 1)
        cos = xp.reshape(xp.cos(ph), bshape)
        sin = xp.reshape(xp.sin(ph), bshape)
        st.z_acc_real = st.z_acc_real + xp.sum(removed * cos, axis=0)
        st.z_acc_imag = st.z_acc_imag + xp.sum(removed * sin, axis=0)

    def _fll_step(self, z_fund: np.ndarray, beta: float) -> None:
        """Nudge ``omega`` by the pooled rotation of the fundamental complex line
        phasor ``z_fund`` (numpy, per channel) since the previous window. Static
        per-channel phase cancels in the cross-product, so no alignment needed."""
        st = self._state
        if st.z_phasor_prev is not None:
            cross = np.sum(z_fund * np.conj(st.z_phasor_prev))
            if cross != 0:
                st.omega = st.omega + beta * float(np.angle(cross)) / st.block_len
        st.z_phasor_prev = z_fund

    def _freq_update(self, beta: float) -> None:
        """Notch-mode window boundary: run the FLL off the fundamental demod of
        the removed line, refresh the observable phasor, reset accumulators."""
        st = self._state
        # e^{-j phase} = cos - j sin, so Z = acc_real - j acc_imag (per channel).
        z = _to_numpy(st.z_acc_real).reshape(-1) - 1j * _to_numpy(st.z_acc_imag).reshape(-1)
        self._fll_step(z, beta)
        st.line_phasor = complex(np.sum(z))  # observable pooled line phasor

        # Clear accumulators for the next window.
        st.z_acc_real = st.z_acc_real * 0
        st.z_acc_imag = st.z_acc_imag * 0

    # ----- "subtract" mode -------------------------------------------------- #
    def _accumulate_subtract(self, x_seg: npt.NDArray, seg_len: int, n_harm: int, xp: Any) -> None:
        """Demodulate the *input* at every harmonic into the per-harmonic window
        accumulators (the rank-1 pool will separate the common-mode line from
        independent in-band signal)."""
        st = self._state
        base = st.phase + st.omega * xp.arange(seg_len)
        bshape = (seg_len,) + (1,) * (x_seg.ndim - 1)
        for kidx in range(n_harm):
            ph = (kidx + 1) * base
            cos = xp.reshape(xp.cos(ph), bshape)
            sin = xp.reshape(xp.sin(ph), bshape)
            st.sub_z_real[kidx] = st.sub_z_real[kidx] + xp.sum(x_seg * cos, axis=0)
            st.sub_z_imag[kidx] = st.sub_z_imag[kidx] + xp.sum(x_seg * sin, axis=0)

    def _commit_subtract(self, n_harm: int, beta: float, alpha: float, tracking: bool, xp: Any) -> None:
        """Window boundary: demod -> EWMA-smooth (time constant
        ``adapt_time_constant``, via ``alpha``) the per-harmonic per-channel line
        phasor, then commit the rank-1 estimate (global phase + per-channel
        amplitude) for the next window's reconstruction, optionally step the FLL,
        and reset the accumulators."""
        st = self._state
        scale = 2.0 / st.block_len  # demod sum -> sinusoid amplitude (window is ~1 period)
        smoothed = []
        amps, offs = [], []
        z0 = None
        for kidx in range(n_harm):
            z = scale * (_to_numpy(st.sub_z_real[kidx]).reshape(-1) - 1j * _to_numpy(st.sub_z_imag[kidx]).reshape(-1))
            if kidx == 0:
                z0 = z
            if st.sub_yc_smooth is not None:
                z = (1.0 - alpha) * st.sub_yc_smooth[kidx] + alpha * z
            smoothed.append(z)
            # Global common-mode phase: amplitude-weighted pool (rank-1 direction).
            theta = float(np.angle(np.sum(z * np.abs(z)))) if np.any(z) else 0.0
            a = np.clip(np.real(z * np.exp(-1j * theta)), 0.0, None)  # per-channel amplitude
            amps.append(xp.asarray(a))
            offs.append(theta)  # global phase per harmonic (scalar)
        st.sub_yc_smooth = smoothed
        st.sub_amp = amps
        st.sub_phase_off = offs

        if tracking and z0 is not None:
            self._fll_step(z0, beta)

        st.sub_z_real = [z * 0 for z in st.sub_z_real]
        st.sub_z_imag = [z * 0 for z in st.sub_z_imag]

    def _reconstruct(self, seg_len: int, n_harm: int, xp: Any) -> npt.NDArray | None:
        """Reconstruct the committed common-mode line over one segment, using the
        previous window's estimate. ``None`` before the first commit."""
        st = self._state
        if st.sub_amp is None:
            return None
        base = st.phase + st.omega * xp.arange(seg_len)
        line = None
        for kidx in range(n_harm):
            k = kidx + 1
            # arg[n] = k*base[n] + theta_k (global phase; same across channels)
            arg = k * xp.reshape(base, (seg_len, 1)) + st.sub_phase_off[kidx]
            term = st.sub_amp[kidx] * xp.cos(arg)
            line = term if line is None else line + term
        return line

    def _process(self, message: AxisArray) -> AxisArray:
        if self.settings.cancel_method == CancelMethod.PASSTHROUGH:
            # No cancellation and no frequency tracking; emit the input as-is.
            return message

        ax_idx = message.get_axis_idx(self.settings.axis)
        x_data = message.data
        xp, is_mlx = _namespace(x_data)
        moved = ax_idx != 0
        if moved:
            x_data = xp.moveaxis(x_data, ax_idx, 0)

        n = x_data.shape[0]
        st = self._state
        dtype = x_data.dtype
        fs = 1.0 / message.axes[self.settings.axis].gain

        # Time constants -> gains (independent of chunk size and fs).
        # mu = 2 / (tau_adapt * fs); beta = 1 - exp(-window_dt / tau_freq).
        mu = 2.0 / (self.settings.adapt_time_constant * fs)
        n_harm = max(1, int(self.settings.num_harmonics))
        tau_freq = self.settings.freq_time_constant
        tracking = tau_freq is not None and tau_freq > 0
        beta = 1.0 - np.exp(-(st.block_len / fs) / tau_freq) if tracking else 0.0

        if self.settings.cancel_method == CancelMethod.SUBTRACT:
            # Reconstruct-and-subtract: always walk the window grid (the per-
            # window estimate updates there); the FLL steps omega only if
            # tracking. The first window subtracts nothing while it learns.
            # Line-estimate EWMA gain from adapt_time_constant (per window).
            alpha = 1.0 - np.exp(-(st.block_len / fs) / self.settings.adapt_time_constant)
            pieces = []
            pos = 0
            while pos < n:
                seg_len = min(st.block_len - st.samples_in_block, n - pos)
                x_seg = x_data[pos : pos + seg_len]
                line_seg = self._reconstruct(seg_len, n_harm, xp)
                pieces.append(x_seg if line_seg is None else x_seg - line_seg)
                self._accumulate_subtract(x_seg, seg_len, n_harm, xp)
                st.phase = st.phase + st.omega * seg_len
                st.samples_in_block += seg_len
                pos += seg_len
                if st.samples_in_block >= st.block_len:
                    self._commit_subtract(n_harm, beta, alpha, tracking, xp)
                    st.samples_in_block = 0
            y_out = self._cat(pieces, is_mlx, xp)
        elif not tracking:
            # Notch, frozen NCO: constant frequency, whole chunk in one sosfilt.
            sos = self._ensure_sos(mu, n_harm, is_mlx)
            y_out = self._sosfilt(sos, x_data, is_mlx)
            st.phase = st.phase + st.omega * n
        else:
            # Notch, tracking: walk segments bounded by the frequency-update
            # window; when a window fills the FLL fires and omega may step.
            pieces = []
            pos = 0
            while pos < n:
                seg_len = min(st.block_len - st.samples_in_block, n - pos)
                x_seg = x_data[pos : pos + seg_len]
                sos = self._ensure_sos(mu, n_harm, is_mlx)
                y_notch = self._sosfilt(sos, x_seg, is_mlx)
                removed = x_seg - y_notch
                pieces.append(y_notch)
                self._accumulate_demod(removed, seg_len, xp)
                st.phase = st.phase + st.omega * seg_len
                st.samples_in_block += seg_len
                pos += seg_len
                if st.samples_in_block >= st.block_len:
                    self._freq_update(beta)
                    st.samples_in_block = 0
            y_out = self._cat(pieces, is_mlx, xp)

        if not is_mlx:
            y_out = xp.astype(y_out, dtype)  # scipy upcasts float32 -> float64
        if moved:
            y_out = xp.moveaxis(y_out, 0, ax_idx)
        return replace(message, data=y_out)


class AdaptiveLNC(
    BaseTransformerUnit[
        AdaptiveLNCSettings,
        AxisArray,
        AxisArray,
        AdaptiveLNCTransformer,
    ]
):
    SETTINGS = AdaptiveLNCSettings


def design_lnc_sos(omega: float, mu: float, num_harmonics: int = 1) -> npt.NDArray:
    """SOS notch cascade equivalent to the quadrature-LMS line canceller.

    A fixed-frequency quadrature-LMS canceller (unit references, step ``mu``,
    ``control=1``) is *exactly* a linear time-invariant 2nd-order notch (Glover
    1977 / Widrow). For the fundamental its input->output (``y = x - nr``)
    transfer function is::

        H(z) = (z^2 - 2cos(w)z + 1) / (z^2 - (2 - mu)cos(w)z + (1 - mu))

    i.e. zeros exactly on the unit circle at ``e^{+/-j w}`` (a perfect notch)
    and poles at radius ``sqrt(1 - mu)``. Harmonic ``k`` is the same notch at
    ``k * omega``; the cascade of the per-harmonic sections approximates the
    parallel multi-reference LMS to a small fraction of a dB (exact for a
    single harmonic). Verified against the per-sample LMS to ~1e-12 (float64).

    This is the building block for an Array-API / GPU (MLX ``sosfilt``)
    implementation that replaces the per-sample Python recursion.

    Parameters
    ----------
    omega : float
        Fundamental angular frequency in rad/sample (the FLL-tracked value).
    mu : float
        LMS step size, i.e. ``2 / (adapt_time_constant * fs)``.
    num_harmonics : int
        Number of harmonics (sections), ``k = 1 .. num_harmonics``.

    Returns
    -------
    np.ndarray
        ``(num_harmonics, 6)`` SOS array (``[b0, b1, b2, a0, a1, a2]`` per row,
        ``a0 = 1``), float64. Apply with ``scipy.signal.sosfilt`` (or the MLX
        ``sosfilt_mlx_metal``) carrying ``zi`` across chunks.
    """
    n_harm = max(1, int(num_harmonics))
    sos = np.empty((n_harm, 6), dtype=np.float64)
    for i in range(n_harm):
        c = np.cos((i + 1) * omega)
        sos[i] = (1.0, -2.0 * c, 1.0, 1.0, -(2.0 - mu) * c, 1.0 - mu)
    return sos
