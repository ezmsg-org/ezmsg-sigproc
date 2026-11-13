"""Minimum-phase conversion of FIR kernels, real or complex."""

import numpy as np
import numpy.typing as npt
import scipy.signal as sps

from ..spectrum import OptionsEnum


class MinPhaseMode(OptionsEnum):
    """How to convert a linear-phase FIR kernel to minimum phase."""

    NONE = "No kernel modification"
    HILBERT = (
        "Hilbert Method; designed to be used with equiripple filters (e.g., from remez) with unity or zero gain regions"
    )
    HOMOMORPHIC = (
        "Works best with filters with an odd number of taps, and the resulting minimum phase filter "
        "will have a magnitude response that approximates the square root of the original filter’s "
        "magnitude response using half the number of taps"
    )
    HOMOMORPHICFULL = "Like HOMOMORPHIC, but uses the full number of taps and same magnitude"


def _complex_homomorphic(h: npt.NDArray, half: bool) -> npt.NDArray:
    """Homomorphic minimum phase for a complex kernel, mirroring scipy's real-only routine.

    The log-magnitude spectrum is real even when ``h`` is complex, so its cepstrum is
    Hermitian and the usual fold onto positive quefrencies still yields the causal,
    minimum-phase cepstrum. Unlike the real case the result is kept complex.
    """
    n = h.shape[0]
    n_fft = 2 ** int(np.ceil(np.log2(2 * (n - 1) / 0.01)))  # scipy.signal.minimum_phase default
    mag = np.abs(np.fft.fft(h, n_fft))
    log_mag = np.log(np.maximum(mag, 1e-7 * mag.max()))
    if half:
        log_mag *= 0.5
    win = np.zeros(n_fft)
    win[0] = 1
    win[1 : n_fft // 2] = 2
    win[n_fft // 2] = 1  # n_fft is a power of 2, so this is the Nyquist bin
    h_min = np.fft.ifft(np.exp(np.fft.fft(np.fft.ifft(log_mag) * win)))
    return h_min[: (n + 1) // 2 if half else n]


def min_phase_kernel(h: npt.NDArray, mode: MinPhaseMode) -> npt.NDArray:
    """Convert a linear-phase FIR kernel to its minimum-phase equivalent.

    Real kernels go to :func:`scipy.signal.minimum_phase`. Complex kernels (e.g. analytic
    wavelets) use an equivalent homomorphic routine; the Hilbert method has no complex
    counterpart and raises.

    Args:
        h: 1-D FIR kernel.
        mode: HILBERT and HOMOMORPHIC return ``(len(h) + 1) // 2`` taps with approximately
            the square root of the original magnitude response; HOMOMORPHICFULL returns
            ``len(h)`` taps with approximately the original magnitude response; NONE
            returns ``h`` unchanged.

    Returns:
        The converted kernel.
    """
    if mode == MinPhaseMode.NONE:
        return h
    if np.iscomplexobj(h):
        if mode == MinPhaseMode.HILBERT:
            raise ValueError(
                "MinPhaseMode.HILBERT does not support complex kernels; "
                "use MinPhaseMode.HOMOMORPHIC or MinPhaseMode.HOMOMORPHICFULL."
            )
        if h.ndim != 1 or h.shape[0] <= 2:
            raise ValueError("h must be 1-D and at least 3 samples long")
        return _complex_homomorphic(h, half=mode == MinPhaseMode.HOMOMORPHIC)
    if mode == MinPhaseMode.HILBERT:
        return sps.minimum_phase(h, method="hilbert")
    return sps.minimum_phase(h, method="homomorphic", half=mode == MinPhaseMode.HOMOMORPHIC)


def is_linear_phase(h: npt.NDArray) -> bool:
    """Whether a 1-D FIR kernel is (generalized) linear phase.

    True for symmetric and antisymmetric real kernels and for conjugate-symmetric or
    conjugate-antisymmetric complex ones, i.e. ``h == ±conj(h[::-1])``.
    """
    h = np.asarray(h)
    h_rev = np.conj(h[::-1])
    atol = 1e-9 * float(np.max(np.abs(h), initial=0.0))
    return bool(np.allclose(h, h_rev, rtol=0, atol=atol) or np.allclose(h, -h_rev, rtol=0, atol=atol))


def kernel_delay(h: npt.NDArray) -> float:
    """Delay of a 1-D FIR kernel, in samples.

    Exactly ``(len(h) - 1) / 2`` for a linear-phase kernel (see :func:`is_linear_phase`).
    Otherwise the group delay at the frequency where ``|H|`` peaks -- the envelope delay at
    band centre, which is an approximation because the delay varies with frequency.
    """
    h = np.asarray(h)
    if is_linear_phase(h):
        return (h.shape[0] - 1) / 2
    whole = np.iscomplexobj(h)  # a complex kernel's response is not mirrored about DC
    n_freqs = max(1024, 8 * h.shape[0])
    w, resp = sps.freqz(h, worN=n_freqs, whole=whole)
    _, gd = sps.group_delay((h, 1), w=[w[np.argmax(np.abs(resp))]], whole=whole)
    return float(gd[0])
