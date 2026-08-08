"""Call scipy's SOS kernel directly, skipping the per-call wrapper cost.

``scipy.signal.sosfilt`` spends a fixed ~12-18 us per call on validation, dtype
resolution and shape bookkeeping before reaching its Cython kernel. That cost is
independent of chunk size, so it is invisible offline and dominant online:

    n_ch x N     sosfilt   direct   saved
     16 x 30     15.3 us   6.1 us   59.9%
     32 x 30     18.4 us   9.0 us   51.0%
     64 x 30     23.8 us  14.6 us   38.8%
    256 x 30     59.2 us  49.3 us   16.6%
    256 x 128   207.0 us 198.4 us    4.1%

This module reproduces ``sosfilt``'s semantics exactly -- same dtype promotion,
same layout, same Cython kernel -- with the invariant work hoisted into
:class:`DirectSosfilt`, built once per stream. Output is bit-identical to
``scipy.signal.sosfilt``, which is the point: it must be a pure latency
optimization, never a numerical one.

It depends on ``scipy.signal._sosfilt._sosfilt``, a private entry point. Two
guards cover that: the import is optional, and :func:`available` verifies the
private kernel against the public function once per process before anything
trusts it. If either fails, callers fall back to ``scipy.signal.sosfilt`` and
the only consequence is the lost microseconds.

Note the state (``zi``) is deliberately kept in scipy's public layout and
converted per call. Keeping it in the kernel's layout would save a further
~10-15% but would change the meaning of ``FilterState.zi``, which other code
reads; the conversion is cheap enough that the compatibility is worth more.
"""

import typing

import numpy as np
import numpy.typing as npt

try:
    from scipy.signal._sosfilt import _sosfilt as _sosfilt_kernel

    _HAS_KERNEL = True
except ImportError:  # pragma: no cover - depends on the installed scipy build
    _HAS_KERNEL = False

#: dtypes the Cython kernel handles and for which we have verified equivalence.
#: Anything else (longdouble, object, complex) falls back to public scipy.
_SUPPORTED_DTYPES = frozenset({np.dtype(np.float32), np.dtype(np.float64)})

_verified: bool | None = None


def available() -> bool:
    """Whether the direct path may be used, verifying the private kernel once.

    The check filters a small random signal through both the private kernel and
    ``scipy.signal.sosfilt`` and requires bit-identical output and state. It runs
    at most once per process and costs well under a millisecond.
    """
    global _verified
    if _verified is not None:
        return _verified
    if not _HAS_KERNEL:
        _verified = False
        return False
    try:
        import scipy.signal

        sos = scipy.signal.butter(4, 0.25, output="sos")
        rng = np.random.default_rng(0)
        x = rng.standard_normal((3, 37))
        zi = rng.standard_normal((sos.shape[0], 3, 2))
        want, want_zf = scipy.signal.sosfilt(sos, x, axis=-1, zi=zi)
        got, got_zf = DirectSosfilt(sos, np.float64).apply(x, -1, zi)
        _verified = bool(np.array_equal(want, got) and np.array_equal(want_zf, got_zf))
    except Exception:  # pragma: no cover - defensive; any failure disables the path
        _verified = False
    return _verified


def can_apply(sos: npt.NDArray, data: typing.Any, zi: typing.Any) -> bool:
    """Whether ``data``/``zi`` are shaped and typed for the direct path."""
    if not isinstance(data, np.ndarray) or not isinstance(zi, np.ndarray):
        return False
    return np.result_type(sos, data, zi) in _SUPPORTED_DTYPES


class DirectSosfilt:
    """A SOS filter with scipy's per-call setup hoisted to construction.

    Args:
        sos: ``(n_sections, 6)`` coefficients.
        dtype: The promoted dtype the stream will run in. Must match what
            ``np.result_type(sos, data, zi)`` yields, or :meth:`apply` would
            diverge from scipy; :func:`can_apply` is the guard for that.
    """

    __slots__ = ("_sos", "_n_sections", "_dtype")

    def __init__(self, sos: npt.NDArray, dtype: npt.DTypeLike):
        sos = np.asarray(sos)
        if sos.ndim != 2 or sos.shape[1] != 6:
            raise ValueError(f"sos must have shape (n_sections, 6); got {tuple(sos.shape)}")
        self._dtype = np.dtype(dtype)
        # Hoisted: contiguity and dtype conversion of the coefficients.
        self._sos = np.ascontiguousarray(sos, dtype=self._dtype)
        self._n_sections = sos.shape[0]

    def apply(
        self,
        data: npt.NDArray,
        axis_idx: int,
        zi: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Filter ``data`` along ``axis_idx``, returning ``(y, zf)``.

        Mirrors ``scipy.signal.sosfilt`` step for step so the result is
        bit-identical: move the sample axis last, flatten to 2-D, copy into a
        C-contiguous buffer (the kernel works in place), run, restore shapes.
        """
        n_sections = self._n_sections
        axis = axis_idx % data.ndim

        x = np.moveaxis(data, axis, -1)
        z = np.moveaxis(zi, (0, axis + 1), (-2, -1))
        x_shape, zi_shape = x.shape, z.shape

        x = np.reshape(x, (-1, x.shape[-1]))
        # The kernel mutates its input; this copy is what scipy does too, and is
        # also what keeps us from writing through a caller's view.
        x = np.array(x, self._dtype, order="C")
        z = np.ascontiguousarray(np.reshape(z, (-1, n_sections, 2)), dtype=self._dtype)

        _sosfilt_kernel(self._sos, x, z)

        x = np.moveaxis(x.reshape(x_shape), -1, axis)
        z = np.moveaxis(z.reshape(zi_shape), (-2, -1), (0, axis + 1))
        return x, z
