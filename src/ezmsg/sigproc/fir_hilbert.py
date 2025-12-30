import functools
import typing

import ezmsg.core as ez
import numpy as np
import scipy.signal as sps
from ezmsg.baseproc import BaseStatefulTransformer, processor_state
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ezmsg.sigproc.filter import (
    BACoeffs,
    BaseFilterByDesignTransformerUnit,
    BaseTransformerUnit,
    FilterBaseSettings,
    FilterByDesignTransformer,
)


class FIRHilbertFilterSettings(FilterBaseSettings):
    """Settings for :obj:`FIRHilbertFilter`."""

    # axis inherited from FilterBaseSettings

    coef_type: str = "ba"
    """
    Coefficient type. Must be 'ba' for FIR.
    """

    order: int = 170
    """
    Filter order (taps = order + 1).
    Hilbert (type-III) filters require even order (odd taps).
    If odd order (even taps), order will be incremented by 1.
    """

    f_lo: float = 1.0
    """
    Lower corner of Hilbert “pass” band (Hz).
    Transition starts at f_lo.
    """

    f_hi: float | None = None
    """
    Upper corner of Hilbert “pass” band (Hz).
    Transition starts at f_hi.
    If None, highpass from f_lo to Nyquist.
    """

    trans_lo: float = 1.0
    """
    Transition width (Hz) below f_lo.
    Decrease to sharpen transition.
    """

    trans_hi: float = 1.0
    """
    Transition width (Hz) at high end.
    Decrease to sharpen transition.
    """

    weight_pass: float = 1.0
    """
    Weight for Hilbert pass region.
    """

    weight_stop_lo: float = 1.0
    """
    Weight for low stop band.
    """

    weight_stop_hi: float = 1.0
    """
    Weight for high stop band.
    """

    norm_band: tuple[float, float] | None = None
    """
    Optional normalization band (f_lo, f_hi) in Hz for gain normalization.
    If None, no normalization is applied.
    """

    norm_freq: float | None = None
    """
    Optional normalization frequency in Hz for gain normalization.
    If None, no normalization is applied.
    """


def fir_hilbert_design_fun(
    fs: float,
    order: int = 170,
    f_lo: float = 1.0,
    f_hi: float | None = None,
    trans_lo: float = 1.0,
    trans_hi: float = 1.0,
    weight_pass: float = 1.0,
    weight_stop_lo: float = 1.0,
    weight_stop_hi: float = 1.0,
    norm_band: tuple[float, float] | None = None,
    norm_freq: float | None = None,
) -> BACoeffs | None:
    """
    Hilbert FIR filter design using the Remez exchange algorithm.
    Design an `order`th-order FIR Hilbert filter and return the filter coefficients.
    See :obj:`FIRHilbertFilterSettings` for argument description.

    Returns:
        The filter coefficients as a tuple of (b, a).
    """
    if order <= 0:
        return None
    if order % 2 == 1:
        order += 1
    nyq = fs / 2.0
    taps = order + 1
    f1 = max(f_lo, 0.0) + trans_lo
    f2 = (nyq - trans_hi) if (f_hi is None) else min(f_hi, nyq - trans_hi)
    if not (0.0 < f1 < f2 < nyq):
        raise ValueError(
            f"Hilbert passband collapsed or invalid: "
            f"f_lo={f_lo}, f_hi={f_hi}, trans_lo={trans_lo}, trans_hi={trans_hi}, fs={fs}"
        )
    # Bands: [0, f1-trans_lo] stop ; [f1, f2] pass (Hilbert) ; [f2+trans_hi, nyq] stop
    bands = [0.0, max(f1 - trans_lo, 0.0), f1, f2, min(f2 + trans_hi, nyq), nyq]
    desired = [0.0, 1.0, 0.0]
    weight = [max(weight_stop_lo, 0.0), max(weight_pass, 0.0), max(weight_stop_hi, 0.0)]
    for i in range(1, len(bands) - 1):
        if bands[i] <= bands[i - 1]:
            bands[i] = np.nextafter(bands[i - 1], np.inf)
    if bands[-2] >= nyq:
        ez.logger.warning("Hilbert upper stopband collapsed; using 2-band (stop/pass) design.")
        bands = bands[:-3] + [nyq]
        desired = desired[:-1]
        weight = weight[:-1]
    b = sps.remez(taps, bands, desired, weight=weight, type="hilbert", fs=fs)
    a = np.array([1.0])
    g = None
    if norm_freq is not None:
        if norm_freq < f1 or norm_freq > f2:
            ez.logger.warning("Invalid normalization frequency specifications. Skipping normalization.")
        else:
            f0 = float(norm_freq)
            w = 2.0 * np.pi * (np.asarray([f0], dtype=np.float64) / fs)
            _, H = sps.freqz(b, a, worN=w)
            g = float(np.abs(H[0]))
    elif norm_band is not None:
        lo, hi = norm_band
        if lo < f1 or hi > f2:
            lo = max(lo, f1)
            hi = min(hi, f2)
            ez.logger.warning("Normalization band outside passband. Clipping to passband for normalization.")
        if lo >= hi:
            ez.logger.warning("Invalid normalization band specifications. Skipping normalization.")
        else:
            freqs = np.linspace(lo, hi, 2048, dtype=np.float64)
            w = 2.0 * np.pi * (np.asarray(freqs, dtype=np.float64) / fs)
            _, H = sps.freqz(b, a, worN=w)
            g = float(np.median(np.abs(H)))
    if g is not None and g > 0:
        b = b / g
    return (b, a)


class FIRHilbertFilterTransformer(FilterByDesignTransformer[FIRHilbertFilterSettings, BACoeffs]):
    def get_design_function(self) -> typing.Callable[[float], BACoeffs | None]:
        if self.settings.coef_type != "ba":
            ez.logger.error("FIRHilbert only supports coef_type='ba'.")
            raise ValueError("FIRHilbert only supports coef_type='ba'.")

        return functools.partial(
            fir_hilbert_design_fun,
            order=self.settings.order,
            f_lo=self.settings.f_lo,
            f_hi=self.settings.f_hi,
            trans_lo=self.settings.trans_lo,
            trans_hi=self.settings.trans_hi,
            weight_pass=self.settings.weight_pass,
            weight_stop_lo=self.settings.weight_stop_lo,
            weight_stop_hi=self.settings.weight_stop_hi,
            norm_band=self.settings.norm_band,
            norm_freq=self.settings.norm_freq,
        )

    def get_taps(self) -> int | None:
        if self._state.filter is None:
            return None
        b, _ = self._state.filter.settings.coefs
        return b.size if b is not None else None


class FIRHilbertFilterUnit(BaseFilterByDesignTransformerUnit[FIRHilbertFilterSettings, FIRHilbertFilterTransformer]):
    SETTINGS = FIRHilbertFilterSettings


@processor_state
class FIRHilbertEnvelopeState:
    filter: FIRHilbertFilterTransformer | None = None
    delay_buf: np.ndarray | None = None
    dly: int | None = None


class FIRHilbertEnvelopeTransformer(
    BaseStatefulTransformer[FIRHilbertFilterSettings, AxisArray, AxisArray, FIRHilbertEnvelopeState]
):
    """
    Processor for computing the envelope of a signal using the Hilbert transform.

    This processor applies a Hilbert FIR filter to the input signal to obtain the analytic signal, from which the
    envelope is computed.

    The processor expects and outputs `AxisArray` messages with a `"time"` (time) axis.

    Settings:
    ---------
    order : int
        Filter order (taps = order + 1).
        Hilbert (type-III) filters require even order (odd taps).
        If odd order (even taps), order will be incremented by 1.
    f_lo : float
        Lower corner of Hilbert “pass” band (Hz).
        Transition starts at f_lo.
    f_hi : float, optional
        Upper corner of Hilbert “pass” band (Hz).
        Transition starts at f_hi.
        If None, highpass from f_lo to Nyquist.
    trans_lo : float
        Transition width (Hz) below f_lo.
        Decrease to sharpen transition.
    trans_hi : float
        Transition width (Hz) above f_hi.
        Decrease to sharpen transition.
    weight_pass : float
        Weight for Hilbert pass region.
    weight_stop_lo : float
        Weight for low stop band.
    weight_stop_hi : float
        Weight for high stop band.
    norm_band : tuple(float, float), optional
        Optional normalization band (f_lo, f_hi) in Hz for gain normalization.
        If None, no normalization is applied.
    norm_freq : float, optional
        Optional normalization frequency in Hz for gain normalization.
        If None, no normalization is applied.

    Example:
    -----------------------------
    ```python
    processor = FIRHilbertEnvelopeTransformer(
        settings=FIRHilbertFilterSettings(
            order=170,
            f_lo=1.0,
            f_hi=50.0,
        )
    )
    ```

    """

    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[0]
        gain = getattr(self._state.filter, "gain", 0.0)
        axis_idx = message.get_axis_idx(axis)
        samp_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        return hash((message.key, samp_shape, gain))

    def _reset_state(self, message: AxisArray) -> None:
        self._state.filter = FIRHilbertFilterTransformer(settings=self.settings)
        self._state.delay_buf = None
        self._state.dly = None

    def _process(self, message: AxisArray) -> AxisArray:
        y_imag_msg = self._state.filter(message)
        y_imag = y_imag_msg.data

        axis_name = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis_name)
        if self._state.dly is None:
            taps = self._state.filter.get_taps()
            self._state.dly = (taps - 1) // 2

        x = message.data

        move_axis = False
        if axis_idx != x.ndim - 1:
            x = np.moveaxis(x, axis_idx, -1)
            y_imag = np.moveaxis(y_imag, axis_idx, -1)
            move_axis = True

        if self._state.delay_buf is None:
            lead_shape = x.shape[:-1]
            self._state.delay_buf = np.zeros(lead_shape + (self._state.dly,), dtype=x.dtype)

        x_cat = np.concatenate([self._state.delay_buf, x], axis=-1)
        x_delayed_full = x_cat[..., : -self._state.dly]
        y_real = x_delayed_full[..., -x.shape[-1] :]

        self._state.delay_buf = x_cat[..., -self._state.dly :].copy()

        analytic = y_real.astype(np.complex64) + 1j * y_imag.astype(np.complex64)
        out = np.abs(analytic)

        if move_axis:
            out = np.moveaxis(out, -1, axis_idx)

        return replace(message, data=out, axes=message.axes)


class FIRHilbertEnvelopeUnit(
    BaseTransformerUnit[
        FIRHilbertFilterSettings,
        AxisArray,
        AxisArray,
        FIRHilbertEnvelopeTransformer,
    ]
):
    """
    Unit wrapper for the `FIRHilbertEnvelopeTransformer`.

    This unit provides a plug-and-play interface for calculating the envelope using the FIR Hilbert transform on a
    signal in an ezmsg graph-based system. It takes in `AxisArray` inputs and outputs processed data in the same format.

    Example:
    --------
    ```python
    unit = FIRHilbertEnvelopeUnit(
        settings=FIRHilbertFilterSettings(
            order=170,
            f_lo=1.0,
            f_hi=50.0,
        )
    )
    ```
    """

    SETTINGS = FIRHilbertFilterSettings
