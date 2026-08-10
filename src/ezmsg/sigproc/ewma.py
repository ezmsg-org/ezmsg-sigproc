"""Exponentially weighted moving average (EWMA) utilities and parameter conversion."""

import functools
from dataclasses import field

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import scipy.signal as sps
from array_api_compat import get_namespace, is_numpy_array
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.messages.util import replace


def _ewma_mlx_metal_xp(data, axis_idx: int, zi, alpha: float, chunk_sizes: tuple[int, ...]):
    """Run EWMA through the MLX Metal helper while preserving scipy zi layout."""
    import mlx.core as mx

    from .util.ewma_mlx_metal import ewma_mlx_metal

    zi = mx.asarray(zi, dtype=data.dtype)
    last_data_axis = data.ndim - 1
    last_zi_axis = zi.ndim - 1
    x_mx = mx.moveaxis(data, axis_idx, last_data_axis) if axis_idx != last_data_axis else data
    zi_mx = mx.moveaxis(zi, axis_idx, last_zi_axis) if axis_idx != last_zi_axis else zi
    y_mx, zf_mx = ewma_mlx_metal(x_mx, alpha, zi_mx, chunk_sizes=chunk_sizes)
    y = mx.moveaxis(y_mx, last_data_axis, axis_idx) if axis_idx != last_data_axis else y_mx
    zf = mx.moveaxis(zf_mx, last_zi_axis, axis_idx) if axis_idx != last_zi_axis else zf_mx
    return y, zf


def _tau_from_alpha(alpha: float, dt: float) -> float:
    """
    Inverse of _alpha_from_tau. See that function for explanation.
    """
    return -dt / np.log(1 - alpha)


def _alpha_from_tau(tau: float, dt: float) -> float:
    """
    # https://en.wikipedia.org/wiki/Exponential_smoothing#Time_constant
    :param tau: The amount of time for the smoothed response of a unit step function to reach
        1 - 1/e approx-eq 63.2%.
    :param dt: sampling period, or 1 / sampling_rate.
    :return: alpha, the "fading factor" in exponential smoothing.
    """
    return 1 - np.exp(-dt / tau)


def ewma_step(sample: npt.NDArray, zi: npt.NDArray, alpha: float, beta: float | None = None):
    """
    Do an exponentially weighted moving average step.

    Args:
        sample: The new sample.
        zi: The output of the previous step.
        alpha: Fading factor.
        beta: Persisting factor. If None, it is calculated as 1-alpha.

    Returns:
        alpha * sample + beta * zi

    """
    # Potential micro-optimization:
    #  Current: scalar-arr multiplication, scalar-arr multiplication, arr-arr addition
    #  Alternative: arr-arr subtraction, arr-arr multiplication, arr-arr addition
    # return zi + alpha * (new_sample - zi)
    beta = beta or (1 - alpha)
    return alpha * sample + beta * zi


class EWMA_Deprecated:
    """
    Grabbed these methods from https://stackoverflow.com/a/70998068 and other answers in that topic,
    but they ended up being slower than the scipy.signal.lfilter method.
    Additionally, `compute` and `compute2` suffer from potential errors as the vector length increases
    and beta**n approaches zero.
    """

    def __init__(self, alpha: float, max_len: int):
        self.alpha = alpha
        self.beta = 1 - alpha
        self.prev: npt.NDArray | None = None
        self.weights = np.empty((max_len + 1,), float)
        self._precalc_weights(max_len)
        self._step_func = functools.partial(ewma_step, alpha=self.alpha, beta=self.beta)

    def _precalc_weights(self, n: int):
        #   (1-α)^0, (1-α)^1, (1-α)^2, ..., (1-α)^n
        np.power(self.beta, np.arange(n + 1), out=self.weights)

    def compute(self, arr: npt.NDArray, out: npt.NDArray | None = None) -> npt.NDArray:
        if out is None:
            out = np.empty(arr.shape, arr.dtype)

        n = arr.shape[0]
        weights = self.weights[:n]
        weights = np.expand_dims(weights, list(range(1, arr.ndim)))

        #   α*P0, α*P1, α*P2, ..., α*Pn
        np.multiply(self.alpha, arr, out)

        #   α*P0/(1-α)^0, α*P1/(1-α)^1, α*P2/(1-α)^2, ..., α*Pn/(1-α)^n
        np.divide(out, weights, out)

        #   α*P0/(1-α)^0, α*P0/(1-α)^0 + α*P1/(1-α)^1, ...
        np.cumsum(out, axis=0, out=out)

        #   (α*P0/(1-α)^0)*(1-α)^0, (α*P0/(1-α)^0 + α*P1/(1-α)^1)*(1-α)^1, ...
        np.multiply(out, weights, out)

        # Add the previous output
        if self.prev is None:
            self.prev = arr[:1]

        out += self.prev * np.expand_dims(self.weights[1 : n + 1], list(range(1, arr.ndim)))

        self.prev = out[-1:]

        return out

    def compute2(self, arr: npt.NDArray) -> npt.NDArray:
        """
        Compute the Exponentially Weighted Moving Average (EWMA) of the input array.

        Args:
            arr: The input array to be smoothed.

        Returns:
            The smoothed array.
        """
        n = arr.shape[0]
        if n > len(self.weights):
            self._precalc_weights(n)
        weights = self.weights[:n][::-1]
        weights = np.expand_dims(weights, list(range(1, arr.ndim)))

        result = np.cumsum(self.alpha * weights * arr, axis=0)
        result = result / weights

        # Handle the first call when prev is unset
        if self.prev is None:
            self.prev = arr[:1]

        result += self.prev * np.expand_dims(self.weights[1 : n + 1], list(range(1, arr.ndim)))

        # Store the result back into prev
        self.prev = result[-1]

        return result

    def compute_sample(self, new_sample: npt.NDArray) -> npt.NDArray:
        if self.prev is None:
            self.prev = new_sample
        self.prev = self._step_func(new_sample, self.prev)
        return self.prev


class EWMASettings(ez.Settings):
    time_constant: float = 1.0
    """The amount of time for the smoothed response of a unit step function to reach 1 - 1/e approx-eq 63.2%."""

    axis: str | None = None

    accumulate: bool = True
    """If True, update the EWMA state with each sample. If False, only apply
    the current EWMA estimate without updating state (useful for inference
    periods where you don't want to adapt statistics)."""

    passthrough: bool = False
    """If True, return the input unchanged (identity) without touching the
    EWMA. Unlike a very large time_constant -- which still applies a (stale)
    baseline estimate -- passthrough leaves the data untouched. May be toggled
    at runtime without resetting the filter state; see ``reset_on_resume`` for
    what that means for the first message after the gap."""

    reset_on_resume: bool = False
    """Whether switching ``passthrough`` back off discards the filter state.

    The filter sees none of the samples that go by during passthrough, so ``zi``
    describes an exponentially-weighted window that ended when passthrough was
    switched on -- and the state cannot tell a 10 ms blip from a 10 minute
    outage, since both resume identically. For a scaler that means z-scoring
    post-gap data against pre-gap statistics.

    False (the default) resumes from the preserved state, which is right for a
    short blip and keeps an estimate that may have taken many ``time_constant``\\ s
    to converge. True rebuilds from the first post-gap message instead: with the
    bias correction below, that first output is exactly the first sample, and the
    estimate re-converges over ``time_constant``. Prefer True where passthrough
    may be left on long enough for the signal to drift, which is the case
    :obj:`ezmsg.sigproc.binned_aggregate.BinnedAggregateTransformer` always
    assumes.

    Empty chunks are not gaps -- they carry no samples past the filter -- so they
    never trigger this."""

    mlx_metal_chunk_sizes: tuple[int, ...] = (32, 1024)
    """Allowable compile-time chunk sizes for EWMA Metal kernels. The smallest
    size that fits the remaining samples is selected on each launch; otherwise
    the largest size is repeated. Specializations compile lazily on first use.
    Values must be in ``[1, 1024]``."""


@processor_state
class EWMAState:
    alpha: float = field(default_factory=lambda: _alpha_from_tau(1.0, 1000.0))
    zi: npt.NDArray | None = None
    n_seen: int = 0
    """Cumulative sample count since reset, used to bias-correct the output."""


class EWMATransformer(BaseStatefulTransformer[EWMASettings, AxisArray, AxisArray, EWMAState]):
    # `accumulate` is read live in `_process` to gate state updates and
    # `passthrough`/`reset_on_resume` are read live in `__call__`/`__acall__`;
    # other fields are cached into state (alpha, zi) during `_reset_state`.
    NONRESET_SETTINGS_FIELDS = frozenset({"accumulate", "passthrough", "reset_on_resume", "mlx_metal_chunk_sizes"})

    def _skip(self, message: AxisArray) -> bool:
        """Whether to bypass the filter, flagging a reset if this is a real gap.

        The two bypass conditions have to be kept apart: passthrough lets samples
        past unfiltered and so leaves a hole in ``zi``'s history, while an empty
        chunk carries nothing past and leaves the history intact. Only the former
        is a gap, so only the former can invalidate the state.
        """
        if self.settings.passthrough:
            if self.settings.reset_on_resume and np.prod(message.data.shape) != 0:
                self._request_reset()
            return True
        return bool(np.prod(message.data.shape) == 0)

    def __call__(self, message: AxisArray) -> AxisArray:
        if self._skip(message):
            return message
        return super().__call__(message)

    async def __acall__(self, message: AxisArray) -> AxisArray:
        if self._skip(message):
            return message
        return await super().__acall__(message)

    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        sample_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        return hash((sample_shape, message.axes[axis].gain, message.key))

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)
        self._state.alpha = _alpha_from_tau(self.settings.time_constant, message.axes[axis].gain)
        # Start from zero; _process divides out the missing-history bias.
        sub_dat = slice_along_axis(message.data, slice(None, 1, None), axis=axis_idx)
        xp = np if is_numpy_array(message.data) else get_namespace(message.data)
        self._state.zi = xp.zeros_like(sub_dat)
        self._state.n_seen = 0

    def _lfilter_axis_last(
        self, data: npt.NDArray, axis_idx: int, zi: npt.NDArray | None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Run the EWMA recurrence with the filter axis contiguous.

        scipy's IIR loop strides by the trailing dimension whenever the filter
        axis is not last, and the cost grows *superlinearly* with that dimension:
        at 300 samples the step from 256 to 1024 channels is 4x the data but 8.7x
        the time, versus linear when the axis is already last. Acquisition sources
        emit ``(time, ch)``, so the streaming case hits the bad orientation by
        default.

        Hoisting the axis to the end and moving the result back costs two copies
        that pay for themselves at every size measured, and never lose:
        1.04x at 300x256, 1.78x at 30x1024, 4.41x at 300x1024, 1.72x at 3000x1024
        -- all *including* the copies. ``filter.py``'s SOS kernel already does
        this (``util/sosfilt_direct``), which is why ``ButterworthZeroPhase``
        measures layout-neutral while this did not.

        ``zi`` is one sample slice, so hoisting it too is negligible and keeps the
        stored state in the caller's layout for anything that inspects it.
        """
        b = [self._state.alpha]
        a = [1.0, self._state.alpha - 1.0]
        last = data.ndim - 1
        if axis_idx == last:
            return sps.lfilter(b, a, data, axis=-1, zi=zi)

        x = np.ascontiguousarray(np.moveaxis(data, axis_idx, last))
        zi_last = None if zi is None else np.ascontiguousarray(np.moveaxis(zi, axis_idx, last))
        y, zf = sps.lfilter(b, a, x, axis=-1, zi=zi_last)
        # Materialize back into the caller's layout rather than returning a
        # transposed view. The view saves a full-size pass here and is 12-17%
        # faster *in isolation*, but it loses end to end at the sizes that matter
        # (measured on the whole scaler at 1024 ch: 6% worse at 300 samples, 2%
        # worse at 1000, 5% better only by 3000) because every downstream op then
        # reads strided. Keep the copy; it is also the less surprising contract
        # for a library consumer.
        return (
            np.ascontiguousarray(np.moveaxis(y, last, axis_idx)),
            np.ascontiguousarray(np.moveaxis(zf, last, axis_idx)),
        )

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis or message.dims[0]
        axis_idx = message.get_axis_idx(axis)

        xp = np if is_numpy_array(message.data) else get_namespace(message.data)
        if xp is not np and xp.__name__ == "mlx.core":
            expected, zf = _ewma_mlx_metal_xp(
                message.data,
                axis_idx,
                self._state.zi,
                self._state.alpha,
                self.settings.mlx_metal_chunk_sizes,
            )
            if self.settings.accumulate:
                self._state.zi = zf
        elif self.settings.accumulate:
            # Normal behavior: update state with new samples.
            if self._state.zi is not None and not is_numpy_array(self._state.zi):
                self._state.zi = np.asarray(self._state.zi)
            expected, self._state.zi = self._lfilter_axis_last(message.data, axis_idx, self._state.zi)
        else:
            # Process-only: compute output without updating state.
            if self._state.zi is not None and not is_numpy_array(self._state.zi):
                self._state.zi = np.asarray(self._state.zi)
            expected, _ = self._lfilter_axis_last(message.data, axis_idx, self._state.zi)

        # The zero-initialized EWMA under-counts by 1-(1-alpha)^t at cumulative
        # sample t; dividing it out gives the exact exponentially-weighted
        # average of the samples seen so far (the "Adam" bias correction).
        n = message.data.shape[axis_idx]
        t = self._state.n_seen + np.arange(1, n + 1)
        corr = 1.0 - (1.0 - self._state.alpha) ** t
        corr = corr.reshape([n if i == axis_idx else 1 for i in range(message.data.ndim)])
        expected = expected / xp.asarray(corr, dtype=expected.dtype)
        if self.settings.accumulate:
            self._state.n_seen += n
        return replace(message, data=expected)


class EWMAUnit(BaseTransformerUnit[EWMASettings, AxisArray, AxisArray, EWMATransformer]):
    SETTINGS = EWMASettings
