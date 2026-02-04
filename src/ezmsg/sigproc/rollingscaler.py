import math
from collections import deque

import ezmsg.core as ez
import numpy.typing as npt
from array_api_compat import get_namespace
from ezmsg.baseproc import (
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class RollingScalerSettings(ez.Settings):
    axis: str = "time"
    """
    Axis along which samples are arranged.
    """

    k_samples: int | None = 20
    """
    Rolling window size in number of samples.
    """

    window_size: float | None = None
    """
    Rolling window size in seconds.
    If set, overrides `k_samples`.
    `update_with_signal` likely should be True if using this option.
    """

    update_with_signal: bool = False
    """
    If True, update rolling statistics using the incoming process stream.
    """

    min_samples: int = 1
    """
    Minimum number of samples required to compute statistics.
    Used when `window_size` is not set.
    """

    min_seconds: float = 1.0
    """
    Minimum duration in seconds required to compute statistics.
    Used when `window_size` is set.
    """

    artifact_z_thresh: float | None = None
    """
    Threshold for z-score based artifact detection.
    If set, samples with any channel exceeding this z-score will be excluded
    from updating the rolling statistics.
    """

    clip: float | None = 10.0
    """
    If set, clip the output values to the range [-clip, clip].
    """


@processor_state
class RollingScalerState:
    mean: npt.NDArray | None = None
    N: int = 0
    M2: npt.NDArray | None = None
    samples: deque | None = None
    k_samples: int | None = None
    min_samples: int | None = None


class RollingScalerProcessor(BaseAdaptiveTransformer[RollingScalerSettings, AxisArray, AxisArray, RollingScalerState]):
    """
    Processor for rolling z-score normalization of input `AxisArray` messages.

    The processor maintains rolling statistics (mean and variance) over the last `k_samples`
    samples received via the `partial_fit()` method. When processing an `AxisArray` message,
    it normalizes the data using the current rolling statistics.

    The input `AxisArray` messages are expected to have shape `(time, ch)`, where `ch` is the
    channel axis. The processor computes the z-score for each channel independently.

    Note: You should consider instead using the AdaptiveStandardScalerTransformer which
    is computationally more efficient and uses less memory. This RollingScalerProcessor
    is primarily provided to reproduce processing in the literature.

    Settings:
    ---------
    k_samples: int
        Number of previous samples to use for rolling statistics.

    Example:
    -----------------------------
    ```python
    processor = RollingScalerProcessor(
        settings=RollingScalerSettings(
            k_samples=20  # Number of previous samples to use for rolling statistics
        )
    )
    ```
    """

    def _hash_message(self, message: AxisArray) -> int:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        gain = message.axes[axis].gain if hasattr(message.axes[axis], "gain") else 1
        axis_idx = message.get_axis_idx(axis)
        samp_shape = message.data.shape[:axis_idx] + message.data.shape[axis_idx + 1 :]
        return hash((message.key, samp_shape, gain))

    def _reset_state(self, message: AxisArray) -> None:
        xp = get_namespace(message.data)
        ch = message.data.shape[-1]
        self._state.mean = xp.zeros(ch, dtype=xp.float64)
        self._state.N = 0
        self._state.M2 = xp.zeros(ch, dtype=xp.float64)
        self._state.k_samples = (
            math.ceil(self.settings.window_size / message.axes[self.settings.axis].gain)
            if self.settings.window_size is not None
            else self.settings.k_samples
        )
        if self._state.k_samples is not None and self._state.k_samples < 1:
            ez.logger.warning("window_size smaller than sample gain; setting k_samples to 1.")
            self._state.k_samples = 1
        elif self._state.k_samples is None:
            ez.logger.warning("k_samples is None; z-score accumulation will be unbounded.")
        self._state.samples = deque(maxlen=self._state.k_samples)
        self._state.min_samples = (
            math.ceil(self.settings.min_seconds / message.axes[self.settings.axis].gain)
            if self.settings.window_size is not None
            else self.settings.min_samples
        )
        if self._state.k_samples is not None and self._state.min_samples > self._state.k_samples:
            ez.logger.warning("min_samples is greater than k_samples; adjusting min_samples to k_samples.")
            self._state.min_samples = self._state.k_samples

    def _add_batch_stats(self, x: npt.NDArray) -> None:
        xp = get_namespace(x)
        x = xp.asarray(x, dtype=xp.float64)
        n_b = x.shape[0]
        mean_b = xp.mean(x, axis=0)
        M2_b = xp.sum((x - mean_b) ** 2, axis=0)

        if self._state.k_samples is not None and len(self._state.samples) == self._state.k_samples:
            n_old, mean_old, M2_old = self._state.samples.popleft()
            N_T = self._state.N
            N_new = N_T - n_old

            if N_new <= 0:
                self._state.N = 0
                self._state.mean = xp.zeros_like(self._state.mean)
                self._state.M2 = xp.zeros_like(self._state.M2)
            else:
                delta = mean_old - self._state.mean
                self._state.N = N_new
                self._state.mean = (N_T * self._state.mean - n_old * mean_old) / N_new
                self._state.M2 = self._state.M2 - M2_old - (delta * delta) * (N_T * n_old / N_new)

        N_A = self._state.N
        N = N_A + n_b
        delta = mean_b - self._state.mean
        self._state.mean = self._state.mean + delta * (n_b / N)
        self._state.M2 = self._state.M2 + M2_b + (delta * delta) * (N_A * n_b / N)
        self._state.N = N

        self._state.samples.append((n_b, mean_b, M2_b))

    def partial_fit(self, message: AxisArray) -> None:
        x = message.data
        self._add_batch_stats(x)

    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)

        if self._state.N == 0 or self._state.N < self._state.min_samples:
            if self.settings.update_with_signal:
                x = message.data
                if self.settings.artifact_z_thresh is not None and self._state.N > 0:
                    varis = self._state.M2 / self._state.N
                    raw_std = varis**0.5
                    std = xp.where(xp.isnan(raw_std), raw_std, xp.clip(raw_std, min=1e-8))
                    z = xp.abs((x - self._state.mean) / std)
                    mask = xp.any(z > self.settings.artifact_z_thresh, axis=1)
                    x = x[~mask]
                if x.size > 0:
                    self._add_batch_stats(x)
            return message

        varis = self._state.M2 / self._state.N
        raw_std = varis**0.5
        # Preserve NaN from negative variance (will be caught below), clip positive std to floor
        std = xp.where(xp.isnan(raw_std), raw_std, xp.clip(raw_std, min=1e-8))
        result = (message.data - self._state.mean) / std
        # Replace NaN/inf with 0 (equivalent to nan_to_num with nan=0, posinf=0, neginf=0)
        result = xp.where(xp.isfinite(result), result, xp.asarray(0.0, dtype=result.dtype))
        if self.settings.clip is not None:
            result = xp.clip(result, -self.settings.clip, self.settings.clip)

        if self.settings.update_with_signal:
            x = message.data
            if self.settings.artifact_z_thresh is not None:
                z_scores = xp.abs((x - self._state.mean) / std)
                mask = xp.any(z_scores > self.settings.artifact_z_thresh, axis=1)
                x = x[~mask]
            if x.size > 0:
                self._add_batch_stats(x)

        return replace(message, data=result)


class RollingScalerUnit(
    BaseAdaptiveTransformerUnit[
        RollingScalerSettings,
        AxisArray,
        AxisArray,
        RollingScalerProcessor,
    ]
):
    """
    Unit wrapper for :obj:`RollingScalerProcessor`.

    This unit performs rolling z-score normalization on incoming `AxisArray` messages. The unit maintains rolling
    statistics (mean and variance) over the last `k_samples` samples received. When processing an `AxisArray` message,
    it normalizes the data using the current rolling statistics.

    Example:
    -----------------------------
    ```python
    unit = RollingScalerUnit(
        settings=RollingScalerSettings(
            k_samples=20  # Number of previous samples to use for rolling statistics
        )
    )
    ```
    """

    SETTINGS = RollingScalerSettings
