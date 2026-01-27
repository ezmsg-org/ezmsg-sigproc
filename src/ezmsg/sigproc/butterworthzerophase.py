"""
Streaming zero-phase Butterworth filter implemented as a two-stage composite processor.

Stage 1: Forward causal Butterworth filter (from ezmsg.sigproc.butterworthfilter)
Stage 2: Backward acausal filter with buffering (ButterworthBackwardFilterTransformer)

The output is delayed by `pad_length` samples to ensure the backward pass has sufficient
future context. The pad_length is computed analytically using scipy's heuristic.
"""

import functools
import typing

import numpy as np
import scipy.signal
from ezmsg.baseproc import BaseTransformerUnit
from ezmsg.baseproc.composite import CompositeProcessor
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .butterworthfilter import (
    ButterworthFilterSettings,
    ButterworthFilterTransformer,
    butter_design_fun,
)
from .filter import BACoeffs, FilterByDesignTransformer, SOSCoeffs
from .util.axisarray_buffer import HybridAxisArrayBuffer


class ButterworthZeroPhaseSettings(ButterworthFilterSettings):
    """
    Settings for :obj:`ButterworthZeroPhase`.

    This implements a streaming zero-phase Butterworth filter using forward-backward
    filtering. The output is delayed by `pad_length` samples to ensure the backward
    pass has sufficient future context.

    The pad_length is computed by finding where the filter's impulse response decays
    to `settle_cutoff` fraction of its peak value. This accounts for the filter's
    actual time constant rather than just its order.
    """

    # Inherits from ButterworthFilterSettings:
    # axis, coef_type, order, cuton, cutoff, wn_hz

    settle_cutoff: float = 0.01
    """
    Fraction of peak impulse response used to determine settling time.
    The pad_length is set to the number of samples until the impulse response
    decays to this fraction of its peak. Default is 0.01 (1% of peak).
    """

    max_pad_duration: float | None = None
    """
    Maximum pad duration in seconds. If set, the pad_length will be capped
    at this value times the sampling rate. Use this to limit latency for
    filters with very long impulse responses. Default is None (no limit).
    """


class ButterworthBackwardFilterTransformer(FilterByDesignTransformer[ButterworthFilterSettings, BACoeffs | SOSCoeffs]):
    """
    Backward (acausal) Butterworth filter with buffering.

    This transformer buffers its input and applies the filter in reverse,
    outputting only the "settled" portion where transients have decayed.
    This introduces a lag of ``pad_length`` samples.

    Intended to be used as stage 2 in a zero-phase filter pipeline, receiving
    forward-filtered data from a ButterworthFilterTransformer.
    """

    # Instance attributes (initialized in _reset_state)
    _buffer: HybridAxisArrayBuffer | None
    _coefs_cache: BACoeffs | SOSCoeffs | None
    _zi_tiled: np.ndarray | None
    _pad_length: int

    def get_design_function(
        self,
    ) -> typing.Callable[[float], BACoeffs | SOSCoeffs | None]:
        return functools.partial(
            butter_design_fun,
            order=self.settings.order,
            cuton=self.settings.cuton,
            cutoff=self.settings.cutoff,
            coef_type=self.settings.coef_type,
            wn_hz=self.settings.wn_hz,
        )

    def _compute_pad_length(self, fs: float) -> int:
        """
        Compute pad length based on the filter's impulse response settling time.

        The pad_length is determined by finding where the impulse response decays
        to `settle_cutoff` fraction of its peak value. This is then optionally
        capped by `max_pad_duration`.

        Args:
            fs: Sampling frequency in Hz.

        Returns:
            Number of samples for the pad length.
        """
        # Design the filter to compute impulse response
        coefs = self.get_design_function()(fs)
        if coefs is None:
            # Filter design failed or is disabled
            return 0

        # Generate impulse response - use a generous length initially
        # Start with scipy's heuristic as minimum, then extend if needed
        if self.settings.coef_type == "ba":
            min_length = 3 * (self.settings.order + 1)
        else:
            n_sections = (self.settings.order + 1) // 2
            min_length = 3 * n_sections * 2

        # Use 10x the minimum as initial impulse length, or at least 10000 samples
        # (10000 samples allows for ~333ms at 30kHz, covering most practical cases)
        impulse_length = max(min_length * 10, 10000)

        # Cap impulse length computation if max_pad_duration is set
        if self.settings.max_pad_duration is not None:
            max_samples = int(self.settings.max_pad_duration * fs)
            impulse_length = min(impulse_length, max_samples + 1)

        impulse = np.zeros(impulse_length)
        impulse[0] = 1.0

        if self.settings.coef_type == "ba":
            b, a = coefs
            h = scipy.signal.lfilter(b, a, impulse)
        else:
            h = scipy.signal.sosfilt(coefs, impulse)

        # Find where impulse response settles to settle_cutoff of peak
        abs_h = np.abs(h)
        peak = abs_h.max()
        if peak == 0:
            return min_length

        threshold = self.settings.settle_cutoff * peak
        above_threshold = np.where(abs_h > threshold)[0]

        if len(above_threshold) == 0:
            pad_length = min_length
        else:
            pad_length = above_threshold[-1] + 1

        # Ensure at least the scipy heuristic minimum
        pad_length = max(pad_length, min_length)

        # Apply max_pad_duration cap if set
        if self.settings.max_pad_duration is not None:
            max_samples = int(self.settings.max_pad_duration * fs)
            pad_length = min(pad_length, max_samples)

        return pad_length

    def _reset_state(self, message: AxisArray) -> None:
        """Reset filter state when stream changes."""
        self._coefs_cache = None
        self._zi_tiled = None
        self._buffer = None
        # Compute pad_length based on the message's sampling rate
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        fs = 1 / message.axes[axis].gain
        self._pad_length = self._compute_pad_length(fs)
        self.state.needs_redesign = True

    def _compute_zi_tiled(self, data: np.ndarray, ax_idx: int) -> None:
        """Compute and cache the tiled zi for the given data shape.

        Called once per stream (or after filter redesign). The result is
        broadcast-ready for multiplication by the edge sample on each chunk.
        """
        if self.settings.coef_type == "ba":
            b, a = self._coefs_cache
            zi_base = scipy.signal.lfilter_zi(b, a)
        else:  # sos
            zi_base = scipy.signal.sosfilt_zi(self._coefs_cache)

        n_tail = data.ndim - ax_idx - 1

        if self.settings.coef_type == "ba":
            zi_expand = (None,) * ax_idx + (slice(None),) + (None,) * n_tail
            n_tile = data.shape[:ax_idx] + (1,) + data.shape[ax_idx + 1 :]
        else:  # sos
            zi_expand = (slice(None),) + (None,) * ax_idx + (slice(None),) + (None,) * n_tail
            n_tile = (1,) + data.shape[:ax_idx] + (1,) + data.shape[ax_idx + 1 :]

        self._zi_tiled = np.tile(zi_base[zi_expand], n_tile)

    def _initialize_zi(self, data: np.ndarray, ax_idx: int) -> np.ndarray:
        """Initialize filter state (zi) scaled by edge value."""
        if self._zi_tiled is None:
            self._compute_zi_tiled(data, ax_idx)
        first_sample = np.take(data, [0], axis=ax_idx)
        return self._zi_tiled * first_sample

    def _process(self, message: AxisArray) -> AxisArray:
        axis = message.dims[0] if self.settings.axis is None else self.settings.axis
        ax_idx = message.get_axis_idx(axis)
        fs = 1 / message.axes[axis].gain

        # Check if we need to redesign filter
        if self._coefs_cache is None or self.state.needs_redesign:
            self._coefs_cache = self.get_design_function()(fs)
            self._pad_length = self._compute_pad_length(fs)
            self._zi_tiled = None  # Invalidate; recomputed on next use.
            self.state.needs_redesign = False

            # Initialize buffer with duration based on pad_length
            # Add some margin to handle variable chunk sizes
            buffer_duration = (self._pad_length + 1) / fs
            self._buffer = HybridAxisArrayBuffer(duration=buffer_duration, axis=axis)

        # Early exit if filter is effectively disabled
        if self._coefs_cache is None or self.settings.order <= 0 or message.data.size <= 0:
            return message

        # Write new data to buffer
        self._buffer.write(message)
        n_available = self._buffer.available()
        n_output = n_available - self._pad_length

        # If we don't have enough data yet, return empty
        if n_output <= 0:
            new_shape = list(message.data.shape)
            new_shape[ax_idx] = 0
            empty_data = np.empty(new_shape, dtype=message.data.dtype)
            return replace(message, data=empty_data)

        # Peek all available data from buffer
        # Note: HybridAxisArrayBuffer moves the target axis to position 0
        buffered = self._buffer.peek(n_available)
        combined = buffered.data
        buffer_ax_idx = 0  # Buffer always puts time axis at position 0

        # Backward filter on reversed data
        combined_rev = np.flip(combined, axis=buffer_ax_idx)
        backward_zi = self._initialize_zi(combined_rev, buffer_ax_idx)

        if self.settings.coef_type == "ba":
            b, a = self._coefs_cache
            y_bwd_rev, _ = scipy.signal.lfilter(b, a, combined_rev, axis=buffer_ax_idx, zi=backward_zi)
        else:  # sos
            y_bwd_rev, _ = scipy.signal.sosfilt(self._coefs_cache, combined_rev, axis=buffer_ax_idx, zi=backward_zi)

        # Reverse back to get output in correct time order
        y_bwd = np.flip(y_bwd_rev, axis=buffer_ax_idx)

        # Output the settled portion (first n_output samples)
        y = y_bwd[:n_output]

        # Advance buffer read head to discard output samples, keep pad_length
        self._buffer.seek(n_output)

        # Build output with adjusted time axis
        # LinearAxis offset is already correct from the buffer
        out_axis = buffered.axes[axis]

        # Move axis back to original position if needed
        if ax_idx != 0:
            y = np.moveaxis(y, 0, ax_idx)

        return replace(
            message,
            data=y,
            axes={**message.axes, axis: out_axis},
        )


class ButterworthZeroPhaseTransformer(CompositeProcessor[ButterworthZeroPhaseSettings, AxisArray, AxisArray]):
    """
    Streaming zero-phase Butterworth filter as a composite of two stages.

    Stage 1 (forward): Standard causal Butterworth filter with state
    Stage 2 (backward): Acausal Butterworth filter with buffering

    The output is delayed by ``pad_length`` samples.
    """

    @staticmethod
    def _initialize_processors(
        settings: ButterworthZeroPhaseSettings,
    ) -> dict[str, typing.Any]:
        # Both stages use the same filter design settings
        return {
            "forward": ButterworthFilterTransformer(settings),
            "backward": ButterworthBackwardFilterTransformer(settings),
        }

    @classmethod
    def get_message_type(cls, dir: str) -> type[AxisArray]:
        if dir in ("in", "out"):
            return AxisArray
        raise ValueError(f"Invalid direction: {dir}. Must be 'in' or 'out'.")


class ButterworthZeroPhase(
    BaseTransformerUnit[ButterworthZeroPhaseSettings, AxisArray, AxisArray, ButterworthZeroPhaseTransformer]
):
    SETTINGS = ButterworthZeroPhaseSettings
