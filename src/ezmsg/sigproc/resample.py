"""Resample signals to a target rate using interpolation."""

import asyncio
import math
import time
import warnings

import ezmsg.core as ez
import numpy as np
import scipy.interpolate
from array_api_compat import get_namespace
from ezmsg.baseproc import (
    BaseConsumerUnit,
    BaseStatefulProcessor,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis, slice_along_axis
from ezmsg.util.messages.util import replace

from .util.axisarray_buffer import HybridAxisArrayBuffer, HybridAxisBuffer
from .util.buffer import UpdateStrategy
from .util.deprecation import warn_axis_deprecated
from .util.message import has_samples_along, resolve_configured_chunk_dim


def _as_limit(value: float | None) -> float | None:
    """Normalize an optional "no limit" setting to ``None``.

    ``None`` is the canonical way to say "no limit" because it survives JSON
    serialization, which ``inf`` does not. Non-finite values are accepted and
    folded onto ``None`` so pipelines that predate this convention keep working.
    """
    if value is None or not math.isfinite(value):
        return None
    return value


class ResampleSettings(ez.Settings):
    axis: str | None = None
    """.. deprecated:: 3.8
        Scheduled for removal in 4.0. The dimension messages accumulate along
        now comes from :attr:`~ezmsg.util.messages.axisarray.AxisArray.chunk_dim`;
        see :mod:`ezmsg.sigproc.util.deprecation`."""

    def __post_init__(self) -> None:
        warn_axis_deprecated(self)

    resample_rate: float | None = None
    """target resample rate in Hz. If None, the resample rate will be determined by the reference signal."""

    max_chunk_delay: float | None = None
    """
    Maximum delay between outputs in seconds. If the delay exceeds this value, the
    transformer will extrapolate. ``None`` (the default) means no limit: the transformer
    never extrapolates on wall-clock alone. A non-finite value is treated as ``None``.
    """

    fill_value: str = "extrapolate"
    """
    Value to use for out-of-bounds samples.
    If 'extrapolate', the transformer will extrapolate.
    If 'last', the transformer will use the last sample.
    See scipy.interpolate.interp1d for more options.
    """

    buffer_duration: float = 2.0

    buffer_update_strategy: UpdateStrategy = "immediate"
    """
    The buffer update strategy. See :obj:`ezmsg.sigproc.util.buffer.UpdateStrategy`.
    If you expect to push data much more frequently than it is resampled, then "on_demand"
    might be more efficient. For most other scenarios, "immediate" is best.
    """

    output_reference: bool = False
    """
    If True, also buffer the *data* carried by INPUT_REFERENCE messages and, for each
    output chunk, gather the reference samples that sit on the exact same grid the signal
    was resampled onto. The gathered reference signal is exposed via
    :attr:`ResampleState.reference_output` (and published on ``OUTPUT_REFERENCE`` by
    :class:`ResampleUnit`). This is what lets a downstream consumer concatenate the
    reference stream with the resampled stream without a second time-alignment, because
    the two share an identical axis *by construction*.

    Only meaningful when ``resample_rate is None`` (reference-driven mode); in prescribed-
    rate mode the reference grid is synthetic and carries no data.
    """

    reference_reset_after_chunks: int | None = 3
    """
    Robustness against a non-monotonic reference clock.

    The resampler only emits at reference values greater than the last one it returned
    (a high-water mark). A *sustained* backward jump in the reference clock (e.g. a
    misbehaving source whose chunk offsets reset to an earlier time) would otherwise
    leave every incoming reference value behind the high-water mark forever, so the
    transformer would stop producing output.

    If this many consecutive reference messages arrive entirely at or below the
    high-water mark, the jump is treated as a clock reset: the high-water mark is
    re-anchored to the new (lower) clock and a ``RuntimeWarning`` is emitted. Output then
    resumes on the new clock. Small, self-correcting jitter (a few out-of-order samples)
    does not trigger this and is simply skipped, keeping the output monotonic.

    Set to ``None`` to disable reset recovery (the transformer may then stall
    indefinitely on a backward clock jump). Output across a genuine reset is necessarily
    discontinuous; sanitising the reference timestamps upstream remains the robust fix.
    A non-finite value is treated as ``None``.
    """


@processor_state
class ResampleState:
    axis: str = ""
    """The resolved chunk dimension, fixed at reset so every later use agrees."""

    src_buffer: HybridAxisArrayBuffer | None = None
    """
    Buffer for the incoming signal data. This is the source for training the interpolation function.
    Its contents are rarely empty because we usually hold back some data to allow for accurate
    interpolation and optionally extrapolation.
    """

    ref_axis_buffer: HybridAxisBuffer | None = None
    """
    The buffer for the reference axis (usually a time axis). The interpolation function
    will be evaluated at the reference axis values.
    When resample_rate is None, this buffer will be filled with the axis from incoming
    _reference_ messages.
    When resample_rate is not None (i.e., prescribed float resample_rate), this buffer
    is filled with a synthetic axis that is generated from the incoming signal messages.
    """

    last_ref_ax_val: float | None = None
    """
    The last value of the reference axis that was returned. This helps us to know
    what the _next_ returned value should be, and to avoid returning the same value.
    TODO: We can eliminate this variable if we maintain "by convention" that the
    reference axis always has 1 value at its start that we exclude from the resampling.
    """

    last_write_time: float = -np.inf
    """
    Monotonic time of the last write to the signal buffer.
    This is used to determine if we need to extrapolate the reference axis
    if we have not received an update within max_chunk_delay.
    """

    ref_data_buffer: HybridAxisArrayBuffer | None = None
    """
    Buffer for the *data* carried by reference messages, used only when
    ``output_reference`` is set. Held in lockstep with ``ref_axis_buffer`` (same writes,
    same seeks) so that the reference data at index ``i`` corresponds to the reference
    axis value at index ``i``. Lets :meth:`__next__` gather the reference signal on the
    exact output grid.
    """

    reference_output: AxisArray | None = None
    """
    The reference signal gathered onto the most recent output grid (only populated when
    ``output_reference`` is set and the last :meth:`__next__` produced data). Shares its
    axis with the resampled output, so the two can be concatenated without re-aligning.
    """

    stale_ref_pushes: int = 0
    """
    Number of consecutive reference messages that arrived entirely at or below the
    high-water mark. Used to detect a sustained backward clock jump (reset). See
    :attr:`ResampleSettings.reference_reset_after_chunks`.
    """


class ResampleProcessor(BaseStatefulProcessor[ResampleSettings, AxisArray, AxisArray, ResampleState]):
    # Both fields are read per-chunk inside `__next__` / `_process` only;
    # `resample_rate` / `buffer_duration` / `axis` all size cached buffers.
    NONRESET_SETTINGS_FIELDS = frozenset({"max_chunk_delay", "fill_value", "reference_reset_after_chunks"})

    def _seed_axis(self, message: AxisArray) -> str:
        """Resolve the chunk dimension, seeding it if the reference stream got here first.

        :meth:`push_reference` is an independent entry point and can be called
        before any signal message has arrived, while :meth:`__next__` has no
        message at all. Both read the resolved value off state, so it is fixed
        once and shared rather than re-derived from whichever message happens to
        be in hand.
        """
        if not self.state.axis:
            self.state.axis = resolve_configured_chunk_dim(self, message, self.settings.axis, legacy_default="time")
        return self.state.axis

    def _reset_state(self, message: AxisArray) -> None:
        """
        Reset the internal state based on the incoming message.
        """
        # The signal stream is authoritative, so this overwrites any value the
        # reference stream seeded above.
        self.state.axis = resolve_configured_chunk_dim(self, message, self.settings.axis, legacy_default="time")
        self.state.src_buffer = HybridAxisArrayBuffer(
            duration=self.settings.buffer_duration,
            axis=self.state.axis,
            update_strategy=self.settings.buffer_update_strategy,
            overflow_strategy="grow",
        )
        if self.settings.resample_rate is not None:
            # If we are resampling at a prescribed rate, then we synthesize a reference axis
            self.state.ref_axis_buffer = HybridAxisBuffer(
                duration=self.settings.buffer_duration,
            )
            in_ax = message.axes[self.state.axis]
            out_gain = 1 / self.settings.resample_rate
            t0 = in_ax.data[0] if hasattr(in_ax, "data") else in_ax.value(0)
            self.state.last_ref_ax_val = t0 - out_gain
        self.state.last_write_time = -np.inf
        self.state.reference_output = None
        self.state.stale_ref_pushes = 0

    @staticmethod
    def _axis_first_step(ax) -> tuple[float, float]:
        """Return ``(first_value, step)`` for either a LinearAxis or a CoordinateAxis.

        A CoordinateAxis has no ``gain``; estimate the step from consecutive samples
        (falling back to 0.0 for a single sample). The step is only used to seed the
        high-water mark just below the first reference value, so an estimate is fine.
        """
        if hasattr(ax, "data"):
            first = float(ax.data[0])
            step = float(ax.data[1] - ax.data[0]) if len(ax.data) > 1 else 0.0
        else:
            first = ax.value(0)
            step = ax.gain
        return first, step

    def push_reference(self, message: AxisArray) -> None:
        axis = self._seed_axis(message)
        ax = message.axes[axis]
        ax_idx = message.get_axis_idx(axis)
        n = message.data.shape[ax_idx]
        if self.state.ref_axis_buffer is None:
            self.state.ref_axis_buffer = HybridAxisBuffer(
                duration=self.settings.buffer_duration,
                update_strategy=self.settings.buffer_update_strategy,
                overflow_strategy="grow",
            )
            first, step = self._axis_first_step(ax)
            self.state.last_ref_ax_val = first - step
            if self.settings.output_reference:
                # Sister buffer for the reference *data*, kept in lockstep with the axis
                # buffer above so a shared index addresses the same sample in both.
                self.state.ref_data_buffer = HybridAxisArrayBuffer(
                    duration=self.settings.buffer_duration,
                    axis=axis,
                    update_strategy=self.settings.buffer_update_strategy,
                    overflow_strategy="grow",
                )

        # --- Detect a sustained backward clock jump (reset) ---
        # A whole message arriving at/below the high-water mark is abnormal; a run of them
        # means the clock jumped back and is not catching up. Re-anchor so we don't stall
        # forever. Self-correcting jitter (a single stale chunk) does not trigger this.
        if self.state.last_ref_ax_val is not None and n > 0:
            incoming_max = float(np.max(ax.data)) if hasattr(ax, "data") else ax.value(n - 1)
            if incoming_max <= self.state.last_ref_ax_val:
                self.state.stale_ref_pushes += 1
            else:
                self.state.stale_ref_pushes = 0
            reset_after = _as_limit(self.settings.reference_reset_after_chunks)
            if reset_after is not None and self.state.stale_ref_pushes >= reset_after:
                first, step = self._axis_first_step(ax)
                warnings.warn(
                    "ResampleProcessor: reference clock jumped backward and stayed behind "
                    f"the last emitted value for {self.state.stale_ref_pushes} consecutive "
                    "messages; treating it as a clock reset and re-anchoring to the new "
                    "clock. Output across the reset is discontinuous; make the reference "
                    "timestamps monotonic upstream to avoid this.",
                    RuntimeWarning,
                )
                self.state.last_ref_ax_val = first - step
                self.state.stale_ref_pushes = 0

        self.state.ref_axis_buffer.write(ax, n_samples=n)
        if self.state.ref_data_buffer is not None:
            self.state.ref_data_buffer.write(message)

    def _process(self, message: AxisArray) -> None:
        """
        Add a new data message to the buffer and update the reference axis if needed.
        """
        # Note: The src_buffer will copy and permute message if ax_idx != 0
        self.state.src_buffer.write(message)

        # If we are resampling at a prescribed rate (i.e., not by reference msgs),
        #  then we use this opportunity to extend our synthetic reference axis.
        ax_idx = message.get_axis_idx(self.state.axis)
        if self.settings.resample_rate is not None and message.data.shape[ax_idx] > 0:
            in_ax = message.axes[self.state.axis]
            in_t_end = in_ax.data[-1] if hasattr(in_ax, "data") else in_ax.value(message.data.shape[ax_idx] - 1)
            out_gain = 1 / self.settings.resample_rate
            prev_t_end = self.state.last_ref_ax_val
            n_synth = math.ceil((in_t_end - prev_t_end) * self.settings.resample_rate)
            synth_ref_axis = LinearAxis(unit="s", gain=out_gain, offset=prev_t_end + out_gain)
            self.state.ref_axis_buffer.write(synth_ref_axis, n_samples=n_synth)

        self.state.last_write_time = time.monotonic()

    def __next__(self) -> AxisArray:
        # Cleared each call; only repopulated on the success path below.
        self.state.reference_output = None
        if self.state.src_buffer is None or self.state.ref_axis_buffer is None:
            # If we have not received any data, or we require reference data
            #  that we do not yet have, then return an empty template.
            return AxisArray(data=np.array([]), dims=[""], axes={}, key="null")

        src = self.state.src_buffer
        ref = self.state.ref_axis_buffer

        # If we have no reference or the source is insufficient for interpolation
        #  then return the empty template
        if ref.is_empty() or src.available() < 3:
            src_axarr = src.peek(0)
            return replace(
                src_axarr,
                axes={
                    **src_axarr.axes,
                    self.state.axis: ref.peek(0),
                },
            )

        # Build the reference xvec.
        #  Note: The reference axis buffer may grow upon `.peek()`
        #   as it flushes data from its deque to its buffer.
        ref_ax = ref.peek()
        if hasattr(ref_ax, "data"):
            ref_xvec = ref_ax.data
        else:
            ref_xvec = ref_ax.value(np.arange(ref.available()))

        # If we do not rely on an external reference, and we have not received new data in a while,
        #  then extrapolate our reference vector out beyond the delay limit.
        max_chunk_delay = _as_limit(self.settings.max_chunk_delay)
        b_project = (
            self.settings.resample_rate is not None
            and max_chunk_delay is not None
            and time.monotonic() > (self.state.last_write_time + max_chunk_delay)
        )
        if b_project:
            n_append = math.ceil(max_chunk_delay / ref_ax.gain)
            xvec_append = ref_xvec[-1] + np.arange(1, n_append + 1) * ref_ax.gain
            ref_xvec = np.hstack((ref_xvec, xvec_append))

        # Get source to train interpolation. The buffer preserves the input layout,
        # so the resample axis is wherever the incoming messages put it.
        src_axarr = src.peek()
        src_ax_idx = src_axarr.get_axis_idx(self.state.axis)
        src_axis = src_axarr.axes[self.state.axis]
        x = src_axis.data if hasattr(src_axis, "data") else src_axis.value(np.arange(src_axarr.data.shape[src_ax_idx]))

        # Only resample at reference values that have not been interpolated over previously.
        b_ref = ref_xvec > self.state.last_ref_ax_val
        if not b_project:
            # Not extrapolating -- Do not resample beyond the end of the source buffer.
            b_ref = np.logical_and(b_ref, ref_xvec <= x[-1])
        ref_idx = np.where(b_ref)[0]

        if len(ref_idx) == 0:
            # Nothing to interpolate over; return empty data
            null_ref = replace(ref_ax, data=ref_ax.data[:0]) if hasattr(ref_ax, "data") else ref_ax
            return replace(
                src_axarr,
                data=slice_along_axis(src_axarr.data, slice(0, 0), src_ax_idx),
                axes={**src_axarr.axes, self.state.axis: null_ref},
            )

        xnew = ref_xvec[ref_idx]

        # Identify source data indices around ref tvec with some padding for better interpolation.
        src_start_ix = max(0, np.where(x > xnew[0])[0][0] - 2 if np.any(x > xnew[0]) else 0)

        x = x[src_start_ix:]
        y = slice_along_axis(src_axarr.data, slice(src_start_ix, None), src_ax_idx)

        if isinstance(self.settings.fill_value, str) and self.settings.fill_value == "last":
            # Edge samples with the resample axis dropped, as interp1d expects.
            fill_value = (
                slice_along_axis(y, 0, src_ax_idx),
                slice_along_axis(y, -1, src_ax_idx),
            )
        else:
            fill_value = self.settings.fill_value
        f = scipy.interpolate.interp1d(
            x,
            y,
            kind="linear",
            axis=src_ax_idx,
            copy=False,
            bounds_error=False,
            fill_value=fill_value,
            assume_sorted=True,
        )

        # Calculate output
        resampled_data = f(xnew)

        # scipy.interpolate.interp1d always returns numpy. Coerce back to the
        # source's namespace so downstream merges with same-backend streams
        # don't see a backend mismatch.
        src_xp = get_namespace(src_axarr.data)
        if get_namespace(resampled_data) is not src_xp:
            resampled_data = src_xp.asarray(resampled_data)

        # Create output message
        if hasattr(ref_ax, "data"):
            out_ax = replace(ref_ax, data=xnew)
        else:
            out_ax = replace(ref_ax, offset=xnew[0])
        result = replace(
            src_axarr,
            data=resampled_data,
            axes={
                **src_axarr.axes,
                self.state.axis: out_ax,
            },
        )

        # Gather the reference signal on this same output grid, if requested. We index by
        # the very same ref_idx used to build the output, so the gathered reference shares
        # an identical axis with `result` and the two need no further alignment.
        if self.state.ref_data_buffer is not None:
            ref_axarr = self.state.ref_data_buffer.peek()
            if ref_axarr is not None:
                ref_ax_idx = ref_axarr.get_axis_idx(self.state.axis)
                if ref_axarr.data.shape[ref_ax_idx] == ref.available():
                    ref_xp = get_namespace(ref_axarr.data)
                    self.state.reference_output = replace(
                        ref_axarr,
                        data=ref_xp.take(ref_axarr.data, ref_idx, axis=ref_ax_idx),
                        axes={**ref_axarr.axes, self.state.axis: out_ax},
                    )

        # Update the state. For state buffers, seek beyond samples that are no longer needed.
        # src: keep at least 1 sample before the final resampled value
        seek_ix = np.where(x >= xnew[-1])[0]
        if len(seek_ix) > 0:
            self.state.src_buffer.seek(max(0, src_start_ix + seek_ix[0] - 1))
        # ref: remove samples that have been sent to output. Keep the data buffer in
        # lockstep with the axis buffer so a shared index keeps addressing the same sample.
        self.state.ref_axis_buffer.seek(ref_idx[-1] + 1)
        if self.state.ref_data_buffer is not None:
            self.state.ref_data_buffer.seek(ref_idx[-1] + 1)
        self.state.last_ref_ax_val = xnew[-1]

        return result

    def send(self, message: AxisArray) -> AxisArray:
        self(message)
        return next(self)


class ResampleUnit(BaseConsumerUnit[ResampleSettings, AxisArray, ResampleProcessor]):
    SETTINGS = ResampleSettings
    INPUT_REFERENCE = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    OUTPUT_REFERENCE = ez.OutputStream(AxisArray)
    """The reference signal gathered onto the same grid as ``OUTPUT_SIGNAL``.

    Only published when ``output_reference`` is set. Because it shares an identical axis
    with ``OUTPUT_SIGNAL`` by construction, a downstream :class:`~ezmsg.sigproc.concat.Concat`
    can combine the two without a second time-alignment (see
    :class:`~ezmsg.sigproc.resampleconcat.ResampleConcat`, which fuses both steps).
    """

    async def initialize(self) -> None:
        await super().initialize()
        self._wake = asyncio.Event()

    @ez.subscriber(BaseConsumerUnit.INPUT_SIGNAL)
    async def on_signal(self, message: AxisArray) -> None:
        await self.processor.__acall__(message)
        self._wake.set()

    @ez.subscriber(INPUT_REFERENCE)
    async def on_reference(self, message: AxisArray):
        self.processor.push_reference(message)
        self._wake.set()

    @ez.publisher(OUTPUT_SIGNAL)
    @ez.publisher(OUTPUT_REFERENCE)
    async def gen_resampled(self):
        while True:
            # Wake on a push from either input. In prescribed-rate mode with a
            # finite max_chunk_delay, also wake when the delay budget elapses so
            # the wall-clock extrapolation in `__next__` can fire without input.
            # Reference-driven mode never becomes ready by wall-clock, so there
            # a pure event wait suffices.
            timeout = _as_limit(self.SETTINGS.max_chunk_delay) if self.SETTINGS.resample_rate is not None else None
            if timeout is not None:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass
            else:
                await self._wake.wait()
            # Clear before draining: a set() landing mid-drain re-arms the event
            # so the next wait() returns immediately instead of losing the push.
            self._wake.clear()
            while True:
                result: AxisArray = next(self.processor)
                # A real chunk has a nonzero resample axis; an empty axis or the
                # pre-init null template (which lacks the axis entirely) means
                # "nothing ready". A chunk that is empty only along other axes
                # (e.g. zero channels) is still real output and must be published.
                if not has_samples_along(result, self.processor.state.axis):
                    break
                yield self.OUTPUT_SIGNAL, result
                ref_out = self.processor.state.reference_output
                if ref_out is not None:
                    yield self.OUTPUT_REFERENCE, ref_out
