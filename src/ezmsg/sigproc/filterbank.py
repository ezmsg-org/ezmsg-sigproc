"""Parallel filterbank processing with convolution, FFT, or automatic mode selection."""

import functools
import math
import typing
from copy import deepcopy

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import scipy.fft as sp_fft
import scipy.signal as sps
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
    resolve_configured_stream_dim,
    suppress_axis_deprecation,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from scipy.special import lambertw

from .spectrum import OptionsEnum
from .util.deprecation import warn_axis_deprecated
from .util.message import with_fingerprint
from .util.minphase import MinPhaseMode, is_linear_phase, kernel_delay, min_phase_kernel
from .util.startup import StartupMode, hold_for_startup, prime_with_reflection, startup_n_pad
from .window import WindowTransformer


class FilterbankMode(OptionsEnum):
    """The mode of operation for the filterbank."""

    CONV = "Direct Convolution"
    FFT = "FFT Convolution"
    AUTO = "Automatic"


class FilterbankSettings(ez.Settings):
    kernels: list[npt.NDArray] | tuple[npt.NDArray, ...]

    mode: FilterbankMode = FilterbankMode.CONV
    """
    "conv", "fft", or "auto". If "auto", the mode is determined by the size of the input data.
      fft mode is more efficient for long kernels. However, fft mode uses non-overlapping windows and will
      incur a delay equal to the window length, which is larger than the largest kernel.
      conv mode is less efficient but will return data for every incoming chunk regardless of how small it is
      and thus can provide shorter latency updates.
    """

    min_phase: MinPhaseMode = MinPhaseMode.NONE
    """
    If not NONE, convert the kernels to minimum-phase equivalents with
      :func:`~ezmsg.sigproc.util.minphase.min_phase_kernel`. HILBERT and HOMOMORPHIC return
      half-length kernels with the square root of the original magnitude response;
      HOMOMORPHICFULL keeps the length and magnitude response. Complex kernels support
      the homomorphic modes only.
    """

    compensate_delay: bool = False
    """
    If True, put every kernel on one time base and stamp the output with it: each kernel's
      delay is measured with :func:`~ezmsg.sigproc.util.minphase.kernel_delay`, leading zeros
      are prepended so all kernels share the largest delay (to within half a sample), and that
      delay is subtracted from the stream axis offset. With ``min_phase=HOMOMORPHICFULL`` only the
      kernels that would otherwise set the common delay are converted; the rest stay linear
      phase. HILBERT and HOMOMORPHIC still convert every kernel, since they change the magnitude
      response.
    """

    startup: StartupMode = StartupMode.CONSTANT
    """
    What the filterbank assumes before the first sample.

    * CONSTANT (default): the first sample's value held for the length of the longest kernel, so
      a DC offset does not ring through as a step. Every kernel starts at its steady state for
      that value.
    * ZERO: a zero pre-history, so the start of a stream matches an offline convolution.
    * FIRST_CHUNK, BUFFERED: an odd reflection of the first samples; see
      :attr:`ezmsg.sigproc.filter.FilterBaseSettings.startup`, including why that suits low-pass
      kernels and not band-pass ones. CONV mode only: in FFT mode (including AUTO resolving to
      FFT) they raise ``ValueError``.
    """

    startup_max_dur: float | None = 1.0
    """Upper limit, in seconds, on the start-up pad; None for no limit."""

    axis: str | None = None
    """.. deprecated:: 3.8
        Scheduled for removal in 4.0. The dimension messages accumulate along
        now comes from :attr:`~ezmsg.util.messages.axisarray.AxisArray.stream_dim`;
        see :mod:`ezmsg.sigproc.util.deprecation`."""

    def __post_init__(self) -> None:
        warn_axis_deprecated(self)

    new_axis: str = "kernel"
    """The name of the new axis corresponding to the kernel index."""

    bands: list[tuple[float, float]] | None = None
    """
    Optional ``(lo, hi)`` pass-band edges in Hz, one per kernel. If given, the output carries a
      :obj:`~ezmsg.util.messages.axisarray.AxisArray.CoordinateAxis` on ``new_axis`` whose data is
      a structured array with fields ``lo`` and ``hi``. Purely descriptive; the kernels are not checked.
    """


@processor_state
class FilterbankState:
    tail: npt.NDArray | None = None
    template: AxisArray | None = None
    dest_arr: npt.NDArray | None = None
    prep_kerns: npt.NDArray | list[npt.NDArray] | None = None
    windower: WindowTransformer | None = None
    fft: typing.Callable | None = None
    ifft: typing.Callable | None = None
    nfft: int | None = None
    infft: int | None = None
    overlap: int | None = None
    mode: FilterbankMode | None = None
    delay: float = 0.0
    """Common delay (samples) subtracted from the output offset when ``compensate_delay``."""
    startup_pending: bool = False
    startup_held: list | None = None
    startup_n_pad: int = 0
    const_tail: npt.NDArray | None = None


class FilterbankTransformer(BaseStatefulTransformer[FilterbankSettings, AxisArray, AxisArray, FilterbankState]):
    def _hash_message(self, message: AxisArray) -> int:
        """Extend the default with the input dtype.

        The kernels are cast to the message's dtype at reset, so a real input
        followed by a complex one needs a rebuild even though nothing about the
        stream's shape or channels changed. The default cannot see dtype.
        """
        return self._message_hash(message, extra=(message.data.dtype.kind,))

    def _reset_state(self, message: AxisArray) -> None:
        axis = resolve_configured_stream_dim(self, message, self.settings.axis)
        gain = message.axes[axis].gain if axis in message.axes else 1.0
        targ_ax_ix = message.get_axis_idx(axis)
        in_shape = message.data.shape[:targ_ax_ix] + message.data.shape[targ_ax_ix + 1 :]

        kernels = self._prepare_kernels(gain)

        # Determine if this will be operating with complex data.
        b_complex = message.data.dtype.kind == "c" or any([_.dtype.kind == "c" for _ in kernels])

        # Calculate window_dur, window_shift, nfft
        max_kernel_len = max([_.size for _ in kernels])
        # From sps._calc_oa_lens, where s2=max_kernel_len,:
        # fallback_nfft = n_input + max_kernel_len - 1, but n_input is unbound.
        self._state.overlap = max_kernel_len - 1

        # Prepare previous iteration's overlap tail to add to input -- all zeros.
        tail_shape = in_shape + (len(kernels), self._state.overlap)
        self._state.tail = np.zeros(tail_shape, dtype="complex" if b_complex else "float")

        # Prepare output template -- kernels axis immediately before the target axis.
        dummy_shape = in_shape + (len(kernels), 0)
        self._state.template = AxisArray(
            data=np.zeros(dummy_shape, dtype="complex" if b_complex else "float"),
            dims=message.dims[:targ_ax_ix] + message.dims[targ_ax_ix + 1 :] + [self.settings.new_axis, axis],
            axes={k: deepcopy(v) for k, v in message.axes.items()},
            key=message.key,
        )
        if self.settings.bands is not None:
            if len(self.settings.bands) != len(kernels):
                raise ValueError(f"bands has {len(self.settings.bands)} entries for {len(kernels)} kernels.")
            self._state.template.axes[self.settings.new_axis] = with_fingerprint(
                AxisArray.CoordinateAxis(
                    data=np.array([tuple(b) for b in self.settings.bands], dtype=[("lo", float), ("hi", float)]),
                    dims=[self.settings.new_axis],
                    unit="Hz",
                )
            )

        # Determine optimal mode. Assumes 100 msec chunks.
        self._state.mode = self.settings.mode
        if self._state.mode == FilterbankMode.AUTO:
            # concatenate kernels into 1 mega kernel then check what's faster.
            # Will typically return fft when combined kernel length is > 1500.
            concat_kernel = np.concatenate(kernels)
            n_dummy = max(2 * len(concat_kernel), int(0.1 / gain))
            dummy_arr = np.zeros(n_dummy)
            self._state.mode = (
                FilterbankMode.CONV
                if sps.choose_conv_method(dummy_arr, concat_kernel, mode="full") == "direct"
                else FilterbankMode.FFT
            )
        reflect = self.settings.startup in (StartupMode.FIRST_CHUNK, StartupMode.BUFFERED)
        if reflect and self._state.mode == FilterbankMode.FFT:
            raise ValueError(
                f"startup={self.settings.startup} is not supported in FFT mode (resolved from "
                f"mode={self.settings.mode}). Use mode=CONV, or startup CONSTANT or ZERO."
            )
        self._state.startup_held = []
        self._state.startup_pending = self.settings.startup != StartupMode.ZERO
        # Overlap-add carry left by a unit constant input: tail[j] = sum(k[j + 1 :]).
        self._state.const_tail = np.array(
            [np.pad(np.cumsum(k[::-1])[::-1][1:], (0, max_kernel_len - k.size)) for k in kernels]
        )
        if reflect:
            # The tail is fully determined by the last `overlap` inputs.
            self._state.startup_n_pad = startup_n_pad(self._state.overlap, message, axis, self.settings.startup_max_dur)
        if self._state.mode == FilterbankMode.CONV:
            # Preallocate memory for convolution result and overlap-add
            dest_shape = in_shape + (
                len(kernels),
                self._state.overlap + message.data.shape[targ_ax_ix],
            )
            self._state.dest_arr = np.zeros(dest_shape, dtype="complex" if b_complex else "float")
            self._state.prep_kerns = kernels
        else:  # FFT mode
            # Calculate optimal nfft and windowing size.
            opt_size = -self._state.overlap * lambertw(-1 / (2 * math.e * self._state.overlap), k=-1).real
            self._state.nfft = sp_fft.next_fast_len(math.ceil(opt_size))
            win_len = self._state.nfft - self._state.overlap
            # infft same as nfft. Keeping as separate variable because I might need it again.
            self._state.infft = win_len + self._state.overlap

            # Create windowing node.
            # Note: We could do windowing manually to avoid the overhead of the message structure,
            #  but windowing is difficult to do correctly, so we lean on the heavily-tested `windowing` generator.
            win_dur = win_len * gain
            with suppress_axis_deprecation():  # our own resolved axis, not the user's setting
                self._state.windower = WindowTransformer(
                    axis=axis,
                    newaxis="win",
                    window_dur=win_dur,
                    window_shift=win_dur,
                    zero_pad_until="none",
                )

            # Windowing output has an extra "win" dimension, so we need our tail to match.
            self._state.tail = np.expand_dims(self._state.tail, -2)

            # Prepare fft functions
            # Note: We could instead use `spectrum` but this adds overhead in creating the message structure
            #  for a rather simple calculation. We may revisit if `spectrum` gets additional features, such as
            #  more fft backends.
            if b_complex:
                self._state.fft = functools.partial(sp_fft.fft, n=self._state.nfft, norm="backward")
                self._state.ifft = functools.partial(sp_fft.ifft, n=self._state.infft, norm="backward")
            else:
                self._state.fft = functools.partial(sp_fft.rfft, n=self._state.nfft, norm="backward")
                self._state.ifft = functools.partial(sp_fft.irfft, n=self._state.infft, norm="backward")

            # Calculate fft of kernels
            self._state.prep_kerns = np.array([self._state.fft(_) for _ in kernels])
            self._state.prep_kerns = np.expand_dims(self._state.prep_kerns, -2)
            # TODO: If fft_kernels have significant stretches of zeros, convert to sparse array.

    def _prepare_kernels(self, gain: float) -> list[npt.NDArray]:
        """Apply ``min_phase`` and, with ``compensate_delay``, align the kernels on one delay. Logs the delays."""
        mode = self.settings.min_phase
        kernels = [np.asarray(k) for k in self.settings.kernels]
        converted = [False] * len(kernels)
        if self.settings.compensate_delay and mode == MinPhaseMode.HOMOMORPHICFULL:
            # Convert only the bottleneck: while the slowest kernel is still linear phase, convert it.
            delays = [kernel_delay(k) for k in kernels]
            while not converted[k_ix := int(np.argmax(delays))]:
                kernels[k_ix] = min_phase_kernel(kernels[k_ix], mode)
                converted[k_ix] = True
                delays[k_ix] = kernel_delay(kernels[k_ix])
        elif mode != MinPhaseMode.NONE:
            kernels = [min_phase_kernel(k, mode) for k in kernels]
            converted = [True] * len(kernels)
            delays = [kernel_delay(k) for k in kernels]
        else:
            delays = [kernel_delay(k) for k in kernels]

        approx = "" if all(is_linear_phase(k) for k in kernels) else "approximate "
        per_kernel = ", ".join(
            f"{k_ix}: {d:.4g} ({d * gain * 1e3:.4g} ms{', min-phase' if c else ''})"
            for k_ix, (d, c) in enumerate(zip(delays, converted))
        )
        msg = f"Filterbank {approx}kernel delays in samples: {per_kernel}."
        self._state.delay = 0.0
        if self.settings.compensate_delay:
            d_max = max(delays)
            pads = [round(d_max - d) for d in delays]
            kernels = [np.concatenate([np.zeros(p, dtype=k.dtype), k]) for k, p in zip(kernels, pads)]
            residual = max(abs(d_max - d - p) for d, p in zip(delays, pads))
            self._state.delay = d_max
            msg += (
                f" Equalised to a common {approx}delay of {d_max:.4g} samples ({d_max * gain * 1e3:.4g} ms),"
                f" worst residual misalignment {residual:.3g} samples; compensated on the time axis."
            )
        else:
            msg += " Not compensated on the time axis."
        ez.logger.info(msg)
        return kernels

    def _process(self, message: AxisArray) -> AxisArray:
        axis = resolve_configured_stream_dim(self, message, self.settings.axis)
        if self._state.startup_pending and message.data.size > 0:
            if self.settings.startup == StartupMode.BUFFERED:
                released = hold_for_startup(self._state.startup_held, message, axis, self._state.startup_n_pad + 1)
                if released is None:
                    return self._output(self._state.template.data, message, axis)
                message = released
            self._state.startup_pending = False
            if self.settings.startup == StartupMode.CONSTANT:
                self._seed_constant(message)
            else:
                prime_with_reflection(self._prime, message, axis, self._state.startup_n_pad)
        targ_ax_ix = message.get_axis_idx(axis)

        # Make sure target axis is in -1th position.
        if targ_ax_ix != (message.data.ndim - 1):
            in_dat = np.moveaxis(message.data, targ_ax_ix, -1)
            if self._state.mode == FilterbankMode.FFT:
                # Fix message.dims because we will pass it to windower
                move_dims = message.dims[:targ_ax_ix] + message.dims[targ_ax_ix + 1 :] + [axis]
                message = replace(message, data=in_dat, dims=move_dims)
        else:
            in_dat = message.data

        if self._state.mode == FilterbankMode.CONV:
            n_dest = in_dat.shape[-1] + self._state.overlap
            if self._state.dest_arr.shape[-1] < n_dest:
                pad = np.zeros(self._state.dest_arr.shape[:-1] + (n_dest - self._state.dest_arr.shape[-1],))
                self._state.dest_arr = np.concatenate([self._state.dest_arr, pad], axis=-1)
            self._state.dest_arr.fill(0)

            # Note: I tried several alternatives to this loop; all were slower than this.
            #  numba.jit; stride_tricks + np.einsum; threading. Latter might be better with Python 3.13.
            for k_ix, k in enumerate(self._state.prep_kerns):
                n_out = in_dat.shape[-1] + k.shape[-1] - 1
                self._state.dest_arr[..., k_ix, :n_out] = np.apply_along_axis(np.convolve, -1, in_dat, k, mode="full")
            self._state.dest_arr[..., : self._state.overlap] += self._state.tail
            new_tail = self._state.dest_arr[..., in_dat.shape[-1] : n_dest]
            if new_tail.size > 0:
                # COPY overlap for next iteration
                self._state.tail = new_tail.copy()
            res = self._state.dest_arr[..., : in_dat.shape[-1]].copy()
        else:  # FFT mode
            # Slice into non-overlapping windows
            win_msg = self._state.windower.send(message)
            # Calculate spectrum of each window
            spec_dat = self._state.fft(win_msg.data, axis=-1)
            # Insert axis for filters
            spec_dat = np.expand_dims(spec_dat, -3)

            # Do the FFT convolution
            # TODO: handle fft_kernels being sparse. Maybe need np.dot.
            conv_spec = spec_dat * self._state.prep_kerns
            overlapped = self._state.ifft(conv_spec, axis=-1)

            # Do the overlap-add on the `axis` axis
            # Previous iteration's tail:
            overlapped[..., :1, : self._state.overlap] += self._state.tail
            # window-to-window:
            overlapped[..., 1:, : self._state.overlap] += overlapped[..., :-1, -self._state.overlap :]
            # Save tail:
            new_tail = overlapped[..., -1:, -self._state.overlap :]
            if new_tail.size > 0:
                # All of the above code works if input is size-zero, but we don't want to save a zero-size tail.
                self._state.tail = new_tail
            # Concat over win axis, without overlap.
            res = overlapped[..., : -self._state.overlap].reshape(overlapped.shape[:-2] + (-1,))

        # FFT output starts at the first buffered window, not at this message.
        offset = win_msg.axes["win"].offset if self._state.mode == FilterbankMode.FFT else None
        return self._output(res, message, axis, offset)

    def _seed_constant(self, message: AxisArray) -> None:
        """Set the overlap-add tail as if ``message``'s first sample had been held before it."""
        axis = resolve_configured_stream_dim(self, message, self.settings.axis)
        x0 = np.moveaxis(message.data, message.get_axis_idx(axis), -1)[..., :1, None]
        tail = x0 * self._state.const_tail  # (*in_shape, kernel, overlap)
        self._state.tail = tail[..., None, :] if self._state.mode == FilterbankMode.FFT else tail

    def _prime(self, pad: AxisArray) -> None:
        # A pad capped short of the kernels still starts from a constant, not from zeros.
        self._seed_constant(pad)
        self._process(pad)

    def _output(self, res: npt.NDArray, message: AxisArray, axis: str, offset: float | None = None) -> AxisArray:
        out_ax = message.axes[axis]
        if offset is not None:
            out_ax = replace(out_ax, offset=offset)
        if self.settings.compensate_delay and self._state.delay:
            out_ax = replace(out_ax, offset=out_ax.offset - self._state.delay * out_ax.gain)
        return replace(
            self._state.template,
            data=res,
            axes={**self._state.template.axes, axis: out_ax},
        )


class Filterbank(BaseTransformerUnit[FilterbankSettings, AxisArray, AxisArray, FilterbankTransformer]):
    SETTINGS = FilterbankSettings


def filterbank(
    kernels: list[npt.NDArray] | tuple[npt.NDArray, ...],
    mode: FilterbankMode = FilterbankMode.CONV,
    min_phase: MinPhaseMode = MinPhaseMode.NONE,
    axis: str | None = None,
    new_axis: str = "kernel",
) -> FilterbankTransformer:
    """
    Perform multiple (direct or fft) convolutions on a signal using a bank of kernels.
     This is intended to be used during online processing, therefore both direct and fft convolutions
     use the overlap-add method.

    Returns: :obj:`FilterbankTransformer`.
    """
    return FilterbankTransformer(
        settings=FilterbankSettings(
            kernels=kernels,
            mode=mode,
            min_phase=min_phase,
            axis=axis,
            new_axis=new_axis,
        )
    )
