"""
Multi-band envelope power from an analytic FIR filterbank.

Pipeline: analytic filterbank (band-pass and Hilbert in one complex convolution per band)
-> envelope ``|z|`` -> [square] -> mean over fixed-duration bins -> [log10].
A ``(time, ch)`` input gives a ``(time, ch, band)`` output, with ``time`` binned.

Unlike squaring a real band-passed signal, the envelope carries no ripple at twice the
carrier, so the bin mean is the band's power at any bin size.
"""

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import (
    BaseProcessor,
    BaseStatefulProcessor,
    BaseStatefulTransformer,
    BaseTransformerUnit,
    CompositeProcessor,
    processor_state,
    resolve_configured_stream_dim,
    suppress_axis_deprecation,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .binned_aggregate import BinnedAggregateSettings, BinnedAggregateTransformer
from .filterbank import FilterbankMode, FilterbankSettings, FilterbankTransformer
from .kaiser import kaiser_design_fun
from .materialize import MaterializeMode, materialize_array
from .math.abs import AbsTransformer
from .math.log import LogSettings, LogTransformer
from .math.pow import PowSettings, PowTransformer
from .spectrum import OptionsEnum
from .util.minphase import MinPhaseMode
from .util.startup import StartupMode


class EnvelopePowerOutput(OptionsEnum):
    """What :obj:`MultiBandEnvelopePowerTransformer` reports per bin."""

    POWER = "Power"
    """Mean of the squared envelope, ``mean(a**2)``."""

    AMPLITUDE = "Amplitude"
    """Mean of the envelope, ``mean(a)``."""

    LOG_POWER = "Log10 Power"
    """``log10(mean(a**2))``."""


class MultiBandEnvelopePowerSettings(ez.Settings):
    """Settings for :obj:`MultiBandEnvelopePowerTransformer`."""

    bands: list[tuple[float, float]]
    """(lo, hi) band limits in Hz, at the filter's -6 dB (half-amplitude) points, so adjacent
    bands cross over where their amplitude responses sum to one. Each band needs
    ``0 < lo - w/2`` and ``hi + w/2 < fs/2``, with ``w`` its transition width (see
    ``transition_ratio``); otherwise its one-sided kernel would pass negative frequencies,
    and reset raises ``ValueError``."""

    transition_ratio: float = 0.5
    """Transition width of each band's filter, as a fraction of that band's width, in (0, 1].
    The default puts a flat pass-band over the central half of the band. Kernel length, and
    so delay, scales as ``1 / (transition_ratio * (hi - lo))``: the narrowest band sets the
    delay. A wide band's transition reaches into a narrower neighbour: with the default,
    13-30 Hz passes 0.5 % of the power of a 10.5 Hz sinusoid. Lower it for sharper edges at
    the cost of longer kernels."""

    stopband_attenuation: float = 60.0
    """Stop-band attenuation in dB of the Kaiser-window design (``scipy.signal.kaiserord``).
    The pass-band ripple is the same fraction, so 60 dB keeps the gain within 0.1 %."""

    bin_duration: float = 0.02
    """Output bin duration in seconds. See :attr:`BinnedAggregateSettings.bin_duration`."""

    fractional: bool = True
    """See :attr:`BinnedAggregateSettings.fractional`."""

    output: EnvelopePowerOutput = EnvelopePowerOutput.POWER
    """Power (mean of ``a**2``), amplitude (mean of ``a``) or log10 power, where ``a`` is the
    envelope. A sinusoid of amplitude ``a`` in a band reports ``a**2`` power."""

    min_phase: MinPhaseMode = MinPhaseMode.NONE
    """Passed to :attr:`FilterbankSettings.min_phase`; the bands are always aligned on one
    delay (``compensate_delay=True``). NONE keeps every kernel linear phase: alignment across
    bands is exact and the common delay is half the narrowest band's kernel. HOMOMORPHICFULL
    converts only the bands that would set the common delay to minimum phase, for lower
    latency; their delay is then frequency dependent, so alignment is approximate.
    HILBERT and HOMOMORPHIC raise: complex kernels do not support the former, and the latter
    halves the magnitude response in dB."""

    startup: StartupMode = StartupMode.CONSTANT
    """Passed to :attr:`FilterbankSettings.startup`. The default holds the first sample as the
    pre-history, so a DC offset does not ring through the bands at the start of a stream."""

    materialize: MaterializeMode = MaterializeMode.ASYNC
    """How to evaluate the output on a lazy backend (MLX); see
    :obj:`~ezmsg.sigproc.materialize.MaterializeMode`. No-op elsewhere."""


def design_analytic_kernels(settings: MultiBandEnvelopePowerSettings, fs: float) -> list[npt.NDArray]:
    """One complex, linear-phase kernel per band: a Kaiser low-pass prototype of cutoff
    ``(hi - lo) / 2`` modulated to the band centre and doubled, so ``|output|`` is the
    amplitude of an in-band sinusoid."""
    if settings.min_phase in (MinPhaseMode.HILBERT, MinPhaseMode.HOMOMORPHIC):
        raise ValueError(f"min_phase={settings.min_phase.name} is not supported; use NONE or HOMOMORPHICFULL.")
    if not 0 < settings.transition_ratio <= 1:
        raise ValueError(f"transition_ratio must be in (0, 1], got {settings.transition_ratio}.")
    nyq = fs / 2
    kernels = []
    for lo, hi in settings.bands:
        if not 0 < lo < hi < nyq:
            raise ValueError(f"Band ({lo}, {hi}) Hz must satisfy 0 < lo < hi < Nyquist ({nyq} Hz).")
        width = hi - lo
        tw = settings.transition_ratio * width
        if lo - tw / 2 <= 0 or hi + tw / 2 >= nyq:
            raise ValueError(
                f"Band ({lo}, {hi}) Hz with a {tw:g} Hz transition reaches 0 Hz or Nyquist ({nyq} Hz), "
                "so its analytic kernel would pass negative frequencies. "
                "Move the band away from the edge, narrow it, or lower transition_ratio."
            )
        proto = kaiser_design_fun(fs, cutoff=width / 2, ripple=settings.stopband_attenuation, width=tw)[0]
        n = np.arange(proto.size) - (proto.size - 1) / 2
        kernels.append(2 * proto * np.exp(2j * np.pi * (lo + hi) / 2 * n / fs))
    return kernels


@processor_state
class AnalyticFilterbankState:
    axis: str = ""
    filterbank: FilterbankTransformer | None = None


class AnalyticFilterbankTransformer(
    BaseStatefulTransformer[MultiBandEnvelopePowerSettings, AxisArray, AxisArray, AnalyticFilterbankState]
):
    """First stage of :obj:`MultiBandEnvelopePowerTransformer`: designs the analytic kernels
    from the sample rate at reset and applies them, delay-compensated. Output is complex,
    with the input's dims followed by ``band``."""

    def _hash_message(self, message: AxisArray) -> int:
        # As FilterbankDesignTransformer: kernels depend on the sample rate only; the inner
        # filterbank rebuilds itself when the channels change.
        axis = resolve_configured_stream_dim(self, message, None)
        return hash((message.key, getattr(message.axes.get(axis), "gain", None)))

    def _reset_state(self, message: AxisArray) -> None:
        self._state.axis = resolve_configured_stream_dim(self, message, None)
        fs = 1 / message.axes[self._state.axis].gain
        with suppress_axis_deprecation():
            self._state.filterbank = FilterbankTransformer(
                FilterbankSettings(
                    kernels=design_analytic_kernels(self.settings, fs),
                    mode=FilterbankMode.CONV,
                    min_phase=self.settings.min_phase,
                    compensate_delay=True,
                    startup=self.settings.startup,
                    axis=self._state.axis,
                    new_axis="band",
                    bands=[tuple(b) for b in self.settings.bands],
                )
            )

    def _process(self, message: AxisArray) -> AxisArray:
        out = self._state.filterbank(message)  # dims: (*other, band, axis)
        return replace(
            out,
            data=np.moveaxis(out.data, -1, message.get_axis_idx(self._state.axis)),
            dims=list(message.dims) + ["band"],
            stream_dim=self._state.axis,
        )


class MultiBandEnvelopePowerTransformer(CompositeProcessor[MultiBandEnvelopePowerSettings, AxisArray, AxisArray]):
    """
    Binned envelope power in several bands at once.

    Pipeline: analytic filterbank -> abs -> [square] -> binned mean -> [log10].
    ``(time, ch)`` in, ``(time, ch, band)`` out. The ``band`` axis is a
    :obj:`~ezmsg.util.messages.axisarray.AxisArray.CoordinateAxis` whose data is a structured
    array with fields ``lo`` and ``hi`` (Hz). All bands share one delay, which is subtracted
    from the time axis offset, so output timestamps refer to the input samples they describe.
    """

    @staticmethod
    def _initialize_processors(
        settings: MultiBandEnvelopePowerSettings,
    ) -> dict[str, BaseProcessor | BaseStatefulProcessor]:
        procs: dict[str, BaseProcessor | BaseStatefulProcessor] = {
            "filterbank": AnalyticFilterbankTransformer(settings),
            "envelope": AbsTransformer(),
        }
        if settings.output != EnvelopePowerOutput.AMPLITUDE:
            procs["square"] = PowTransformer(PowSettings(exponent=2.0))
        procs["bin"] = BinnedAggregateTransformer(
            BinnedAggregateSettings(bin_duration=settings.bin_duration, fractional=settings.fractional)
        )
        if settings.output == EnvelopePowerOutput.LOG_POWER:
            procs["log"] = LogTransformer(LogSettings(base=10.0, clip_zero=True))
        return procs

    def _post_process(self, result: AxisArray | None) -> AxisArray | None:
        if result is not None:
            materialize_array(result.data, self.settings.materialize)
        return result


class MultiBandEnvelopePower(
    BaseTransformerUnit[MultiBandEnvelopePowerSettings, AxisArray, AxisArray, MultiBandEnvelopePowerTransformer]
):
    SETTINGS = MultiBandEnvelopePowerSettings
