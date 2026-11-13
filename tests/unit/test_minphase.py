import numpy as np
import pytest
import scipy.signal as sps
from ezmsg.util.messages.axisarray import slice_along_axis
from ezmsg.util.messages.util import replace

from ezmsg.sigproc.butterworthfilter import ButterworthFilterSettings, ButterworthFilterTransformer
from ezmsg.sigproc.butterworthzerophase import (
    ButterworthBackwardFilterTransformer,
    ButterworthZeroPhaseSettings,
    ButterworthZeroPhaseTransformer,
)
from ezmsg.sigproc.combfilter import CombFilterSettings, CombFilterTransformer
from ezmsg.sigproc.filter import FilterSettings
from ezmsg.sigproc.filterbankdesign import FilterbankDesignSettings, FilterbankDesignTransformer
from ezmsg.sigproc.fir_hilbert import (
    FIRHilbertEnvelopeTransformer,
    FIRHilbertFilterSettings,
    FIRHilbertFilterTransformer,
)
from ezmsg.sigproc.kaiser import KaiserFilterSettings, KaiserFilterTransformer, kaiser_design_fun
from ezmsg.sigproc.util.minphase import MinPhaseMode, min_phase_kernel
from tests.helpers.empty_time import make_msg

FS = 1000.0
FC = 100.0  # passband centre, Hz
N_TAPS = 201


def _real_kernel():
    return sps.firwin(N_TAPS, [FC - 20, FC + 20], pass_zero=False, window=("kaiser", 6.0), fs=FS)


def _analytic_kernel():
    """One-sided band-pass: a real low-pass prototype shifted to +FC, linear phase."""
    lp = sps.firwin(N_TAPS, 20, window=("kaiser", 6.0), fs=FS)
    n = np.arange(N_TAPS) - (N_TAPS - 1) / 2
    return lp * np.exp(2j * np.pi * FC * n / FS)


def _mag(h, freqs):
    return np.abs(sps.freqz(h, worN=freqs, fs=FS)[1])


def _group_delay(h, f):
    return sps.group_delay((h, [1.0]), w=[f], fs=FS)[1][0]


FREQS = np.linspace(-FS / 2, FS / 2, 4001)


@pytest.mark.parametrize("make_kernel", [_real_kernel, _analytic_kernel])
def test_homomorphicfull_preserves_magnitude_and_cuts_delay(make_kernel):
    h = make_kernel()
    h_min = min_phase_kernel(h, MinPhaseMode.HOMOMORPHICFULL)
    assert h_min.shape == h.shape
    assert np.iscomplexobj(h_min) == np.iscomplexobj(h)
    ref = _mag(h, FREQS)
    # Max deviation over the whole axis, relative to the passband peak.
    assert np.max(np.abs(_mag(h_min, FREQS) - ref)) < 1e-4 * ref.max()
    assert _group_delay(h_min, FC) < 0.5 * _group_delay(h, FC)


@pytest.mark.parametrize("make_kernel", [_real_kernel, _analytic_kernel])
def test_homomorphic_half_length_sqrt_magnitude(make_kernel):
    h = make_kernel()
    h_min = min_phase_kernel(h, MinPhaseMode.HOMOMORPHIC)
    assert h_min.shape == ((N_TAPS + 1) // 2,)
    ref = np.sqrt(_mag(h, FREQS))
    # Loose: the half-length kernel only approximates sqrt(|H|); scipy's real result is no closer (~2.6%).
    assert np.max(np.abs(_mag(h_min, FREQS) - ref)) < 5e-2 * ref.max()
    assert _group_delay(h_min, FC) < _group_delay(h, FC)


@pytest.mark.parametrize(
    "mode, kwargs",
    [(MinPhaseMode.HILBERT, {"method": "hilbert"}), (MinPhaseMode.HOMOMORPHIC, {"method": "homomorphic"})],
)
def test_real_matches_scipy(mode, kwargs):
    h = _real_kernel()
    assert np.array_equal(min_phase_kernel(h, mode), sps.minimum_phase(h, **kwargs))
    assert np.array_equal(
        min_phase_kernel(h, MinPhaseMode.HOMOMORPHICFULL), sps.minimum_phase(h, method="homomorphic", half=False)
    )


def test_none_is_identity():
    h = _analytic_kernel()
    assert min_phase_kernel(h, MinPhaseMode.NONE) is h


@pytest.mark.parametrize("mode", [MinPhaseMode.HOMOMORPHIC, MinPhaseMode.HOMOMORPHICFULL])
def test_complex_stays_one_sided(mode):
    h = _analytic_kernel()
    neg = (FREQS > -FC - 20) & (FREQS < -FC + 20)  # mirror image of the passband

    def rejection_db(k):
        mag = _mag(k, FREQS)
        return 20 * np.log10(mag[neg].max() / mag.max())

    before = rejection_db(h)
    # HOMOMORPHIC square-roots the magnitude, which halves the rejection in dB.
    expected = before / 2 if mode == MinPhaseMode.HOMOMORPHIC else before
    assert before < -50
    assert rejection_db(min_phase_kernel(h, mode)) < expected + 3


def test_complex_hilbert_raises():
    with pytest.raises(ValueError, match="HILBERT"):
        min_phase_kernel(_analytic_kernel(), MinPhaseMode.HILBERT)


# --- Single-kernel filter designs ---

KAISER_KW = dict(cutoff=[80.0, 120.0], ripple=60.0, width=10.0, pass_zero="bandpass")


def test_kaiser_min_phase_matches_lfilter():
    mode = MinPhaseMode.HOMOMORPHICFULL
    msg = make_msg(n_time=1000, n_ch=3, fs=FS)
    proc = KaiserFilterTransformer(KaiserFilterSettings(axis="time", min_phase=mode, **KAISER_KW))
    out = np.concatenate([proc(msg_chunk).data for msg_chunk in _chunks(msg, 100)], axis=0)

    b = min_phase_kernel(kaiser_design_fun(FS, **KAISER_KW)[0], mode)
    # The filter starts from the constant-x0 steady state (edge-scaled lfilter_zi); mirror that.
    zi = sps.lfilter_zi(b, [1.0])[:, None] * msg.data[:1]
    expected, _ = sps.lfilter(b, [1.0], msg.data, axis=0, zi=zi)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)

    # A live settings change goes through the same design step.
    proc.update_settings(cutoff=[60.0, 90.0])
    proc(msg)
    b_new = min_phase_kernel(kaiser_design_fun(FS, **{**KAISER_KW, "cutoff": [60.0, 90.0]})[0], mode)
    np.testing.assert_array_equal(proc.state.filter.settings.coefs[0], b_new)


def _chunks(msg, step):
    for i in range(0, msg.data.shape[0], step):
        yield replace(msg, data=slice_along_axis(msg.data, slice(i, i + step), 0))


MP = MinPhaseMode.HOMOMORPHICFULL


@pytest.mark.parametrize(
    "make_proc, match",
    [
        (lambda: ButterworthFilterTransformer(ButterworthFilterSettings(order=4, cutoff=50.0, min_phase=MP)), "FIR"),
        (
            lambda: ButterworthFilterTransformer(
                ButterworthFilterSettings(order=4, cutoff=50.0, coef_type="sos", min_phase=MP)
            ),
            "FIR",
        ),
        (
            lambda: KaiserFilterTransformer(KaiserFilterSettings(coef_type="sos", min_phase=MP, **KAISER_KW)),
            "coef_type='ba'",
        ),
        (lambda: CombFilterTransformer(CombFilterSettings(min_phase=MP)), "FIR"),
        (
            lambda: ButterworthBackwardFilterTransformer(
                ButterworthZeroPhaseSettings(order=4, cutoff=50.0, min_phase=MP)
            ),
            "FIR",
        ),
        (
            lambda: ButterworthZeroPhaseTransformer(ButterworthZeroPhaseSettings(order=4, cutoff=50.0, min_phase=MP)),
            "FIR",
        ),
        (lambda: FIRHilbertFilterTransformer(FIRHilbertFilterSettings(f_lo=5.0, min_phase=MP)), "Hilbert"),
        (lambda: FIRHilbertEnvelopeTransformer(FIRHilbertFilterSettings(f_lo=5.0, min_phase=MP)), "Hilbert"),
        (
            lambda: FilterbankDesignTransformer(
                FilterbankDesignSettings(filters=[KaiserFilterSettings(min_phase=MP, **KAISER_KW)])
            ),
            "FilterbankDesignSettings.min_phase",
        ),
    ],
)
def test_unsupported_min_phase_raises(make_proc, match):
    proc = make_proc()
    with pytest.raises(ValueError, match=match):
        proc(make_msg(n_time=100, n_ch=2, fs=FS))


def test_filter_settings_rejects_min_phase():
    with pytest.raises(ValueError, match="min_phase"):
        FilterSettings(coefs=(np.ones(3), np.ones(1)), min_phase=MP)
