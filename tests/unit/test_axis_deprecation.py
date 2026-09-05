"""The deprecation window for the per-processor ``axis`` setting.

Deprecated in 3.8, removed in 4.0. See :mod:`ezmsg.sigproc.util.deprecation`.

The contract under test has four parts:

* setting ``axis`` warns, once, pointing at the *caller's* line;
* leaving it unset is silent, including across message processing;
* the setting is still honoured, so nothing changes behaviour until removal;
* stages that forward the setting internally do not warn on the user's behalf.

The last one is what makes the window usable: without it, every filter-by-design
would warn on every state reset, mid-stream, naming whatever happened to be
driving the pipeline.
"""

import inspect
import sys
import warnings

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.butterworthfilter import (
    ButterworthFilterSettings,
    ButterworthFilterTransformer,
)
from ezmsg.sigproc.decimate import DecimateSettings
from ezmsg.sigproc.diff import DiffSettings, DiffTransformer
from ezmsg.sigproc.flatten import FlattenSettings, FlattenTransformer
from ezmsg.sigproc.gaussiansmoothing import (
    GaussianSmoothingFilterTransformer,
    GaussianSmoothingSettings,
)
from ezmsg.sigproc.merge import MergeProcessor, MergeSettings
from ezmsg.sigproc.scaler import (
    AdaptiveStandardScalerSettings,
    AdaptiveStandardScalerTransformer,
)
from ezmsg.sigproc.util.deprecation import (
    AXIS_REMOVAL_VERSION,
    suppress_axis_deprecation,
)
from ezmsg.sigproc.window import WindowSettings, WindowTransformer

FS = 100.0

# Every Settings class carrying the deprecation hook. Pinned so that adding or
# dropping one is a deliberate edit rather than a side effect, and so the 4.0
# removal has a checklist.
DEPRECATED_SETTINGS = {
    "AdaptiveLNCSettings",
    "AdaptiveLatticeNotchFilterSettings",
    "AdaptiveStandardScalerSettings",
    "AlignAlongAxisSettings",
    "BinnedAggregateSettings",
    "ButterworthFilterSettings",
    "ButterworthZeroPhaseSettings",
    "CWTSettings",
    "ChebyshevFilterSettings",
    "CombFilterSettings",
    "DecimateSettings",
    "DiffSettings",
    "EWMASettings",
    "FIRFilterSettings",
    "FIRHilbertFilterSettings",
    "FilterBaseSettings",
    "FilterSettings",
    "FilterbankDesignSettings",
    "FilterbankSettings",
    "GaussianSmoothingSettings",
    "KaiserFilterSettings",
    "ParksMcClellanFIRSettings",
    "ResampleSettings",
    "RiverAdaptiveStandardScalerSettings",
    "RollingScalerSettings",
    "SamplerSettings",
    "WindowSettings",
}

# The second wave: these defaulted to a hardcoded ``axis="time"`` rather than to
# a positional guess, so flipping them to follow ``chunk_dim`` changes results
# wherever the chunk dimension is not ``"time"``. They pass ``legacy_default``
# to surface that population; the rest do not.
LEGACY_TIME_DEFAULT = {
    "AdaptiveLNCSettings",
    "AdaptiveLatticeNotchFilterSettings",
    "BinnedAggregateSettings",
    "CWTSettings",
    "FilterbankDesignSettings",
    "ResampleSettings",
    "RollingScalerSettings",
}


def msg(n_time=32, n_ch=3, chunk_dim="time"):
    kwargs = {"chunk_dim": chunk_dim} if chunk_dim else {}
    return AxisArray(
        np.random.default_rng(0).standard_normal((n_time, n_ch)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=FS)},
        key="dev",
        **kwargs,
    )


def axis_warnings(records):
    return [r for r in records if issubclass(r.category, FutureWarning) and "deprecated" in str(r.message)]


class TestTheInventoryIsPinned:
    def test_exactly_these_classes_carry_the_hook(self):
        import importlib
        import pkgutil

        import ezmsg.sigproc

        # The walk reads sys.modules, so every submodule has to be imported
        # first -- otherwise this passes by simply not looking at most of them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for info in pkgutil.walk_packages(ezmsg.sigproc.__path__, "ezmsg.sigproc."):
                try:
                    importlib.import_module(info.name)
                except Exception:  # optional deps (river, mlx, ...) may be absent
                    pass

        found = set()
        for name, mod in list(sys.modules.items()):
            if not name.startswith("ezmsg.sigproc"):
                continue
            for obj in vars(mod).values():
                if inspect.isclass(obj) and issubclass(obj, ez.Settings) and "__post_init__" in dir(obj):
                    try:
                        src = inspect.getsource(obj.__post_init__)
                    except (OSError, TypeError):
                        continue
                    if "warn_axis_deprecated" in src:
                        found.add(obj.__name__)
        assert found == DEPRECATED_SETTINGS

    def test_flatten_is_deliberately_excluded(self):
        """Flatten carries no data between messages and already handles the chunk
        dimension being folded away, so preserving a non-chunk axis is coherent."""
        assert "FlattenSettings" not in DEPRECATED_SETTINGS
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            FlattenTransformer(FlattenSettings(preserve_axis="time"))(msg())
        assert not axis_warnings(rec)


class TestSettingItWarns:
    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda: WindowSettings(axis="time", window_dur=0.1, window_shift=0.05), id="window"),
            pytest.param(lambda: DiffSettings(axis="time"), id="diff"),
            pytest.param(lambda: ButterworthFilterSettings(axis="time", order=2, cutoff=20.0), id="butterworth"),
            pytest.param(lambda: AdaptiveStandardScalerSettings(axis="time"), id="scaler"),
            pytest.param(lambda: DecimateSettings(axis="time", target_rate=50.0), id="decimate"),
        ],
    )
    def test_it_warns_once(self, build):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            build()
        assert len(axis_warnings(rec)) == 1

    def test_the_warning_names_the_removal_version(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            DiffSettings(axis="time")
        assert AXIS_REMOVAL_VERSION in str(axis_warnings(rec)[0].message)

    def test_it_is_a_futurewarning_not_a_deprecationwarning(self):
        """DeprecationWarning is suppressed by default outside __main__, so a
        pipeline -- library code -- would never see it. ezmsg core uses
        FutureWarning for its own deprecations."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            DiffSettings(axis="time")
        assert rec[0].category is FutureWarning

    def test_it_blames_the_callers_line_not_ours_via_settings(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            WindowSettings(axis="time", window_dur=0.1, window_shift=0.05)
            expected_line = inspect.currentframe().f_lineno - 1
        (record,) = axis_warnings(rec)
        assert (record.filename, record.lineno) == (__file__, expected_line)

    def test_it_blames_the_callers_line_not_ours_via_transformer(self):
        """The depth differs from the path above -- the transformer builds its
        settings through ``_unify_settings`` -- so a fixed stacklevel would be
        wrong for at least one of the two."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            WindowTransformer(axis="time", window_dur=0.1, window_shift=0.05)
            expected_line = inspect.currentframe().f_lineno - 1
        (record,) = axis_warnings(rec)
        assert (record.filename, record.lineno) == (__file__, expected_line)

    def test_the_factory_functions_blame_the_caller_too(self):
        from ezmsg.sigproc.scaler import scaler_np

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            scaler_np(time_constant=1.0, axis="time")
            expected_line = inspect.currentframe().f_lineno - 1
        (record,) = axis_warnings(rec)
        assert (record.filename, record.lineno) == (__file__, expected_line)


class TestLeavingItUnsetIsSilent:
    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda: WindowTransformer(window_dur=0.1, window_shift=0.05), id="window"),
            pytest.param(lambda: DiffTransformer(DiffSettings()), id="diff"),
            pytest.param(
                lambda: ButterworthFilterTransformer(ButterworthFilterSettings(order=2, cutoff=20.0)),
                id="butterworth",
            ),
            pytest.param(
                lambda: GaussianSmoothingFilterTransformer(GaussianSmoothingSettings(sigma=0.01)),
                id="gaussian",
            ),
            pytest.param(
                lambda: AdaptiveStandardScalerTransformer(AdaptiveStandardScalerSettings()),
                id="scaler",
            ),
        ],
    )
    def test_construction_and_processing_are_both_silent(self, build):
        """Processing matters as much as construction: filter-by-design rebuilds a
        child FilterSettings on every state reset."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            proc = build()
            proc(msg())
            proc(msg())
        assert not axis_warnings(rec)

    def test_merge_with_no_align_axis_is_silent(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            MergeProcessor(MergeSettings(axis="ch"))
        assert not axis_warnings(rec)


class TestInternalForwardingDoesNotMultiply:
    def test_the_scaler_warns_once_for_its_two_child_ewmas(self):
        """AdaptiveStandardScaler builds two EWMATransformers from its own axis,
        inside ``_reset_state`` -- so without suppression this would warn twice
        per reset, mid-stream, naming the pipeline driver."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            proc = AdaptiveStandardScalerTransformer(AdaptiveStandardScalerSettings(axis="time"))
            proc(msg())
            proc(msg())
        assert len(axis_warnings(rec)) == 1

    def test_filter_by_design_warns_once_across_resets(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            proc = ButterworthFilterTransformer(ButterworthFilterSettings(axis="time", order=2, cutoff=20.0))
            proc(msg())
            proc(msg(n_ch=5))  # a channel change forces a fresh reset
        assert len(axis_warnings(rec)) == 1

    def test_the_context_manager_silences_and_restores(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with suppress_axis_deprecation():
                DiffSettings(axis="time")
            assert not axis_warnings(rec)
            DiffSettings(axis="time")
        assert len(axis_warnings(rec)) == 1


class TestBehaviourIsUnchangedUntilRemoval:
    """A deprecation window that changed behaviour would not be a window."""

    def test_a_configured_axis_is_still_honoured(self):
        transposed = AxisArray(
            np.arange(24, dtype=float).reshape(3, 8),
            dims=["ch", "time"],
            axes={"time": AxisArray.TimeAxis(fs=FS)},
            key="dev",
            chunk_dim="time",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # `ch` is not the chunk dim, and is exactly what removal will stop
            # allowing -- but today it must still be obeyed.
            proc = DiffTransformer(DiffSettings(axis="ch"))
            out = proc(transposed)
        assert out.data.shape == (3, 8)
        # Differences along `ch` are 8 apart in this row-major fixture.
        assert np.allclose(out.data[1:, :], 8.0)

    def test_the_runtime_mismatch_warning_still_fires_alongside(self, caplog):
        """The construction-time warning says "delete this"; the runtime one says
        "deleting this will change what the stage computes"."""
        windowed = AxisArray(
            np.zeros((4, 8, 3)),
            dims=["win", "time", "ch"],
            axes={"win": AxisArray.TimeAxis(fs=FS / 8), "time": AxisArray.TimeAxis(fs=FS)},
            key="dev",
            chunk_dim="win",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proc = DiffTransformer(DiffSettings(axis="time"))
            with caplog.at_level("WARNING"):
                proc(windowed)
        assert any("chunk_dim" in r.message for r in caplog.records)


class TestTheLegacyTimeDefault:
    """The second wave defaulted to a hardcoded ``axis="time"``, not to a
    positional guess. Flipping those to follow ``chunk_dim`` changes results
    wherever the chunk dimension is not ``"time"`` -- most obviously downstream
    of a windowing stage -- and no setting exists to warn about, because the
    affected caller set nothing. ``legacy_default`` is what surfaces them."""

    @staticmethod
    def _windowed():
        """``(win, time, ch)``: the old default would pick ``time``, the new one
        picks ``win``."""
        return AxisArray(
            np.random.default_rng(0).standard_normal((4, 8, 2)),
            dims=["win", "time", "ch"],
            axes={"win": AxisArray.TimeAxis(fs=FS / 8), "time": AxisArray.TimeAxis(fs=FS)},
            key="dev",
            chunk_dim="win",
        )

    def test_it_warns_when_the_resolved_dim_is_not_the_old_default(self, caplog):
        from ezmsg.sigproc.util.message import resolve_configured_chunk_dim

        class Proc:
            STREAMING_DIMS = ("time",)

        proc = Proc()
        with caplog.at_level("WARNING"):
            resolved = resolve_configured_chunk_dim(proc, self._windowed(), None, legacy_default="time")
        assert resolved == "win"
        assert any("used to operate on axis='time'" in r.message for r in caplog.records)

    def test_it_warns_only_once(self, caplog):
        from ezmsg.sigproc.util.message import resolve_configured_chunk_dim

        class Proc:
            STREAMING_DIMS = ("time",)

        proc = Proc()
        with caplog.at_level("WARNING"):
            for _ in range(3):
                resolve_configured_chunk_dim(proc, self._windowed(), None, legacy_default="time")
        assert sum("used to operate on" in r.message for r in caplog.records) == 1

    def test_a_raw_stream_is_silent(self, caplog):
        """The overwhelmingly common case: chunk_dim is already "time", so
        nothing changed and there is nothing to say."""
        from ezmsg.sigproc.util.message import resolve_configured_chunk_dim

        class Proc:
            STREAMING_DIMS = ("time",)

        with caplog.at_level("WARNING"):
            resolved = resolve_configured_chunk_dim(Proc(), msg(), None, legacy_default="time")
        assert resolved == "time"
        assert not caplog.records

    def test_a_stream_without_the_old_default_dim_is_silent(self, caplog):
        """If ``time`` is not even present, the old default could not have been
        operating on it, so there is no behaviour change to report."""
        from ezmsg.sigproc.util.message import resolve_configured_chunk_dim

        class Proc:
            STREAMING_DIMS = ("time",)

        no_time = AxisArray(
            np.zeros((4, 2)),
            dims=["win", "ch"],
            axes={"win": AxisArray.TimeAxis(fs=FS)},
            key="dev",
            chunk_dim="win",
        )
        with caplog.at_level("WARNING"):
            resolve_configured_chunk_dim(Proc(), no_time, None, legacy_default="time")
        assert not caplog.records

    def test_stages_still_follow_the_declaration_end_to_end(self, caplog):
        """RollingScaler is the cheapest of the seven to drive; the point is that
        the resolved axis reaches the state, not the arithmetic."""
        from ezmsg.sigproc.rollingscaler import RollingScalerProcessor, RollingScalerSettings

        proc = RollingScalerProcessor(RollingScalerSettings(window_size=0.1))
        with caplog.at_level("WARNING"):
            proc(self._windowed())
        assert proc.state.axis == "win"

    def test_passing_the_old_default_explicitly_preserves_behaviour(self):
        """The escape hatch the warning points at: during the window, callers can
        pin the old axis rather than accept the new resolution."""
        from ezmsg.sigproc.rollingscaler import RollingScalerProcessor, RollingScalerSettings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proc = RollingScalerProcessor(RollingScalerSettings(window_size=0.1, axis="time"))
            proc(self._windowed())
        assert proc.state.axis == "time"


class TestTheSecondWaveWarnsOnExplicitUse:
    """The seven stages whose default was a hardcoded ``"time"``. Their settings
    now default to ``None`` like the rest, so an explicit ``axis=`` is what warns."""

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda a: _lnc(a), id="adaptive_lnc"),
            pytest.param(lambda a: _lattice(a), id="adaptive_lattice_notch"),
            pytest.param(lambda a: _binned(a), id="binned_aggregate"),
            pytest.param(lambda a: _rolling(a), id="rollingscaler"),
            pytest.param(lambda a: _resample(a), id="resample"),
            pytest.param(lambda a: _cwt(a), id="wavelets"),
            pytest.param(lambda a: _fbdesign(a), id="filterbankdesign"),
        ],
    )
    def test_explicit_warns_and_unset_is_silent(self, build):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            build("time")
        assert len(axis_warnings(rec)) == 1

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            build(None)
        assert not axis_warnings(rec)

    @pytest.mark.parametrize("name", sorted(LEGACY_TIME_DEFAULT))
    def test_every_legacy_default_class_is_also_deprecated(self, name):
        assert name in DEPRECATED_SETTINGS


def _lnc(axis):
    from ezmsg.sigproc.adaptive_lnc import AdaptiveLNCSettings

    return AdaptiveLNCSettings(axis=axis)


def _lattice(axis):
    from ezmsg.sigproc.adaptive_lattice_notch import AdaptiveLatticeNotchFilterSettings

    return AdaptiveLatticeNotchFilterSettings(axis=axis)


def _binned(axis):
    from ezmsg.sigproc.binned_aggregate import BinnedAggregateSettings

    return BinnedAggregateSettings(axis=axis)


def _rolling(axis):
    from ezmsg.sigproc.rollingscaler import RollingScalerSettings

    return RollingScalerSettings(axis=axis)


def _resample(axis):
    from ezmsg.sigproc.resample import ResampleSettings

    return ResampleSettings(axis=axis)


def _cwt(axis):
    from ezmsg.sigproc.wavelets import CWTSettings

    return CWTSettings(axis=axis, wavelet="morl", frequencies=[10.0, 20.0])


def _fbdesign(axis):
    from ezmsg.sigproc.filterbankdesign import FilterbankDesignSettings

    return FilterbankDesignSettings(filters=[], axis=axis)
