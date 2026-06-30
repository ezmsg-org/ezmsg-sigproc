"""Unit tests for the fused resample-onto-reference + concatenate transformer."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.resampleconcat import ResampleConcatProcessor, ResampleConcatSettings


def _mk(fs: float, offset: float, n: int, n_ch: int, key: str, encode_time: bool = False) -> AxisArray:
    """Build a [time, ch] message on a LinearAxis.

    When ``encode_time`` is True, ``data[:, j] == time + 1000*j`` so a test can verify
    which samples ended up where after resampling/concatenation.
    """
    t = offset + np.arange(n) / fs
    if encode_time:
        data = t[:, None] + np.arange(n_ch)[None, :] * 1000.0
    else:
        data = np.random.randn(n, n_ch)
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.LinearAxis(gain=1 / fs, offset=offset, unit="s"),
            "ch": AxisArray.CoordinateAxis(
                data=np.array([f"{key}{i}" for i in range(n_ch)]), dims=["ch"], unit="label"
            ),
        },
        key=key,
    )


def test_resampleconcat_combines_two_rates():
    """Reference (4 ch) + signal resampled onto it (2 ch) -> 6 ch on a shared axis."""
    fs_ref, fs_sig, chunk = 100.0, 99.7, 30
    proc = ResampleConcatProcessor(
        ResampleConcatSettings(axis="time", concat_axis="ch", buffer_duration=4.0, label_a="_h1", label_b="_h2")
    )
    outs = []
    for i in range(12):
        proc.push_reference(_mk(fs_ref, i * chunk / fs_ref, chunk, 4, "h1", encode_time=True))
        r = proc(_mk(fs_sig, i * chunk / fs_sig, chunk, 2, "h2", encode_time=True))
        if r is not None and r.data.shape[0] > 0:
            outs.append(r)

    assert outs, "Composite produced no output."
    cat = AxisArray.concatenate(*outs, dim="time")
    assert cat.data.shape[1] == 6, "Expected 4 (reference) + 2 (signal) = 6 channels."

    labels = list(cat.axes["ch"].data)
    assert all(lbl.endswith("_h1") for lbl in labels[:4])
    assert all(lbl.endswith("_h2") for lbl in labels[4:])

    t = cat.axes["time"].value(np.arange(cat.data.shape[0]))
    # Side A is the reference gathered exactly on the output grid: data col 0 == time.
    assert np.max(np.abs(cat.data[:, 0] - t)) < 1e-9
    # Side B is the signal linearly resampled onto the same grid: data col 0 == time.
    assert np.max(np.abs(cat.data[:, 4] - t)) < 1e-6


def test_resampleconcat_no_signal_yields_nothing():
    """With reference but no signal, there is nothing to resample/concatenate."""
    proc = ResampleConcatProcessor(ResampleConcatSettings(buffer_duration=4.0))
    for i in range(4):
        proc.push_reference(_mk(100.0, i * 30 / 100.0, 30, 4, "h1"))
    assert next(proc) is None


def test_resampleconcat_recovers_from_reference_reset():
    """A sustained backward jump in the reference clock must not stop output forever."""
    fs, chunk = 100.0, 30
    proc = ResampleConcatProcessor(ResampleConcatSettings(buffer_duration=4.0, reference_reset_after_chunks=3))
    post_reset = 0
    with pytest.warns(RuntimeWarning):
        for i in range(25):
            off = i * chunk / fs - (100.0 if i >= 10 else 0.0)  # big reset at chunk 10
            proc.push_reference(_mk(fs, off, chunk, 4, "h1"))
            r = proc(_mk(fs, i * chunk / fs, chunk, 2, "h2"))
            if i >= 14 and r is not None and r.data.shape[0] > 0:
                post_reset += r.data.shape[0]
    assert post_reset > 0, "Composite did not recover after the reference clock reset."
