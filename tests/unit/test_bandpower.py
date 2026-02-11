import copy

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.bandpower import (
    AggregationFunction,
    BandPowerSettings,
    BandPowerTransformer,
    SpectrogramSettings,
)
from tests.helpers.util import assert_messages_equal, create_messages_with_periodic_signal, requires_mlx


def _debug_plot(result):
    import matplotlib.pyplot as plt

    t_vec = result.axes["time"].value(np.arange(result.data.shape[0]))
    plt.plot(t_vec, result.data[..., 0])


def test_bandpower():
    win_dur = 1.0
    fs = 1000.0
    bands = [(9, 11), (70, 90), (134, 136)]

    sin_params = [
        {"f": 10.0, "a": 3.0, "dur": 4.0, "offset": 1.0},
        {"f": 10.0, "a": 1.0, "dur": 3.0, "offset": 5.0},
        {"f": 135.0, "a": 4.0, "dur": 4.0, "offset": 1.0},
        {"f": 135.0, "a": 2.0, "dur": 3.0, "offset": 5.0},
    ]
    messages = create_messages_with_periodic_signal(
        sin_params=sin_params,
        fs=fs,
        msg_dur=0.4,
        win_step_dur=None,  # The spectrogram will do the windowing
    )

    # Grab a deepcopy backup of the inputs, so we can check the inputs didn't change
    #  while being processed.
    backup = [copy.deepcopy(_) for _ in messages]

    xformer = BandPowerTransformer(
        BandPowerSettings(
            spectrogram_settings=SpectrogramSettings(
                window_dur=win_dur,
                window_shift=0.1,
            ),
            bands=bands,
            aggregation=AggregationFunction.MEAN,
        )
    )
    results = [xformer(_) for _ in messages]

    assert_messages_equal(messages, backup)

    result = AxisArray.concatenate(*results, dim="time")
    # _debug_plot(result)

    # Check the amplitudes at the midpoints of each of our sinusoids.
    t_vec = result.axes["time"].value(np.arange(result.data.shape[0]))
    mags = []
    for s_p in sin_params[:2]:
        ix = np.argmin(np.abs(t_vec - (s_p["offset"] + s_p["dur"] / 2)))
        mags.append(result.data[ix, 0, 0])
    for s_p in sin_params[2:]:
        ix = np.argmin(np.abs(t_vec - (s_p["offset"] + s_p["dur"] / 2)))
        mags.append(result.data[ix, 2, 0])
    # The sorting of the measured magnitudes should match the sorting of the parameter magnitudes.
    assert np.array_equal(np.argsort(mags), np.argsort([_["a"] for _ in sin_params]))


@requires_mlx
@pytest.mark.benchmark(group="bandpower")
@pytest.mark.parametrize("n_channels", [32, 256, 1024])
@pytest.mark.parametrize("backend", ["numpy", "mlx"])
def test_bandpower_benchmark(backend, n_channels, benchmark):
    """Benchmark BandPowerTransformer end-to-end: numpy vs MLX input."""
    fs = 1000.0
    chunk_samples = 500
    n_chunks = 20
    win_dur = 0.5
    win_shift = 0.1
    bands = [(8, 13), (13, 30), (30, 70), (70, 150)]

    settings = BandPowerSettings(
        spectrogram_settings=SpectrogramSettings(
            window_dur=win_dur,
            window_shift=win_shift,
        ),
        bands=bands,
        aggregation=AggregationFunction.MEAN,
    )

    # Pre-generate chunk messages as numpy
    rng = np.random.default_rng(42)
    np_chunks = []
    for i in range(n_chunks + 1):  # +1 for warmup
        data = rng.standard_normal((chunk_samples, n_channels)).astype(np.float32)
        np_chunks.append(
            AxisArray(
                data,
                dims=["time", "ch"],
                axes={"time": AxisArray.LinearAxis(gain=1.0 / fs, offset=i * chunk_samples / fs)},
            )
        )

    if backend == "mlx":
        import mlx.core as mx

        chunks = [AxisArray(data=mx.array(chunk.data), dims=chunk.dims, axes=chunk.axes) for chunk in np_chunks]
    else:
        chunks = np_chunks

    xformer = BandPowerTransformer(settings)
    xformer(chunks[0])  # Warmup

    def process_all_chunks():
        outputs = [xformer(chunk) for chunk in chunks[1:]]
        if backend == "mlx":
            mx.eval(*[o.data for o in outputs])
        return outputs

    results = benchmark(process_all_chunks)

    if backend == "mlx":
        import mlx.core as mx

        last_mx = next(o for o in reversed(results) if np.asarray(o.data).size > 0)
        assert isinstance(last_mx.data, mx.array), f"Expected mx.array, got {type(last_mx.data)}"
