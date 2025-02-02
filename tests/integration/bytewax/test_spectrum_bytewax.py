import pytest
import numpy as np
import scipy.fft as sp_fft
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.chunker import array_chunker

from ezmsg.sigproc.spectrum import (
    SpectrumTransformer,
    SpectralTransform,
    SpectralOutput,
    WindowFunction,
)

try:
    import bytewax.operators as op
    from bytewax.dataflow import Dataflow
    from bytewax.testing import TestingSource, TestingSink, run_main

    b_bytewax = True
except ImportError:
    b_bytewax = False


@pytest.fixture
def axarr_list() -> list[AxisArray]:
    data = np.arange(3000 * 4 * 5).reshape(3000, 4, 5)
    gen = array_chunker(data, chunk_len=1000, axis=0, fs=1000.0, tzero=0.0)
    return list(gen)


@pytest.mark.skipif(not b_bytewax, reason="Bytewax not installed")
def test_spectrum_bytewax(axarr_list):
    nfft = 1024
    result = []
    flow = Dataflow("test_spectrum_bytewax")
    input_stream = op.input("input", flow, TestingSource(axarr_list))
    keyed_stream = op.key_on("key_stream", input_stream, lambda _: "ALL")
    # perform spectral transform -- use parameters that mirror vanilla scipy rfft.
    spectrum_stream = op.stateful_map(
        "spectrum",
        keyed_stream,
        SpectrumTransformer(
            axis="time",
            window=WindowFunction.NONE,
            transform=SpectralTransform.REAL,
            output=SpectralOutput.POSITIVE,
            norm="backward",
            do_fftshift=False,
            out_axis="freq",
            nfft=nfft,
        ).stateful_op,
    )
    op.output("out", spectrum_stream, TestingSink(result))
    run_main(flow)

    # Collect the output
    out_data = AxisArray.concatenate(*[_[1] for _ in result], dim="win", axis=None).data

    # Calculate expected
    sp_res = [sp_fft.rfft(_.data, n=nfft, axis=0).real for _ in axarr_list]
    sp_res = np.stack(sp_res, axis=0)

    # Compare output to expected
    assert np.allclose(out_data, sp_res)


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
