import math
import typing
from dataclasses import field

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import (
    BaseProcessor,
    BaseStatefulProcessor,
    BaseTransformer,
    BaseTransformerUnit,
    CompositeProcessor,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .filterbankdesign import (
    FilterbankDesignSettings,
    FilterbankDesignTransformer,
)
from .kaiser import KaiserFilterSettings
from .sampler import SampleTriggerMessage
from .window import WindowSettings, WindowTransformer


class FBCCASettings(ez.Settings):
    """
    Settings for :obj:`FBCCATransformer`
    """

    time_dim: str
    """
    The time dim in the data array.
    """

    ch_dim: str
    """
    The channels dim in the data array.
    """

    filterbank_dim: str | None = None
    """
    The filter bank subband dim in the data array. If unspecified, method falls back to CCA
    None (default): the input has no subbands; just use CCA
    """

    harmonics: int = 5
    """
    The number of additional harmonics beyond the fundamental to use for the 'design' matrix.
    5 (default): Evaluate 5 harmonics of the base frequency.
    Many periodic signals are not pure sinusoids, and inclusion of higher harmonics can help evaluate the
    presence of signals with higher frequency harmonic content
    """

    freqs: typing.List[float] = field(default_factory=list)
    """
    Frequencies (in hz) to evaluate the presence of within the input signal.
    [] (default): an empty list; frequencies will be found within the input SampleMessages.
    AxisArrays have no good place to put this metadata, so specify frequencies here if only AxisArrays
    will be passed as input to the generator.  If the input has a `trigger` attr of type :obj:`SampleTriggerMessage`,
    the processor looks for the `freqs` attribute within that trigger for a list of frequencies to evaluate.
    This field is present in the :obj:`SSVEPSampleTriggerMessage` defined in ezmsg.tasks.ssvep from
    the ezmsg-tasks package.
    NOTE: Avoid frequencies that have line-noise (60 Hz/50 Hz) as a harmonic.
    """

    softmax_beta: float = 1.0
    """
    Beta parameter for softmax on output --> "probabilities".
    1.0 (default): Use the shifted softmax transformation to output 0-1 probabilities.
    If 0.0, the maximum singular value of the SVD for each design matrix is output
    """

    target_freq_dim: str = "target_freq"
    """
    Name for dim to put target frequency outputs on.
    'target_freq' (default)
    """

    max_int_time: float = 0.0
    """
    Maximum integration time (in seconds) to use for calculation.
    0 (default): Use all time provided for the calculation.
    Useful for artificially limiting the amount of data used for the CCA method to evaluate
    the necessary integration time for good decoding performance
    """


class FBCCATransformer(BaseTransformer[FBCCASettings, AxisArray, AxisArray]):
    """
    A canonical-correlation (CCA) signal decoder for detection of periodic activity in multi-channel timeseries
    recordings. It is particularly useful for detecting the presence of steady-state evoked responses in multi-channel
    EEG data. Please see Lin et. al. 2007 for a description on the use of CCA to detect the presence of SSVEP in EEG
    data.
    This implementation also includes the "Filterbank" extension of the CCA decoding approach which utilizes a
    filterbank to decompose input multi-channel EEG data into several frequency sub-bands; each of which is analyzed
    with CCA, then combined using a weighted sum; allowing CCA to more readily identify harmonic content in EEG data.
    Read more about this approach in Chen et. al. 2015.

    ## Further reading:
    * [Lin et. al. 2007](https://ieeexplore.ieee.org/document/4015614)
    * [Nakanishi et. al. 2015](https://doi.org/10.1371%2Fjournal.pone.0140703)
    * [Chen et. al. 2015](http://dx.doi.org/10.1088/1741-2560/12/4/046008)
    """

    def _process(self, message: AxisArray) -> AxisArray:
        """
        Input: AxisArray with at least a time_dim, and ch_dim
        Output: AxisArray with time_dim, ch_dim, (and filterbank_dim if specified)
            collapsed, with a new 'target_freq' dim of length 'freqs'
        """

        test_freqs: list[float] = self.settings.freqs
        trigger = message.attrs.get("trigger", None)
        if isinstance(trigger, SampleTriggerMessage):
            if len(test_freqs) == 0:
                test_freqs = getattr(trigger, "freqs", [])

        if len(test_freqs) == 0:
            raise ValueError("no frequencies to test")

        time_dim_idx = message.get_axis_idx(self.settings.time_dim)
        ch_dim_idx = message.get_axis_idx(self.settings.ch_dim)

        filterbank_dim_idx = None
        if self.settings.filterbank_dim is not None:
            filterbank_dim_idx = message.get_axis_idx(self.settings.filterbank_dim)

        # Move (filterbank_dim), time, ch to end of array
        rm_dims = [self.settings.time_dim, self.settings.ch_dim]
        if self.settings.filterbank_dim is not None:
            rm_dims = [self.settings.filterbank_dim] + rm_dims
        new_order = [i for i, dim in enumerate(message.dims) if dim not in rm_dims]
        if filterbank_dim_idx is not None:
            new_order.append(filterbank_dim_idx)
        new_order.extend([time_dim_idx, ch_dim_idx])
        out_dims = [message.dims[i] for i in new_order if message.dims[i] not in rm_dims]
        data_arr = message.data.transpose(new_order)

        # Add a singleton dim for filterbank dim if we don't have one
        if filterbank_dim_idx is None:
            data_arr = data_arr[..., None, :, :]
            filterbank_dim_idx = data_arr.ndim - 3

        # data_arr is now (..., filterbank, time, ch)
        # Get output shape for remaining dims and reshape data_arr for iterative processing
        out_shape = list(data_arr.shape[:-3])
        data_arr = data_arr.reshape([math.prod(out_shape), *data_arr.shape[-3:]])

        # Create output dims and axes with added target_freq_dim
        out_shape.append(len(test_freqs))
        out_dims.append(self.settings.target_freq_dim)
        out_axes = {
            axis_name: axis
            for axis_name, axis in message.axes.items()
            if axis_name not in rm_dims
            and not (isinstance(axis, AxisArray.CoordinateAxis) and any(d in rm_dims for d in axis.dims))
        }
        out_axes[self.settings.target_freq_dim] = AxisArray.CoordinateAxis(
            np.array(test_freqs), [self.settings.target_freq_dim]
        )

        if message.data.size == 0:
            out_data = message.data.reshape(out_shape)
            output = replace(message, data=out_data, dims=out_dims, axes=out_axes)
            return output

        # Get time axis
        t_ax_info = message.ax(self.settings.time_dim)
        t = t_ax_info.values
        t -= t[0]
        max_samp = len(t)
        if self.settings.max_int_time > 0:
            max_samp = int(abs(t_ax_info.values - self.settings.max_int_time).argmin())
        t = t[:max_samp]

        calc_output = np.zeros((*data_arr.shape[:-2], len(test_freqs)))

        for test_freq_idx, test_freq in enumerate(test_freqs):
            # Create the design matrix of base frequency and requested harmonics
            Y = np.column_stack(
                [
                    fn(2.0 * np.pi * k * test_freq * t)
                    for k in range(1, self.settings.harmonics + 1)
                    for fn in (np.sin, np.cos)
                ]
            )

            for test_idx, arr in enumerate(data_arr):  # iterate over first dim; arr is (filterbank x time x ch)
                for band_idx, band in enumerate(arr):  # iterate over second dim: arr is (time x ch)
                    calc_output[test_idx, band_idx, test_freq_idx] = cca_rho_max(band[:max_samp, ...], Y)

        # Combine per-subband canonical correlations using a weighted sum
        # https://iopscience.iop.org/article/10.1088/1741-2560/12/4/046008
        freq_weights = (np.arange(1, calc_output.shape[1] + 1) ** -1.25) + 0.25
        calc_output = ((calc_output**2) * freq_weights[None, :, None]).sum(axis=1)

        if self.settings.softmax_beta != 0:
            calc_output = calc_softmax(calc_output, axis=-1, beta=self.settings.softmax_beta)

        output = replace(
            message,
            data=calc_output.reshape(out_shape),
            dims=out_dims,
            axes=out_axes,
        )

        return output


class FBCCA(BaseTransformerUnit[FBCCASettings, AxisArray, AxisArray, FBCCATransformer]):
    SETTINGS = FBCCASettings


class StreamingFBCCASettings(FBCCASettings):
    """
    Perform rolling/streaming FBCCA on incoming EEG.
    Decomposes the input multi-channel timeseries data into multiple sub-bands using a FilterbankDesign Transformer,
    then accumulates data using Window into short-time observations for analysis using an FBCCA Transformer.
    """

    window_dur: float = 4.0  # sec
    window_shift: float = 0.5  # sec
    window_dim: str = "fbcca_window"
    filter_bw: float = 7.0  # Hz
    filter_low: float = 7.0  # Hz
    trans_bw: float = 2.0  # Hz
    ripple_db: float = 20.0  # dB
    subbands: int = 12


class StreamingFBCCATransformer(CompositeProcessor[StreamingFBCCASettings, AxisArray, AxisArray]):
    @staticmethod
    def _initialize_processors(
        settings: StreamingFBCCASettings,
    ) -> dict[str, BaseProcessor | BaseStatefulProcessor]:
        pipeline = {}

        if settings.filterbank_dim is not None:
            cut_freqs = (np.arange(settings.subbands + 1) * settings.filter_bw) + settings.filter_low
            filters = [
                KaiserFilterSettings(
                    axis=settings.time_dim,
                    cutoff=(c - settings.trans_bw, cut_freqs[-1]),
                    ripple=settings.ripple_db,
                    width=settings.trans_bw,
                    pass_zero=False,
                )
                for c in cut_freqs[:-1]
            ]

            pipeline["filterbank"] = FilterbankDesignTransformer(
                FilterbankDesignSettings(filters=filters, new_axis=settings.filterbank_dim)
            )

        pipeline["window"] = WindowTransformer(
            WindowSettings(
                axis=settings.time_dim,
                newaxis=settings.window_dim,
                window_dur=settings.window_dur,
                window_shift=settings.window_shift,
                zero_pad_until="shift",
            )
        )

        pipeline["fbcca"] = FBCCATransformer(settings)

        return pipeline


class StreamingFBCCA(BaseTransformerUnit[StreamingFBCCASettings, AxisArray, AxisArray, StreamingFBCCATransformer]):
    SETTINGS = StreamingFBCCASettings


def cca_rho_max(X: np.ndarray, Y: np.ndarray) -> float:
    """
    X: (n_time, n_ch)
    Y: (n_time, n_ref)  # design matrix for one frequency
    returns: largest canonical correlation in [0,1]
    """
    # Center columns
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # Drop any zero-variance columns to avoid rank issues
    Xc = Xc[:, Xc.std(axis=0) > 1e-12]
    Yc = Yc[:, Yc.std(axis=0) > 1e-12]
    if Xc.size == 0 or Yc.size == 0:
        return 0.0

    # Orthonormal bases
    Qx, _ = np.linalg.qr(Xc, mode="reduced")  # (n_time, r_x)
    Qy, _ = np.linalg.qr(Yc, mode="reduced")  # (n_time, r_y)

    # Canonical correlations are the singular values of Qx^T Qy
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return float(s[0]) if s.size else 0.0


def calc_softmax(cv: np.ndarray, axis: int, beta: float = 1.0):
    # Calculate softmax with shifting to avoid overflow
    # (https://doi.org/10.1093/imanum/draa038)
    cv = cv - cv.max(axis=axis, keepdims=True)
    cv = np.exp(beta * cv)
    cv = cv / np.sum(cv, axis=axis, keepdims=True)
    return cv
