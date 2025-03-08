import numpy as np
import numpy.typing as npt
import scipy.signal
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from ezmsg.util.messages.util import replace

from .base import processor_state, BaseStatefulTransformer


class AdaptiveLatticeNotchFilterSettings(ez.Settings):
    """Settings for the Adaptive Lattice Notch Filter."""

    gamma: float = 0.995  # Pole-zero contraction factor
    mu: float = 0.99  # Smoothing factor
    eta: float = 0.99  # Forgetting factor
    axis: str = "time"  # Axis to apply filter to
    init_notch_freq: float | None = (
        None  # Initial notch frequency. Should be < nyquist.
    )
    chunkwise: bool = True  # Speed up processing by updating the target freq once per chunk only.


@processor_state
class AdaptiveLatticeNotchFilterState:
    """State for the Adaptive Lattice Notch Filter."""

    s_history: npt.NDArray | None = None
    """Historical `s` values for the adaptive filter."""

    p: npt.NDArray | None = None
    """Accumulated product for reflection coefficient update"""

    q: npt.NDArray | None = None
    """Accumulated product for reflection coefficient update"""

    k1: npt.NDArray | None = None
    """Reflection coefficient"""

    freq_template: CoordinateAxis | None = None
    """Template for the frequency axis on the output"""

    zi: npt.NDArray | None = None
    """Initial conditions for the filter, updated after every chunk"""


class AdaptiveLatticeNotchFilterTransformer(
    BaseStatefulTransformer[
        AdaptiveLatticeNotchFilterSettings,
        AxisArray,
        AxisArray,
        AdaptiveLatticeNotchFilterState,
    ]
):
    """
    Adaptive Lattice Notch Filter implementation as a stateful transformer.

    https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/1475-925X-13-170

    The filter automatically tracks and removes frequency components from the input signal.
    It outputs the estimated frequency (in Hz) and the filtered sample.
    """

    def _hash_message(self, message: AxisArray) -> int:
        ax_idx = message.get_axis_idx(self.settings.axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        return hash((message.key, message.axes[self.settings.axis].gain, sample_shape))

    def _reset_state(self, message: AxisArray) -> None:
        ax_idx = message.get_axis_idx(self.settings.axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]

        fs = 1 / message.axes[self.settings.axis].gain
        init_f = (
            self.settings.init_notch_freq
            if self.settings.init_notch_freq is not None
            else 0.07178314656435313 * fs
        )
        init_omega = init_f * (2 * np.pi) / fs
        init_k1 = -np.cos(init_omega)

        """Reset filter state to initial values."""
        self._state = AdaptiveLatticeNotchFilterState()
        self._state.s_history = np.zeros((2,) + sample_shape, dtype=float)
        self._state.p = np.zeros(sample_shape, dtype=float)
        self._state.q = np.zeros(sample_shape, dtype=float)
        self._state.k1 = init_k1 + np.zeros(sample_shape, dtype=float)
        self._state.freq_template = CoordinateAxis(
            data=np.zeros((0,) + sample_shape, dtype=float),
            dims=[self.settings.axis]
            + message.dims[:ax_idx]
            + message.dims[ax_idx + 1 :],
            unit="Hz",
        )

        # Initialize the initial conditions for the filter
        self._state.zi = np.zeros((2, np.prod(sample_shape)), dtype=float)
        # Note: we could calculate it properly, but as long as we are initializing s_history with zeros,
        #  it will always be zero.
        # a = [1, init_k1 * (1 + self.settings.gamma), self.settings.gamma]
        # b = [1]
        # s = np.reshape(self._state.s_history, (2, -1))
        # for feat_ix in range(np.prod(sample_shape)):
        #     self._state.zi[:, feat_ix] = scipy.signal.lfiltic(b, a, s[::-1, feat_ix], x=None)

    def _process(self, message: AxisArray) -> AxisArray:
        x_data = message.data
        ax_idx = message.get_axis_idx(self.settings.axis)
        if message.dims[0] != self.settings.axis:
            x_data = np.moveaxis(x_data, ax_idx, 0)

        # Access settings once
        gamma = self.settings.gamma
        eta = self.settings.eta
        mu = self.settings.mu
        fs = 1 / message.axes[self.settings.axis].gain

        # Pre-compute constants
        one_minus_eta = 1 - eta
        one_minus_mu = 1 - mu
        gamma_plus_1 = 1 + gamma
        omega_scale = fs / (2 * np.pi)

        if True:  # self.settings.chunkwise:
            # TODO: Time should be moved to -1th axis
            # Update the target frequency once per chunk
            # For the lattice filter with constant k1:
            # s_n = x_n - k1*(1+gamma)*s_n_1 - gamma*s_n_2
            # This is equivalent to an IIR filter with b=1, a=[1, k1*(1+gamma), gamma]

            # For the output filter:
            # y_n = s_n + 2*k1*s_n_1 + s_n_2
            # We can treat this as a direct-form FIR filter applied to s_out

            _s = self._state.s_history.reshape((2, -1))
            _x = x_data.reshape((x_data.shape[0], -1))
            s_n = np.zeros_like(_x)
            for ix, k in enumerate(self._state.k1.flatten()):
                a_s = [1, k * gamma_plus_1, gamma]
                s_n[:, ix], self._state.zi[:, ix] = scipy.signal.lfilter([1], a_s, _x[:, ix], zi=self._state.zi[:, ix])
                b_y = [1, 2 * k, 1]
                y_out[:, ix] = scipy.signal.lfilter(b_y, [1], s_n[:, ix])
            self._state.s_history = s_n[-2:].reshape((2,) + x_data.shape[1:])

            # TODO: Fixup below

            # Calculate frequency from final k1 value
            omega_n = np.arccos(-self._state.k1)
            freq = omega_n * omega_scale
            freq_out = np.full_like(x_data, freq)

            # Update state with the last values
            self._state.s_history[:] = s_out[-2:]

            # Calculate updates to p, q, and k1 based on the final values
            # This would happen once per chunk instead of per sample
            p = eta * p + one_minus_eta * (s_out[-2] * (s_out[-1] + s_out[-3] if len(s_out) > 2 else s_n_2))
            q = eta * q + one_minus_eta * (2 * (s_out[-2] * s_out[-2]))

            # Update reflection coefficient
            new_k1 = -p / (q + 1e-8)
            new_k1 = np.clip(new_k1, -1, 1)
            k1 = mu * k1 + one_minus_mu * new_k1

        else:

            # Perform filtering, sample-by-sample
            y_out = np.zeros_like(x_data)
            freq_out = np.zeros_like(x_data)
            for sample_ix, x_n in enumerate(x_data):
                s_n_1 = self._state.s_history[-1]
                s_n_2 = self._state.s_history[-2]

                s_n = x_n - self._state.k1 * gamma_plus_1 * s_n_1 - gamma * s_n_2
                y_out[sample_ix] = s_n + 2 * self._state.k1 * s_n_1 + s_n_2

                # Update filter parameters
                self._state.p = eta * self._state.p + one_minus_eta * (
                    s_n_1 * (s_n + s_n_2)
                )
                self._state.q = eta * self._state.q + one_minus_eta * (2 * (s_n_1 * s_n_1))

                # Update reflection coefficient
                new_k1 = -self._state.p / (self._state.q + 1e-8)  # Avoid division by zero
                new_k1 = np.clip(new_k1, -1, 1)  # Clip to prevent instability
                self._state.k1 = mu * self._state.k1 + one_minus_mu * new_k1  # Smoothed

                # Compute normalized angular frequency using equation 13 from the paper
                omega_n = np.arccos(-self._state.k1)
                freq_out[sample_ix] = omega_n * omega_scale  # As Hz

                # Update for next iteration
                self._state.s_history[-2] = s_n_1
                self._state.s_history[-1] = s_n

        return replace(
            message,
            data=y_out,
            axes={
                **message.axes,
                "freq": replace(self._state.freq_template, data=freq_out),
            },
        )
