from functools import partial

import numpy as np
import pytest
from frozendict import frozendict
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.aggregate import ranged_aggregate, AggregationFunction

from tests.helpers.util import assert_messages_equal


def get_msg_gen(n_chans=20, n_freqs=100, data_dur=30.0, fs=1024.0, key=""):
    n_samples = int(data_dur * fs)
    data = np.arange(n_samples * n_chans * n_freqs).reshape(n_samples, n_chans, n_freqs)
    n_msgs = int(data_dur / 2)

    def msg_generator():
        offset = 0
        for arr in np.array_split(data, n_samples // n_msgs):
            msg = AxisArray(
                data=arr,
                dims=["time", "ch", "freq"],
                axes=frozendict(
                    {
                        "time": AxisArray.TimeAxis(fs=fs, offset=offset),
                        "freq": AxisArray.LinearAxis(gain=1.0, offset=0.0, unit="Hz"),
                    }
                ),
                key=key,
            )
            offset += arr.shape[0] / fs
            yield msg

    return msg_generator()


@pytest.mark.parametrize(
    "agg_func",
    [
        AggregationFunction.MEAN,
        AggregationFunction.MEDIAN,
        AggregationFunction.STD,
        AggregationFunction.SUM,
    ],
)
def test_aggregate(agg_func: AggregationFunction):
    bands = [(5.0, 20.0), (30.0, 50.0)]
    targ_ax = "freq"

    in_msgs = [_ for _ in get_msg_gen()]

    # Grab a deepcopy backup of the inputs so we can check the inputs didn't change
    #  while being processed.
    import copy

    backup = [copy.deepcopy(_) for _ in in_msgs]

    gen = ranged_aggregate(axis=targ_ax, bands=bands, operation=agg_func)
    out_msgs = [gen.send(_) for _ in in_msgs]

    assert_messages_equal(in_msgs, backup)

    assert all([type(_) is AxisArray for _ in out_msgs])

    # Check output axis
    for out_msg in out_msgs:
        ax = out_msg.axes[targ_ax]
        assert np.array_equal(ax.data, np.array([np.mean(band) for band in bands]))
        assert ax.unit == in_msgs[0].axes[targ_ax].unit

    # Check data
    data = AxisArray.concatenate(*in_msgs, dim="time").data
    targ_ax = in_msgs[0].axes[targ_ax]
    targ_ax_vec = targ_ax.value(np.arange(data.shape[-1]))
    agg_func = {
        AggregationFunction.MEAN: partial(np.mean, axis=-1, keepdims=True),
        AggregationFunction.MEDIAN: partial(np.median, axis=-1, keepdims=True),
        AggregationFunction.STD: partial(np.std, axis=-1, keepdims=True),
        AggregationFunction.SUM: partial(np.sum, axis=-1, keepdims=True),
    }[agg_func]
    expected_data = np.concatenate(
        [
            agg_func(
                data[..., np.logical_and(targ_ax_vec >= start, targ_ax_vec <= stop)]
            )
            for (start, stop) in bands
        ],
        axis=-1,
    )
    received_data = AxisArray.concatenate(*out_msgs, dim="time").data
    assert np.allclose(received_data, expected_data)


@pytest.mark.parametrize(
    "agg_func", [AggregationFunction.ARGMIN, AggregationFunction.ARGMAX]
)
def test_arg_aggregate(agg_func: AggregationFunction):
    bands = [(5.0, 20.0), (30.0, 50.0)]
    in_msgs = [_ for _ in get_msg_gen()]
    gen = ranged_aggregate(axis="freq", bands=bands, operation=agg_func)
    out_msgs = [gen.send(_) for _ in in_msgs]

    if agg_func == AggregationFunction.ARGMIN:
        expected_vals = np.array([np.min(_) for _ in bands])
    else:
        expected_vals = np.array([np.max(_) for _ in bands])
    out_dat = AxisArray.concatenate(*out_msgs, dim="time").data
    expected_dat = np.zeros(out_dat.shape[:-1] + (1,)) + expected_vals[None, None, :]
    assert np.array_equal(out_dat, expected_dat)


def test_trapezoid():
    bands = [(5.0, 20.0), (30.0, 50.0)]
    in_msgs = [_ for _ in get_msg_gen()]
    gen = ranged_aggregate(
        axis="freq", bands=bands, operation=AggregationFunction.TRAPEZOID
    )
    out_msgs = [gen.send(_) for _ in in_msgs]

    out_dat = AxisArray.concatenate(*out_msgs, dim="time").data

    # Calculate expected data using trapezoidal integration
    in_data = AxisArray.concatenate(*in_msgs, dim="time").data
    targ_ax = in_msgs[0].axes["freq"]
    targ_ax_vec = targ_ax.value(np.arange(in_data.shape[-1]))
    expected = []
    for start, stop in bands:
        inds = np.logical_and(targ_ax_vec >= start, targ_ax_vec <= stop)
        expected.append(np.trapezoid(in_data[..., inds], x=targ_ax_vec[inds], axis=-1))
    expected = np.stack(expected, axis=-1)

    assert out_dat.shape == expected.shape
    assert np.allclose(out_dat, expected)


@pytest.mark.parametrize("change_ax", ["ch", "freq"])
def test_aggregate_handle_change(change_ax: str):
    """
    If ranged_aggregate couldn't handle incoming changes, then
    change_ax being 'ch' should work while 'freq' should fail.
    """
    in_msgs1 = [_ for _ in get_msg_gen(n_chans=20, n_freqs=100)]
    in_msgs2 = [
        _
        for _ in get_msg_gen(
            n_chans=17 if change_ax == "ch" else 20,
            n_freqs=70 if change_ax == "freq" else 100,
        )
    ]

    gen = ranged_aggregate(
        axis="freq",
        bands=[(5.0, 20.0), (30.0, 50.0)],
        operation=AggregationFunction.MEAN,
    )

    out_msgs1 = [gen.send(_) for _ in in_msgs1]
    print(len(out_msgs1))
    out_msgs2 = [gen.send(_) for _ in in_msgs2]
    print(len(out_msgs2))
