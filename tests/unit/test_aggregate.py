import copy
from functools import partial

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from frozendict import frozendict

from ezmsg.sigproc.aggregate import (
    AggregateSettings,
    AggregateTransformer,
    AggregationFunction,
    RangedAggregateSettings,
    RangedAggregateTransformer,
)
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

    xformer = RangedAggregateTransformer(RangedAggregateSettings(axis=targ_ax, bands=bands, operation=agg_func))
    out_msgs = [xformer(_) for _ in in_msgs]

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
        [agg_func(data[..., np.logical_and(targ_ax_vec >= start, targ_ax_vec <= stop)]) for (start, stop) in bands],
        axis=-1,
    )
    received_data = AxisArray.concatenate(*out_msgs, dim="time").data
    assert np.allclose(received_data, expected_data)


@pytest.mark.parametrize("agg_func", [AggregationFunction.ARGMIN, AggregationFunction.ARGMAX])
def test_arg_aggregate(agg_func: AggregationFunction):
    bands = [(5.0, 20.0), (30.0, 50.0)]
    in_msgs = [_ for _ in get_msg_gen()]
    xformer = RangedAggregateTransformer(RangedAggregateSettings(axis="freq", bands=bands, operation=agg_func))
    out_msgs = [xformer(_) for _ in in_msgs]

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
    xformer = RangedAggregateTransformer(
        RangedAggregateSettings(axis="freq", bands=bands, operation=AggregationFunction.TRAPEZOID)
    )
    out_msgs = [xformer(_) for _ in in_msgs]

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

    xformer = RangedAggregateTransformer(
        RangedAggregateSettings(
            axis="freq",
            bands=[(5.0, 20.0), (30.0, 50.0)],
            operation=AggregationFunction.MEAN,
        )
    )

    out_msgs1 = [xformer(_) for _ in in_msgs1]
    print(len(out_msgs1))
    out_msgs2 = [xformer(_) for _ in in_msgs2]
    print(len(out_msgs2))


# ============== Tests for AggregateTransformer ==============


def get_simple_msg(n_times=10, n_chans=5, n_freqs=8, fs=100.0):
    """Create a simple AxisArray message for testing AggregateTransformer."""
    data = np.arange(n_times * n_chans * n_freqs, dtype=float).reshape(n_times, n_chans, n_freqs)
    return AxisArray(
        data=data,
        dims=["time", "ch", "freq"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=fs, offset=0.0),
                "ch": AxisArray.CoordinateAxis(
                    data=np.array([f"ch{i}" for i in range(n_chans)]),
                    dims=["ch"],
                ),
                "freq": AxisArray.LinearAxis(gain=2.0, offset=1.0, unit="Hz"),
            }
        ),
    )


@pytest.mark.parametrize(
    "operation",
    [
        AggregationFunction.MEAN,
        AggregationFunction.SUM,
        AggregationFunction.MAX,
        AggregationFunction.MIN,
        AggregationFunction.STD,
        AggregationFunction.MEDIAN,
    ],
)
def test_aggregate_transformer_basic(operation: AggregationFunction):
    """Test AggregateTransformer with basic aggregation operations."""
    msg_in = get_simple_msg()
    backup = copy.deepcopy(msg_in)

    transformer = AggregateTransformer(AggregateSettings(axis="freq", operation=operation))
    msg_out = transformer(msg_in)

    # Verify input wasn't modified
    assert_messages_equal([msg_in], [backup])

    # Verify output type
    assert isinstance(msg_out, AxisArray)

    # Verify axis was removed
    assert "freq" not in msg_out.dims
    assert "freq" not in msg_out.axes
    assert msg_out.dims == ["time", "ch"]

    # Verify output shape
    assert msg_out.data.shape == (10, 5)

    # Verify data correctness
    np_func = getattr(np, operation.value)
    expected = np_func(msg_in.data, axis=2)
    assert np.allclose(msg_out.data, expected)


@pytest.mark.parametrize("axis", ["time", "ch", "freq"])
def test_aggregate_transformer_different_axes(axis: str):
    """Test AggregateTransformer can aggregate along different axes."""
    msg_in = get_simple_msg(n_times=10, n_chans=5, n_freqs=8)

    transformer = AggregateTransformer(AggregateSettings(axis=axis, operation=AggregationFunction.MEAN))
    msg_out = transformer(msg_in)

    # Verify the specified axis was removed
    assert axis not in msg_out.dims
    assert axis not in msg_out.axes

    # Verify remaining dims
    expected_dims = [d for d in ["time", "ch", "freq"] if d != axis]
    assert msg_out.dims == expected_dims

    # Verify shape
    axis_idx = msg_in.get_axis_idx(axis)
    expected_shape = list(msg_in.data.shape)
    expected_shape.pop(axis_idx)
    assert msg_out.data.shape == tuple(expected_shape)

    # Verify data
    expected = np.mean(msg_in.data, axis=axis_idx)
    assert np.allclose(msg_out.data, expected)


def test_aggregate_transformer_none_raises():
    """Test that AggregationFunction.NONE raises an error."""
    msg_in = get_simple_msg()

    transformer = AggregateTransformer(AggregateSettings(axis="freq", operation=AggregationFunction.NONE))

    with pytest.raises(ValueError, match="NONE is not supported"):
        transformer(msg_in)


@pytest.mark.parametrize(
    "operation",
    [
        AggregationFunction.NANMEAN,
        AggregationFunction.NANSUM,
        AggregationFunction.NANMAX,
        AggregationFunction.NANMIN,
        AggregationFunction.NANSTD,
        AggregationFunction.NANMEDIAN,
    ],
)
def test_aggregate_transformer_nan_operations(operation: AggregationFunction):
    """Test AggregateTransformer with NaN-aware operations."""
    msg_in = get_simple_msg()
    # Introduce some NaN values
    msg_in.data[0, 0, 0] = np.nan
    msg_in.data[5, 2, 3] = np.nan

    transformer = AggregateTransformer(AggregateSettings(axis="freq", operation=operation))
    msg_out = transformer(msg_in)

    # Verify output doesn't have NaN where nan-operations should have handled it
    np_func = getattr(np, operation.value)
    expected = np_func(msg_in.data, axis=2)
    assert np.allclose(msg_out.data, expected, equal_nan=True)


@pytest.mark.parametrize("operation", [AggregationFunction.ARGMIN, AggregationFunction.ARGMAX])
def test_aggregate_transformer_argminmax(operation: AggregationFunction):
    """Test AggregateTransformer with argmin/argmax operations."""
    msg_in = get_simple_msg()

    transformer = AggregateTransformer(AggregateSettings(axis="freq", operation=operation))
    msg_out = transformer(msg_in)

    # Verify output shape (axis removed)
    assert msg_out.data.shape == (10, 5)
    assert "freq" not in msg_out.dims

    # Verify data correctness (returns indices)
    np_func = getattr(np, operation.value)
    expected = np_func(msg_in.data, axis=2)
    assert np.array_equal(msg_out.data, expected)


def test_aggregate_transformer_trapezoid():
    """Test AggregateTransformer with trapezoid integration."""
    msg_in = get_simple_msg(n_times=5, n_chans=3, n_freqs=10)

    transformer = AggregateTransformer(AggregateSettings(axis="freq", operation=AggregationFunction.TRAPEZOID))
    msg_out = transformer(msg_in)

    # Verify output shape
    assert msg_out.data.shape == (5, 3)
    assert "freq" not in msg_out.dims

    # Calculate expected result using axis coordinates
    freq_axis = msg_in.axes["freq"]
    x = freq_axis.value(np.arange(msg_in.data.shape[2]))
    expected = np.trapezoid(msg_in.data, x=x, axis=2)

    assert np.allclose(msg_out.data, expected)


def test_aggregate_transformer_trapezoid_coordinate_axis():
    """Test trapezoid integration with CoordinateAxis."""
    n_times, n_chans, n_freqs = 5, 3, 10
    data = np.arange(n_times * n_chans * n_freqs, dtype=float).reshape(n_times, n_chans, n_freqs)
    freq_values = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0])
    msg_in = AxisArray(
        data=data,
        dims=["time", "ch", "freq"],
        axes=frozendict(
            {
                "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
                "freq": AxisArray.CoordinateAxis(data=freq_values, dims=["freq"], unit="Hz"),
            }
        ),
    )

    transformer = AggregateTransformer(AggregateSettings(axis="freq", operation=AggregationFunction.TRAPEZOID))
    msg_out = transformer(msg_in)

    # Calculate expected using the coordinate values
    expected = np.trapezoid(msg_in.data, x=freq_values, axis=2)
    assert np.allclose(msg_out.data, expected)


def test_aggregate_transformer_preserves_other_axes():
    """Test that non-aggregated axes are preserved correctly."""
    msg_in = get_simple_msg()

    transformer = AggregateTransformer(AggregateSettings(axis="freq", operation=AggregationFunction.MEAN))
    msg_out = transformer(msg_in)

    # Verify time axis preserved
    assert "time" in msg_out.axes
    assert msg_out.axes["time"] == msg_in.axes["time"]

    # Verify ch axis preserved
    assert "ch" in msg_out.axes
    ch_ax_in = msg_in.axes["ch"]
    ch_ax_out = msg_out.axes["ch"]
    assert np.array_equal(ch_ax_out.data, ch_ax_in.data)


def test_aggregate_transformer_multiple_calls():
    """Test that transformer works correctly with multiple calls."""
    transformer = AggregateTransformer(AggregateSettings(axis="freq", operation=AggregationFunction.SUM))

    for i in range(3):
        msg_in = get_simple_msg()
        msg_in.data = msg_in.data + i * 1000  # Different data each time

        msg_out = transformer(msg_in)

        expected = np.sum(msg_in.data, axis=2)
        assert np.allclose(msg_out.data, expected)
