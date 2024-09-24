import copy
from pathlib import Path

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.affinetransform import affine_transform, common_rereference

from util import assert_messages_equal


def test_affine_generator():
    n_times = 13
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    axis_arr_in = AxisArray(in_dat, dims=["time", "ch"])

    backup = [copy.deepcopy(axis_arr_in)]

    gen = affine_transform(weights=np.eye(n_chans), axis="ch")
    ax_arr_out = gen.send(axis_arr_in)
    assert ax_arr_out.data.shape == in_dat.shape
    assert np.allclose(ax_arr_out.data, in_dat)
    assert not np.may_share_memory(ax_arr_out.data, in_dat)

    assert_messages_equal([axis_arr_in], backup)

    # Test with weights from a CSV file.
    csv_path = Path(__file__).parent / "resources" / "xform.csv"
    weights = np.loadtxt(csv_path, delimiter=",")
    expected_out = in_dat @ weights.T
    # Same result: expected_out = np.vstack([(step[None, :] * weights).sum(axis=1) for step in in_dat])

    gen = affine_transform(weights=csv_path, axis="ch", right_multiply=False)
    ax_arr_out = gen.send(axis_arr_in)
    assert np.allclose(ax_arr_out.data, expected_out)

    # Try again as str, not Path
    gen = affine_transform(weights=str(csv_path), axis="ch", right_multiply=False)
    ax_arr_out = gen.send(axis_arr_in)
    assert np.allclose(ax_arr_out.data, expected_out)

    # Try again as direct ndarray
    gen = affine_transform(weights=weights, axis="ch", right_multiply=False)
    ax_arr_out = gen.send(axis_arr_in)
    assert np.allclose(ax_arr_out.data, expected_out)

    # One more time, but we pre-transpose the weights and do not override right_multiply
    gen = affine_transform(weights=weights.T, axis="ch", right_multiply=True)
    ax_arr_out = gen.send(axis_arr_in)
    assert np.allclose(ax_arr_out.data, expected_out)


def test_affine_passthrough():
    n_times = 13
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    axis_arr_in = AxisArray(in_dat, dims=["time", "ch"])

    backup = [copy.deepcopy(axis_arr_in)]

    gen = affine_transform(weights="passthrough", axis="does not matter")
    ax_arr_out = gen.send(axis_arr_in)
    assert ax_arr_out.data is in_dat  # This is not desirable in ezmsg pipeline but fine for the generator
    assert_messages_equal([ax_arr_out], backup)


def test_common_rereference():
    n_times = 300
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    axis_arr_in = AxisArray(in_dat, dims=["time", "ch"])

    backup = [copy.deepcopy(axis_arr_in)]

    gen = common_rereference(mode="mean", axis="ch", include_current=True)
    axis_arr_out = gen.send(axis_arr_in)
    assert np.array_equal(
        axis_arr_out.data,
        axis_arr_in.data - np.mean(axis_arr_in.data, axis=1, keepdims=True),
    )

    assert_messages_equal([axis_arr_in], backup)

    # Use a slow deliberate way of calculating the CAR uniquely for each channel, excluding itself.
    #  common_rereference uses a faster way of doing this, but we test against something intuitive.
    expected_out = []
    for ch_ix in range(n_chans):
        idx = np.arange(n_chans)
        idx = np.hstack((idx[:ch_ix], idx[ch_ix + 1 :]))
        expected_out.append(
            axis_arr_in.data[..., ch_ix] - np.mean(axis_arr_in.data[..., idx], axis=1)
        )
    expected_out = np.stack(expected_out).T

    gen = common_rereference(mode="mean", axis="ch", include_current=False)
    axis_arr_out = gen.send(axis_arr_in)  # 41 us
    assert np.allclose(axis_arr_out.data, expected_out)

    # Instead of CAR, we could use affine_transform with weights that reproduce CAR.
    # However, this method is 30x slower than above. (Actual difference varies depending on data shape).
    if False:
        weights = -np.ones((n_chans, n_chans)) / (n_chans - 1)
        np.fill_diagonal(weights, 1)
        gen = affine_transform(weights=weights, axis="ch")
        axis_arr_out = gen.send(axis_arr_in)
        assert np.allclose(axis_arr_out.data, expected_out)


def test_car_passthrough():
    n_times = 300
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    axis_arr_in = AxisArray(in_dat, dims=["time", "ch"])

    gen = common_rereference(mode="passthrough")
    axis_arr_out = gen.send(axis_arr_in)
    assert np.array_equal(axis_arr_out.data, in_dat)
    assert not np.may_share_memory(axis_arr_out.data, in_dat)
