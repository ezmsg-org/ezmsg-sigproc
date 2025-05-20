import copy
from pathlib import Path

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.sigproc.affinetransform import affine_transform, common_rereference

from tests.helpers.util import assert_messages_equal


def test_affine_generator():
    n_times = 13
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(
        data=in_dat,
        dims=["time", "ch"],
        axes={
            "ch": AxisArray.CoordinateAxis(
                data=np.array([f"ch_{i}" for i in range(n_chans)]), dims=["ch"]
            )
        },
    )

    backup = [copy.deepcopy(msg_in)]

    gen = affine_transform(weights=np.eye(n_chans), axis="ch")
    msg_out = gen.send(msg_in)
    assert msg_out.data.shape == in_dat.shape
    assert np.allclose(msg_out.data, in_dat)
    assert not np.may_share_memory(msg_out.data, in_dat)

    assert_messages_equal([msg_in], backup)

    # Send again just to make sure the generator doesn't crash
    _ = gen.send(msg_in)

    # Test with weights from a CSV file.
    csv_path = Path(__file__).parents[1] / "resources" / "xform.csv"
    weights = np.loadtxt(csv_path, delimiter=",")
    expected_out = in_dat @ weights.T
    # Same result: expected_out = np.vstack([(step[None, :] * weights).sum(axis=1) for step in in_dat])

    gen = affine_transform(weights=csv_path, axis="ch", right_multiply=False)
    msg_out = gen.send(msg_in)
    assert np.allclose(msg_out.data, expected_out)
    assert len(msg_out.axes["ch"].data) == weights.shape[0]
    assert (msg_out.axes["ch"].data[:-1] == msg_in.axes["ch"].data).all()

    # Try again as str, not Path
    gen = affine_transform(weights=str(csv_path), axis="ch", right_multiply=False)
    msg_out = gen.send(msg_in)
    assert np.allclose(msg_out.data, expected_out)
    assert len(msg_out.axes["ch"].data) == weights.shape[0]

    # Try again as direct ndarray
    gen = affine_transform(weights=weights, axis="ch", right_multiply=False)
    msg_out = gen.send(msg_in)
    assert np.allclose(msg_out.data, expected_out)
    assert len(msg_out.axes["ch"].data) == weights.shape[0]

    # One more time, but we pre-transpose the weights and do not override right_multiply
    gen = affine_transform(weights=weights.T, axis="ch", right_multiply=True)
    msg_out = gen.send(msg_in)
    assert np.allclose(msg_out.data, expected_out)
    assert len(msg_out.axes["ch"].data) == weights.shape[0]


def test_affine_passthrough():
    n_times = 13
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    backup = [copy.deepcopy(msg_in)]

    gen = affine_transform(weights="passthrough", axis="does not matter")
    msg_out = gen.send(msg_in)
    # We wouldn't want out_data is in_dat ezmsg pipeline but it's fine for the generator
    assert msg_out.data is in_dat
    assert_messages_equal([msg_out], backup)


def test_common_rereference():
    n_times = 300
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    backup = [copy.deepcopy(msg_in)]

    gen = common_rereference(mode="mean", axis="ch", include_current=True)
    msg_out = gen.send(msg_in)
    assert np.array_equal(
        msg_out.data,
        msg_in.data - np.mean(msg_in.data, axis=1, keepdims=True),
    )

    assert_messages_equal([msg_in], backup)

    # Use a slow deliberate way of calculating the CAR uniquely for each channel, excluding itself.
    #  common_rereference uses a faster way of doing this, but we test against something intuitive.
    expected_out = []
    for ch_ix in range(n_chans):
        idx = np.arange(n_chans)
        idx = np.hstack((idx[:ch_ix], idx[ch_ix + 1 :]))
        expected_out.append(
            msg_in.data[..., ch_ix] - np.mean(msg_in.data[..., idx], axis=1)
        )
    expected_out = np.stack(expected_out).T

    gen = common_rereference(mode="mean", axis="ch", include_current=False)
    msg_out = gen.send(msg_in)  # 41 us
    assert np.allclose(msg_out.data, expected_out)

    # Instead of CAR, we could use affine_transform with weights that reproduce CAR.
    # However, this method is 30x slower than above. (Actual difference varies depending on data shape).
    if False:
        weights = -np.ones((n_chans, n_chans)) / (n_chans - 1)
        np.fill_diagonal(weights, 1)
        gen = affine_transform(weights=weights, axis="ch")
        msg_out = gen.send(msg_in)
        assert np.allclose(msg_out.data, expected_out)


def test_car_passthrough():
    n_times = 300
    n_chans = 64
    in_dat = np.arange(n_times * n_chans).reshape(n_times, n_chans)
    msg_in = AxisArray(in_dat, dims=["time", "ch"])

    gen = common_rereference(mode="passthrough")
    msg_out = gen.send(msg_in)
    assert np.array_equal(msg_out.data, in_dat)
    assert np.may_share_memory(msg_out.data, in_dat)
