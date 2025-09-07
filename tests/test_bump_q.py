import numpy as np
from smap_tools_python import bump_q, Quaternion


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_bump_q_matrix():
    Rz = rot_z(np.pi / 2)
    Rx = rot_x(np.pi / 2)
    out = bump_q(Rz, Rx)
    assert out.shape == (3, 3, 1)
    assert np.allclose(out[:, :, 0], Rx @ Rz)


def test_bump_q_quaternion():
    qin = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
    qbump = Quaternion.from_axis_angle([1, 0, 0], np.pi / 2)
    out = bump_q(qin, qbump)
    assert np.allclose(out[:, :, 0], rot_x(np.pi / 2) @ rot_z(np.pi / 2))


def test_bump_q_multiple():
    R_in = np.stack([np.eye(3), rot_z(np.pi / 2)], axis=2)
    R_bump = np.stack([np.eye(3), rot_x(np.pi / 2)], axis=2)
    out = bump_q(R_in, R_bump)
    assert out.shape == (3, 3, 4)
    assert np.allclose(out[:, :, 0], np.eye(3))
    assert np.allclose(out[:, :, 3], rot_x(np.pi / 2) @ rot_z(np.pi / 2))
