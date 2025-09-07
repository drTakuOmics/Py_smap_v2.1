import numpy as np

from smap_tools_python.frealign2smap import frealign2smap
from smap_tools_python.smap2frealign import smap2frealign
from smap_tools_python.cistem2smap import cistem2smap
from smap_tools_python.smap2cistem import smap2cistem
from smap_tools_python.smap2pymol import smap2pymol


def test_smap2frealign_roundtrip():
    ang = np.array([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0], [45.0, 60.0, 90.0]])
    rot, _ = frealign2smap(ang)
    ori, q = smap2frealign(rot)
    assert np.allclose(ori, ang, atol=1e-6)
    assert q.shape == (4, ang.shape[0])


def test_smap2cistem_roundtrip():
    ea = np.array([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0], [90.0, 45.0, 10.0]])
    rot = cistem2smap(ea)
    back = smap2cistem(rot)
    assert np.allclose(back, ea, atol=1e-6)


def test_smap2pymol_axis_angle():
    from scipy.spatial.transform import Rotation as R

    rot = R.from_euler("z", 90, degrees=True).as_matrix()
    axan = smap2pymol(rot)
    assert np.allclose(axan[0, :3], [0, 0, 1], atol=1e-6)
    assert np.allclose(axan[0, 3], 90.0, atol=1e-6)

