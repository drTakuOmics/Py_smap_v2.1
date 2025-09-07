import numpy as np
import pytest

mrcfile = pytest.importorskip("mrcfile")
from smap_tools_python import tr, ri, tw


def test_tr_reads_multiframe_tiff(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = np.arange(12, dtype=np.uint8).reshape(3, 4, 1)
    data = np.concatenate([data, data + 1], axis=2)
    path = tmp_path / "test.tif"
    tifffile.imwrite(str(path), np.moveaxis(data, -1, 0))
    out = tr(path)
    assert out.shape == data.shape
    assert np.all(out == data)


def test_ri_handles_mrc(tmp_path):
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    mrc_path = tmp_path / "vol.mrc"
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(arr)
        mrc.voxel_size = (1.5, 1.5, 1.5)
    data, info = ri(mrc_path)
    assert np.allclose(data, arr)
    assert info["voxel_size"] == (1.5, 1.5, 1.5)


def test_ri_handles_tiff(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(6, dtype=np.uint8).reshape(3, 2)
    path = tmp_path / "im.tif"
    tifffile.imwrite(str(path), arr)
    data, info = ri(path)
    assert data.shape == (3, 2, 1)
    assert info == {}


def test_tw_writes_tiff(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = np.arange(12, dtype=np.float32).reshape(3, 4, 1)
    path = tmp_path / "out.tif"
    tw(data, path, bps=32)
    out = tr(path)
    assert np.allclose(out, data)
