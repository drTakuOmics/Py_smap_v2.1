import os
from smap_tools_python import fov_to_num, num_to_fov, check_base_dir


def test_fov_roundtrip():
    fov = '031523_B_123'
    num = fov_to_num(fov)
    assert fov_to_num(num_to_fov(num)) == num


def test_check_base_dir(monkeypatch):
    monkeypatch.setenv('SMAP_BASE_DIR', '/new')
    monkeypatch.setenv('SMAP_BASE_DIR_ACTUAL', '/old')
    assert check_base_dir('/old/data') == '/new/data'
    assert check_base_dir('rel/path') == '/new/rel/path'
    assert check_base_dir('') == '/new'
