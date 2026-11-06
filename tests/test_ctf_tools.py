"""tools/cryosparc_ctf_to_pars.py and tools/ctf_check.py (new; not part of v2.1)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

from smap import io as sio
from smap.ctf import ctf

ROOT = Path(__file__).resolve().parents[1]


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PAR = """function search_global
nCores 1
imageFile /placeholder.mrc
structureFile /models/capsid.cif
outputDir /placeholder
aPerPix 1.16
defocus 0 0 0
V_acc 300000.0
Cs 0.00002
"""


def _write_cs(path, rows, fields):
    dt = [(f, t) for f, t in fields]
    a = np.array(rows, dtype=dt)
    with open(path, "wb") as fh:
        np.save(fh, a)


def test_cryosparc_ctf_to_pars(tmp_path):
    mod = _tool("cryosparc_ctf_to_pars")
    mdir = tmp_path / "mics"
    mdir.mkdir()
    for n in ("mic_a.mrc", "mic_b.mrc"):
        (mdir / n).write_bytes(b"")
    ctf_f = [("uid", "<u8"), ("ctf/df1_A", "<f4"), ("ctf/df2_A", "<f4"), ("ctf/df_angle_rad", "<f4"),
             ("ctf/cs_mm", "<f4"), ("ctf/accel_kv", "<f4"), ("ctf_stats/ctf_fit_to_A", "<f4")]
    pt_f = [("uid", "<u8"), ("micrograph_blob/path", "S80"), ("micrograph_blob/psize_A", "<f4")]
    _write_cs(tmp_path / "ctf.cs", [(2, 15200, 14800, math.radians(-40), 0.02, 300, 4.1),
                                    (1, 9000, 8700, math.radians(25), 0.02, 300, 12.0),
                                    (3, 20000, 19000, 0.1, 0.02, 300, 3.9)], ctf_f)
    _write_cs(tmp_path / "pt.cs", [(1, b"J1/imported/012345678901234567_mic_a.mrc", 1.16),
                                   (2, b"J1/imported/098765432109876543_mic_b.mrc", 1.16),
                                   (3, b"J1/imported/011111111111111111_mic_c.mrc", 1.16)], pt_f)
    (tmp_path / "t.par").write_text(PAR)
    out = tmp_path / "pars"
    rc = mod.main([str(tmp_path / "ctf.cs"), str(tmp_path / "pt.cs"), "--template", str(tmp_path / "t.par"),
                   "--micrograph-dir", str(mdir), "--out-dir", str(out), "--results-dir",
                   str(tmp_path / "res"), "--max-fit-A", "8"])
    assert rc == 0
    jobs = (out / "jobs.txt").read_text().split()
    assert [Path(j).name for j in jobs] == ["mic_b.par"]          # a: fit 12 A; c: no file
    p, _ = sio.read_params_file(out / "mic_b.par")
    assert np.allclose(p["defocus"], [15200, 14800, -40.0], atol=0.01)
    assert p["imageFile"] == str((mdir / "mic_b.mrc").resolve())
    assert p["outputDir"] == str((tmp_path / "res" / "mic_b").resolve())
    assert p["Cs"] == 0.00002 and p["aPerPix"] == 1.16
    table = (out / "ctf_table.csv").read_text()
    assert "fit 12.0 A > 8.0" in table and "micrograph not found" in table


def test_ctf_check_finds_the_simulated_convention():
    mod = _tool("ctf_check")
    params = dict(sio.SEARCH_GLOBAL_DEFAULTS, aPerPix=1.16, V_acc=300000.0, Cs=2e-5,
                  defocus=[15000.0, 13800.0, 35.0], defocus_format="CTFFind")
    n = 1024
    df_nm_rad = [1500.0, 1380.0, 35.0 * (-math.pi / 180)]
    C = np.imag(np.fft.ifftshift(ctf([df_nm_rad], [n, n], params)[:, :, 0]))
    rng = np.random.default_rng(0)
    im = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * C))
    im += 0.5 * im.std() * rng.standard_normal((n, n))
    res, *_ = mod.check(params, im, tile=256, band_A=(30.0, 4.0), scan_A=400.0)
    assert res["given"] > 0.5
    assert res["given"] > res["angle_flipped"] + 0.1
    assert res["given"] > res["df_swapped"] + 0.1
    assert abs(res["best_offset_A"]) <= 50


def test_cryosparc_pixel_rescale_and_overrides(tmp_path):
    mod = _tool("cryosparc_ctf_to_pars")
    mdir = tmp_path / "mics"
    mdir.mkdir()
    (mdir / "m1.mrc").write_bytes(b"")
    _write_cs(tmp_path / "c.cs", [(5, 15000, 14000, 0.2, b"J1/imported/012345678901234567_m1.mrc", 1.16)],
              [("uid", "<u8"), ("ctf/df1_A", "<f4"), ("ctf/df2_A", "<f4"), ("ctf/df_angle_rad", "<f4"),
               ("micrograph_blob/path", "S80"), ("micrograph_blob/psize_A", "<f4")])
    (tmp_path / "t.par").write_text(PAR + "angle_inc 3.8\n")
    out = tmp_path / "p"
    mod.main([str(tmp_path / "c.cs"), "--template", str(tmp_path / "t.par"), "--micrograph-dir", str(mdir),
              "--out-dir", str(out), "--results-dir", str(tmp_path / "r"), "--pixel", "1.14",
              "--set", "angle_inc=2.5", "--set", "structureFile=/models/capsid_SP.mrc"])
    p, _ = sio.read_params_file(out / "m1.par")
    s = (1.14 / 1.16) ** 2
    assert p["aPerPix"] == 1.14 and p["angle_inc"] == 2.5
    assert p["structureFile"] == "/models/capsid_SP.mrc"
    assert np.allclose(p["defocus"][:2], [15000 * s, 14000 * s], atol=0.06)
