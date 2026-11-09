"""tools/radial_scan.py: synthetic capsids at a known pixel size, defocus and astigmatism (new; not part of v2.1)."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from smap import io as sio
from smap.ctf import ctf

ROOT = Path(__file__).resolve().parents[1]
P_TRUE, P_GIVEN = 1.114, 1.16                 # A per pixel: true, and 4% too large
DF1, DF2, ANG = 12000.0, 10500.0, 35.0        # A, A, degrees (CTFFind convention)
RO, RI = 130.0, 105.0                         # capsid shell radii, A


def _tool():
    spec = importlib.util.spec_from_file_location("radial_scan", ROOT / "tools" / "radial_scan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["radial_scan"] = mod
    spec.loader.exec_module(mod)
    return mod


def _params(**kw):
    p = dict(sio.SEARCH_GLOBAL_DEFAULTS, aPerPix=P_TRUE, V_acc=300000.0, Cs=2e-5)
    p.update(kw)
    return p


def _shell(r):
    return 2 * np.sqrt(np.clip(RO ** 2 - r ** 2, 0, None)) - 2 * np.sqrt(np.clip(RI ** 2 - r ** 2, 0, None))


def _micrograph(n=1536, seed=0):
    """16 shells on a jittered grid, imaged with the smap CTF (as in test_ctf_tools), plus noise."""
    rng = np.random.default_rng(seed)
    step = 330.0 / P_TRUE
    g = n / 2 + (np.arange(4) - 1.5) * step
    centres = np.array([(y, x) for y in g for x in g]) + rng.uniform(-15, 15, (16, 2))
    yy, xx = np.indices((n, n))
    proj = np.zeros((n, n))
    for (y, x) in centres:
        proj += _shell(np.hypot(yy - y, xx - x) * P_TRUE)
    C = np.imag(np.fft.ifftshift(ctf([[DF1 / 10, DF2 / 10, -math.radians(ANG)]], [n, n], _params())[:, :, 0]))
    im = np.real(np.fft.ifft2(np.fft.fft2(proj) * C))
    im += 2.0 * im.std() * rng.standard_normal((n, n))
    return im.astype(np.float32), centres


def _map(path, vox=1.6, n=180):
    c = (n - 1) / 2
    z, y, x = np.ogrid[:n, :n, :n]
    r = np.sqrt((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2) * vox
    sio.mw(((r <= RO) & (r >= RI)).astype(np.float32), path, vox)


def test_ctf_matches_smap_convention():
    """Same CTF as smap.ctf (phase part, no envelope), including the astigmatism angle convention."""
    mod = _tool()
    n, p = 256, 1.5
    sc = mod.Scanner(np.random.default_rng(1).standard_normal((n, n)), p, (np.arange(50.0), np.ones(50), 20.0),
                     Cs_A=2e-5 * 1e10, V=300000.0, ang=-math.radians(ANG), amp=0.0, bin_=1)
    mine = sc.ctf(sc.kpx / p, DF1, DF2)[:, :-1]          # the Nyquist column is stored differently
    ref = np.imag(np.fft.ifftshift(ctf([[DF1 / 10, DF2 / 10, -math.radians(ANG)]], [n, n],
                                       _params(aPerPix=p, F_abs=0, a_i=0, deltaE=0))[:, :, 0]))[:, : n // 2]
    assert np.max(np.abs(mine - ref)) < 1e-6
    sc.ang = math.radians(ANG)                                 # mirrored angle: a different CTF
    assert np.corrcoef(sc.ctf(sc.kpx / p, DF1, DF2)[:, :-1].ravel(), ref.ravel())[0, 1] < 0.5


def test_calibration_and_positions(tmp_path):
    mod = _tool()
    im, centres = _micrograph()
    sio.mw(im, tmp_path / "mic.mrc", P_GIVEN)
    _map(tmp_path / "capsid.mrc")
    s2 = (P_TRUE / P_GIVEN) ** 2                  # a CTF fit at the wrong pixel size overestimates defocus by 1/s^2
    out = tmp_path / "scan"
    mod.main(["--image", str(tmp_path / "mic.mrc"), "--map", str(tmp_path / "capsid.mrc"), "--pixel", str(P_GIVEN),
              "--defocus", str(DF1 / s2), str(DF2 / s2), str(ANG), "--cs", "0.02", "--fine", "--out", str(out)])
    row = next(csv.DictReader(open(out / "pixel_calibration.csv")))
    assert abs(float(row["pixel_A"]) - P_TRUE) < 0.005
    assert abs((float(row["df1_A"]) + float(row["df2_A"])) / 2 - (DF1 + DF2) / 2) <= 600
    assert row["contrast"] == "as built" and row["reliable"] == "yes"
    assert float(row["top10_z"]) > float(row["top10_z_given"]) + 1
    pk = np.array([[float(r["y_px"]), float(r["x_px"])] for r in csv.DictReader(open(out / "mic_peaks.csv"))])
    d = np.min(np.hypot(*(centres[:, None, :] - pk[None, :, :]).transpose(2, 0, 1)), axis=1)
    assert np.sum(d < 10) >= 15                   # of 16 capsids
    assert len(pk) <= 18
    assert (out / "mic_scan.png").exists()
    mod.main(["--image", str(tmp_path / "mic.mrc"), "--map", str(tmp_path / "capsid.mrc"), "--pixel", str(P_TRUE),
              "--defocus", str(DF1), str(DF2), str(ANG), "--cs", "0.02", "--fine", "--name", "mic2", "--out", str(out)])
    assert len(list(csv.DictReader(open(out / "pixel_calibration.csv")))) == 2      # one row per image

    # the given astigmatism angle scores higher than the mirrored one
    prof = mod.particle_profile(sio.mr(tmp_path / "capsid.mrc"), 1.6)
    kw = dict(Cs_A=2e5, V=300000.0, bin_=3)
    given = mod.Scanner(im, P_TRUE, prof, ang=-math.radians(ANG), **kw).score(1.0, DF1, DF2)[0]
    mirrored = mod.Scanner(im, P_TRUE, prof, ang=math.radians(ANG), **kw).score(1.0, DF1, DF2)[0]
    assert given > mirrored + 0.5


def test_search_folder_coarse(tmp_path):
    mod = _tool()
    im, _ = _micrograph(n=1024, seed=2)
    f = tmp_path / "results" / "mic_0001"
    f.mkdir(parents=True)
    sio.mw(np.pad(im, 64), f / "searchImage.mrc", P_GIVEN)  # zero padding, as written by the search
    _map(f / "capsid_SP.mrc")
    txt = f"aPerPix {P_GIVEN}\ndefocus {DF1} {DF2} {ANG}\ndefocus_format CTFFind\nV_acc 300000.0\nCs 0.00002\n"
    (f / "search_status.json").write_text(json.dumps({"provenance": {"params_text": txt}}))
    mod.main([str(f), "--out", str(tmp_path / "scan")])
    assert (tmp_path / "scan" / "mic_0001_scan.png").exists()
    rows = list(csv.DictReader(open(tmp_path / "scan" / "mic_0001_peaks.csv")))
    assert len(rows) >= 4


def test_clear_errors(tmp_path):
    mod = _tool()
    with pytest.raises(SystemExit, match="search_status.json"):
        mod.from_folder(tmp_path)
    with pytest.raises(SystemExit, match="atomic model"):
        mod.read_map(tmp_path / "model.cif")
    with pytest.raises(SystemExit, match="not found"):
        mod.read_map(tmp_path / "missing_SP.mrc")
