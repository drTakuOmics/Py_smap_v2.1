"""tools/export_detections.py: duplicates, crowded micrographs, orientations modulo symmetry, STAR (new; not part of v2.1)."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

from smap import io as sio
from smap.export import read_star_particles, relion_angles

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("export_detections", ROOT / "tools" / "export_detections.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


C4 = np.array([_rz(90 * k) for k in range(4)])


def _search(folder, ops, parts):
    folder.mkdir(parents=True)
    txt = (f"aPerPix 1.0\nV_acc 300000.0\nCs 0.0027\nimageFile /data/{folder.name}.mrc\nsymmetryFile {ops}\n")
    (folder / "search_status.json").write_text(json.dumps({"gaussian_threshold": 7.0, "provenance": {"params_text": txt}}))
    cols = ["num_id", "score_final", "score_refined", "x_orig0_est", "y_orig0_est", "df1_nm", "df2_nm", "ast_rad"] + \
        [f"q_opt_{i}{j}" for i in range(3) for j in range(3)]
    with open(folder / "particles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for k, (x, y, s, R) in enumerate(parts, 1):
            w.writerow([k, s, s, x, y, 1000, 990, 0.1, *R.ravel()])
    rng = np.random.default_rng(len(parts))                         # raw patches: mean 30, SD 2
    sio.mw((30 + 2 * rng.standard_normal((16, 16, len(parts)))).astype(np.float32), folder / "particles.mrc", 1.0)


def test_export_detections(tmp_path):
    mod = _tool()
    ops = tmp_path / "c4.txt"
    sio.write_rotations_file(C4, ops)
    Rs = [_rx(20) @ _rz(10 * k) for k in range(9)]
    f1 = tmp_path / "res" / "mic1"
    _search(f1, ops, [(100, 100, 12.0, Rs[0]), (103, 101, 9.0, Rs[1]), (200, 200, 11.0, Rs[2]), (300, 100, 6.0, Rs[3])])
    f2 = tmp_path / "res" / "mic2"
    _search(f2, ops, [(100 + 20 * (k % 3), 100 + 20 * (k // 3), 10.0, Rs[k]) for k in range(9)])
    ref = tmp_path / "coords"
    ref.mkdir()
    with open(ref / "mic1.csv", "w", newline="") as fh:                # the same poses, symmetry mates
        w = csv.writer(fh)
        w.writerow(["x_px", "y_px", "rot", "tilt", "psi"])
        for (x, y), R in (((101, 100), Rs[0] @ C4[1]), ((200, 201), Rs[2] @ C4[3])):
            w.writerow([x, y, *relion_angles(R)])
    out = tmp_path / "export"
    (out / "stacks").mkdir(parents=True)
    (out / "stacks" / "mic1.mrcs").symlink_to(f1 / "particles.mrc")      # as left by an earlier export
    before = (f1 / "particles.mrc").read_bytes()
    assert mod.main([str(f1), str(f2), "--out-dir", str(out), "--diameter-A", "50", "--reference-dir", str(ref)]) == 0
    t = {r["micrograph"]: r for r in csv.DictReader(open(out / "micrographs.csv"))}
    assert (t["mic1"]["detected"], t["mic1"]["duplicates"], t["mic1"]["exported"]) == ("3", "1", "2")
    assert t["mic2"]["exported"] == "0" and float(t["mic2"]["overlap_per_detection"]) > 0.5
    rows = read_star_particles(out / "particles.star")
    assert len(rows) == 2 and {r["rlnImageName"] for r in rows} == {"000001@stacks/mic1.mrcs", "000002@stacks/mic1.mrcs"}
    assert sorted(r["rlnRandomSubset"] for r in rows) == ["1", "2"]
    st = out / "stacks" / "mic1.mrcs"
    assert not st.is_symlink() and not (out / "stacks" / "mic2.mrcs").exists()
    assert (f1 / "particles.mrc").read_bytes() == before                  # the search output is untouched
    import mrcfile
    with mrcfile.open(st) as m:
        assert m.is_image_stack() and m.data.shape == (2, 16, 16)
        src = sio.mr(f1 / "particles.mrc")                               # (16, 16, n)
        for k, j in enumerate((0, 2)):                                   # kept detections 1 and 3, in order
            raw = src[:, :, j].astype(np.float64)
            # particle radius (25 px) beyond the 16 px box: the whole image is the background
            assert np.allclose(m.data[k], (raw - raw.mean()) / raw.std(), atol=1e-4)
    assert abs(float(t["mic1"]["raw_mean"]) - 30) < 1
    ang = [float(r["angle_deg"]) for r in csv.DictReader(open(out / "poses.csv"))]
    assert len(ang) == 2 and max(ang) < 0.01
    assert mod.sym_angle(Rs[0], Rs[0] @ _rx(5), C4) > 4.9                # a real difference is not hidden
