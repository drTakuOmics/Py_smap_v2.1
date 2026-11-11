"""tools/star_to_pars.py: RELION 3.0 and 3.1 particle STAR files -> per-micrograph .par (new; not part of v2.1)."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

from smap import io as sio

ROOT = Path(__file__).resolve().parents[1]
PAR = """function search_global
nCores 1
imageFile /placeholder.mrc
structureFile /models/capsid.cif
outputDir /placeholder
aPerPix 0.788
defocus 0 0 0
V_acc 300000.0
Cs 0.0027
"""
COLS30 = ["rlnVoltage", "rlnDefocusAngle", "rlnSphericalAberration", "rlnDetectorPixelSize", "rlnDefocusU",
          "rlnDefocusV", "rlnMagnification", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnMicrographName",
          "rlnCoordinateX", "rlnCoordinateY", "rlnOriginX", "rlnOriginY"]


def _tool():
    spec = importlib.util.spec_from_file_location("star_to_pars", ROOT / "tools" / "star_to_pars.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["star_to_pars"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows30():
    rows = []
    for m, n, du in (("mics/a_st_Stk_corrected.mrc", 3, 11000.0), ("mics/b_st_Stk_corrected.mrc", 5, 15000.0)):
        for k in range(n):
            ang = 179.0 if k % 2 else 1.0                       # wraps around 0/180: circular mean ~0
            rows.append(["300", f"{ang}", "2.7", "5.0", f"{du + 10 * k}", f"{du - 500 + 10 * k}", "63451.5",
                         "10", "20", "30", m, f"{100 + k}", f"{200 + k}", "1.5", "-2.0"])
    return rows


def test_star30_to_pars(tmp_path):
    mod = _tool()
    star = tmp_path / "p.star"
    star.write_text("data_\nloop_\n" + "".join(f"_{c} #{i + 1}\n" for i, c in enumerate(COLS30))
                    + "\n".join(" ".join(r) for r in _rows30()) + "\n")
    (tmp_path / "t.par").write_text(PAR)
    out = tmp_path / "pars"
    mod.main([str(star), "--template", str(tmp_path / "t.par"), "--micrograph-dir", str(tmp_path / "dw"),
              "--out-dir", str(out), "--results-dir", str(tmp_path / "res"), "--top", "1",
              "--rename", r"_st_Stk_corrected\.mrc$=-a-DW_magcorrect.mrc", "--url-prefix", "ftp://x/03/"])
    jobs = (out / "jobs.txt").read_text().split()
    assert len(jobs) == 1 and jobs[0].endswith("b-a-DW_magcorrect.par")
    p, _ = sio.read_params_file(jobs[0])
    assert str(p["imageFile"]).endswith("dw/b-a-DW_magcorrect.mrc")
    df = [float(x) for x in p["defocus"]]
    assert abs(df[0] - 15020) < 1 and abs(df[1] - 14520) < 1 and (df[2] < 1 or df[2] > 179)
    assert abs(float(p["Cs"]) - 0.0027) < 1e-9
    t = {r["micrograph"]: r for r in csv.DictReader(open(out / "ctf_table.csv"))}
    assert t["b-a-DW_magcorrect"]["particles"] == "5" and abs(float(t["a-a-DW_magcorrect"]["pixel_A"]) - 0.788) < 1e-3
    c = list(csv.DictReader(open(out / "coords" / "b-a-DW_magcorrect.csv")))
    assert len(c) == 5 and float(c[0]["x_px"]) == 98.5 and float(c[0]["y_px"]) == 202.0      # coordinate - origin
    assert (out / "download_urls.txt").read_text().strip() == "ftp://x/03/b-a-DW_magcorrect.mrc"


def test_star31_optics(tmp_path):
    mod = _tool()
    star = tmp_path / "p31.star"
    star.write_text("data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnVoltage #2\n_rlnSphericalAberration #3\n"
                    "_rlnImagePixelSize #4\n1 300 2.7 0.788\n\ndata_particles\nloop_\n_rlnOpticsGroup #1\n"
                    "_rlnMicrographName #2\n_rlnDefocusU #3\n_rlnDefocusV #4\n_rlnDefocusAngle #5\n"
                    "_rlnCoordinateX #6\n_rlnCoordinateY #7\n1 m1.mrc 12000 11500 45 10 20\n1 m1.mrc 12100 11600 45 30 40\n")
    rows = mod.particles(star)
    assert len(rows) == 2 and rows[0]["rlnVoltage"] == "300" and mod.pixel_size(rows[0]) == 0.788


def test_match_positions(tmp_path):
    spec = importlib.util.spec_from_file_location("match_positions", ROOT / "tools" / "match_positions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import json
    import numpy as np
    f = tmp_path / "res"
    f.mkdir()
    sio.mw(np.zeros((400, 600), np.float32), tmp_path / "mic.mrc", 1.0)
    txt = f"imageFile {tmp_path / 'mic.mrc'}\naPerPix 1.0\naPerPix_search 1.5\nmargin_pix 10\n"
    (f / "search_status.json").write_text(json.dumps({"gaussian_threshold": 7.4, "provenance": {"params_text": txt}}))
    ref = [(100, 100), (300, 200), (500, 300), (5, 200)]            # the last one is inside the margin
    det = [(102, 99, 9.0), (301, 203, 7.6), (450, 50, 8.5), (5, 200, 6.0)]
    (f / "particles.csv").write_text("num_id,score_final,x_orig0_est,y_orig0_est\n"
                                     + "".join(f"{k + 1},{s},{x},{y}\n" for k, (x, y, s) in enumerate(det)))
    (tmp_path / "ref.csv").write_text("x_px,y_px\n" + "".join(f"{x},{y}\n" for x, y in ref))
    mod.main([str(f), str(tmp_path / "ref.csv"), "--radius-A", "10", "--cutoffs", "8"])
    rows = list(csv.DictReader(open(f / "match_ref.csv")))
    assert [r["reference_index"] for r in rows] == ["0", "1", "-1", "3"]
    assert mod.pair(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]), 5.0) == [(0, 0)]
    (tmp_path / "coords").mkdir()
    (tmp_path / "coords" / "res.csv").write_text((tmp_path / "ref.csv").read_text())
    assert mod.main([str(f), "--reference-dir", str(tmp_path / "coords"), "--radius-A", "10",
                     "--summary", str(tmp_path / "sum.csv")]) == 0
    r = next(csv.DictReader(open(tmp_path / "sum.csv")))
    assert (r["found"], r["missed"], r["extra"], r["inside"]) == ("2", "1", "1", "3")
