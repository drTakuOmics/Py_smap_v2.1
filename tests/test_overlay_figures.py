"""tools/overlay_figures.py: several searches of one micrograph in one figure (new; not part of v2.1)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from smap import io as sio

ROOT = Path(__file__).resolve().parents[1]
COLS = ("num_id,score_final,score_refined,x_orig0_est,y_orig0_est,row,col,row_opt,col_opt,row_search,col_search,"
        "df1_nm,df2_nm,ast_rad," + ",".join(f"q_opt_{i}{j}" for i in range(3) for j in range(3)) + ",neg_log_p_search")


def _tool():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("overlay_figures", ROOT / "tools" / "overlay_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _search(d: Path, mic: Path, parts, shift=0.0):
    d.mkdir(parents=True)
    yy, xx, zz = np.mgrid[:40, :40, :40] - 20
    V = np.exp(-((xx - 6 - shift) ** 2 + yy ** 2 + zz ** 2) / 30.0) + np.exp(-(xx ** 2 + (yy - 8) ** 2 + zz ** 2) / 20.0)
    sio.mw(V.astype(np.float32), d / "tmpl_SP.mrc", 2.0)
    (d / "search_status.json").write_text(json.dumps(dict(sampling_A=3.0, provenance=dict(
        params_text="function search_global\naPerPix 2.0\n",
        input_sha256=dict(imageFile=dict(path=str(mic), sha256=mic.name))))))
    rows = [",".join(str(v) for v in [k, s, s, x, y, 0, 0, 0, 0, 1, 1, 400, 380, 0.1, *np.eye(3).ravel(), 20])
            for k, (x, y, s) in enumerate(parts, 1)]
    (d / "particles.csv").write_text(COLS + "\n" + "\n".join(rows) + "\n")


def test_overlay(tmp_path):
    mod = _tool()
    rng = np.random.default_rng(0)
    mic = tmp_path / "mic.mrc"
    sio.mw(rng.standard_normal((300, 300)).astype(np.float32), mic, 2.0)
    _search(tmp_path / "closed", mic, [(100, 100, 9.0), (200, 220, 8.0), (60, 240, 6.0)])
    _search(tmp_path / "open", mic, [(102, 101, 8.5), (250, 60, 7.9)], shift=3)
    out = tmp_path / "fig"
    assert mod.main([str(tmp_path / "closed"), str(tmp_path / "open"), "--labels", "closed", "open",
                     "--colors", "#17a2a2", "#d63fa0", "--cutoff", "7", "--out", str(out),
                     "--render-A", "4", "--overview-render-A", "4", "--pair-half-A", "80"]) == 0
    for f in ("overlay.png", "overlay_pairs.png", "overlay.csv", "overlay.json"):
        assert (out / f).stat().st_size > 0
    meta = json.loads((out / "overlay.json").read_text())
    assert [s["detected"] for s in meta["searches"]] == [2, 2] and meta["found_by_first_two"] == 1
    rows = list(csv.DictReader(open(out / "overlay.csv")))
    assert len(rows) == 4                                              # every detection as found, none removed
    r = next(r for r in rows if r["search"] == "closed" and r["id"] == "1")
    assert abs(float(r["nearest_open_A"]) - np.hypot(2, 1) * 2) < 0.1 and r["nearest_open_score"] == "8.5"
    with pytest.raises(SystemExit):                                    # never overwrites
        mod.main([str(tmp_path / "closed"), str(tmp_path / "open"), "--cutoff", "7", "--out", str(out)])


def test_overlay_needs_the_same_micrograph(tmp_path):
    mod = _tool()
    for name in ("a.mrc", "b.mrc"):
        sio.mw(np.zeros((64, 64), np.float32), tmp_path / name, 2.0)
    _search(tmp_path / "s1", tmp_path / "a.mrc", [(60, 60, 9.0)])
    _search(tmp_path / "s2", tmp_path / "b.mrc", [(60, 60, 9.0)])
    with pytest.raises(SystemExit, match="different micrographs"):
        mod.main([str(tmp_path / "s1"), str(tmp_path / "s2"), "--cutoff", "7", "--out", str(tmp_path / "fig")])
