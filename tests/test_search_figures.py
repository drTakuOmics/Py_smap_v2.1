"""tools/search_figures.py: pose convention of the rendered surfaces, and a smoke run."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from smap import io as sio

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("search_figures", ROOT / "tools" / "search_figures.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["search_figures"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def test_rotated_volume_projects_to_the_search_template(tmp_path):
    from synthetic import volume, write_par

    from smap.search_global import make_templates, prepare
    mod = _tool()
    rng = np.random.default_rng(0)
    sio.mw(np.transpose(volume(rng), (1, 2, 0)), tmp_path / "vol.mrc", 2.0)
    sio.mw(rng.standard_normal((144, 144)).astype(np.float32), tmp_path / "img.mrc", 2.0)
    st = prepare(write_par(tmp_path, tmp_path / "img.mrc"), write_files=False)
    V = sio.mr(tmp_path / "vol.mrc").astype(np.float64)
    R = _rot([0.3, -0.8, 0.5], 63.0)
    st.R_full = np.stack([R])
    t = make_templates(st, [1])[0]
    p_ok = mod.rotate_volume(V, R).sum(axis=2)
    p_tr = mod.rotate_volume(V, R.T).sum(axis=2)
    from scipy.ndimage import gaussian_filter

    def hp(x):                       # remove the round envelope shared by all projections
        return x - gaussian_filter(x, 3.0)
    ok = _ncc(hp(t), hp(p_ok))
    assert ok > 0.9
    assert ok > _ncc(hp(t), hp(p_tr)) + 0.3
    assert ok > _ncc(hp(t), hp(p_ok.T)) + 0.3


def test_search_figures_smoke(tmp_path):
    mod = _tool()
    rng = np.random.default_rng(1)
    res = tmp_path / "res"
    res.mkdir()
    a, a_s, n_img = 2.0, 3.0, 300
    img = rng.standard_normal((n_img, n_img)).astype(np.float32)
    sio.mw(img, tmp_path / "mic.mrc", a)
    ng = 200
    z = rng.standard_normal((ng, ng)).astype(np.float32) + 4.9
    z[:10] = 0
    z[50, 60], z[120, 130] = 11.0, 8.5
    sio.mw(z, res / "search_mip_scaled.mrc", a_s)
    yy, xx, zz = np.mgrid[:40, :40, :40] - 20
    V = (np.exp(-((xx - 6) ** 2 + yy ** 2 + zz ** 2) / 30.0) + np.exp(-(xx ** 2 + (yy - 8) ** 2 + zz ** 2) / 20.0))
    sio.mw(V.astype(np.float32), res / "tmpl_SP.mrc", a)
    (res / "search_status.json").write_text(json.dumps(dict(
        sampling_A=a_s, provenance=dict(params_text=f"function search_global\naPerPix {a}\n",
                                        input_sha256=dict(imageFile=dict(path=str(tmp_path / "mic.mrc")))))))
    cols = ("num_id,score_final,score_refined,x_orig0_est,y_orig0_est,row,col,row_opt,col_opt,row_search,"
            "col_search,df1_nm,df2_nm,ast_rad," + ",".join(f"q_opt_{i}{j}" for i in range(3) for j in range(3))
            + ",neg_log_p_search")
    rows = []
    for k, (r, c, s) in enumerate([(51, 61, 12.0), (121, 131, 7.0)], 1):
        y, x = (r - (ng // 2 + 1)) * 1.5 + n_img // 2, (c - (ng // 2 + 1)) * 1.5 + n_img // 2
        q = _rot([1, 2, 3], 30 * k).ravel()
        rows.append(",".join(str(v) for v in [k, s, s, x, y, 0, 0, 0, 0, r, c, 400, 380, 0.1, *q, 20]))
    (res / "particles.csv").write_text(cols + "\n" + "\n".join(rows) + "\n")
    out = tmp_path / "fig"
    assert mod.main([str(res), "--cutoff", "7.5", "--out", str(out), "--label", "test",
                     "--render-A", "4", "--display-bin-A", "4", "--landscape-bin-A", "12"]) == 0
    for f in ("poses.png", "poses_pairs.png", "zmap.png", "landscape.png", "summary.png", "figures.json"):
        assert (out / f).stat().st_size > 0
    meta = json.loads((out / "figures.json").read_text())
    assert meta["n_detected"] == 1 and meta["n_refined"] == 2
