"""tools/rnf213/rnf213_correlogram.py on a synthetic port run: files, coordinates, never overwrite."""

import csv
import importlib.util
from pathlib import Path

import pytest

from smap.search_global import main
from synthetic import make_tile, write_par

spec = importlib.util.spec_from_file_location(
    "rnf213_correlogram", Path(__file__).resolve().parents[1] / "tools" / "rnf213" / "rnf213_correlogram.py")
corr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corr)
spec2 = importlib.util.spec_from_file_location(
    "rnf213_landscape", Path(__file__).resolve().parents[1] / "tools" / "rnf213" / "rnf213_landscape.py")
land = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(land)


def test_correlogram_figures(tmp_path):
    pytest.importorskip("matplotlib")
    tile = tmp_path / "tiles" / "tile_0001"
    tile.mkdir(parents=True)
    make_tile(tile, 1500, 50, 95, snr=14.0, phase_scale=0.002)
    (tmp_path / "tiles" / "tile_offsets.csv").write_text(
        "tile,x0_zero_based,y0_zero_based,width,height\ntile_0001,640,100,144,144\n")
    par = write_par(tile, tile / "img.mrc", n_cores=1, extra="optThr 5.0\ninc_degrees 2\n")
    (tile / "search.par").write_text(par.read_text().replace(f"outputDir {tile / 'out'}",
                                                              f"outputDir {tile / 'output'}"))
    assert main([str(tile / "search.par"), "--device", "cpu", "--batch-size", "16"]) == 0
    out = tmp_path / "figs"
    corr.main(["--tile", str(tile), "--compare", str(tile), "--compare-label", "same",
               "--cutoff", "6", "--window", "60", "--out", str(out)])
    for name in ("overview", "gallery", "candidate_01"):
        for ext in ("png", "pdf", "svg", "json"):
            assert (out / f"{name}.{ext}").exists()
    rows = list(csv.DictReader(open(out / "candidates.csv")))
    best = max(rows, key=lambda r: float(r["final_score"]))
    # planted at 0-based (row 50, col 95) of the tile; tile origin (x0, y0) = (640, 100)
    assert abs(float(best["x_px"]) - (95 + 640)) <= 1 and abs(float(best["y_px"]) - (50 + 100)) <= 1
    with pytest.raises(SystemExit):
        corr.main(["--tile", str(tile), "--out", str(out)])

    # whole-micrograph landscape (one section searched)
    land.main(["--tree", str(tmp_path / "tiles"), "--bin", "6", "--cutoff", "6", "--out", str(tmp_path / "land")])
    assert (tmp_path / "land" / "landscape.png").exists()
    lrows = list(csv.DictReader(open(tmp_path / "land" / "landscape_candidates.csv")))
    assert abs(float(lrows[0]["x_px"]) - 735) <= 1 and abs(float(lrows[0]["y_px"]) - 150) <= 1
