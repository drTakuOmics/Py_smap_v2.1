"""CLI-level tests: batch-size equivalence via smap.compare, path remapping, incomplete runs."""

import json
import shutil

import pytest

from smap.compare import compare_outputs, format_report
from smap.search_global import main
from synthetic import make_tile


@pytest.fixture(scope="module")
def tile(tmp_path_factory):
    d = tmp_path_factory.mktemp("tile")
    make_tile(d, 1500, 50, 95, snr=14.0, phase_scale=0.002)
    return d


def _par(d, out, root="/work"):
    """search.par written with Docker-style /work paths, as in the RNF213 project."""
    p = d / f"{out}.par"
    p.write_text("function search_global\nnCores 1\n"
                 f"imageFile {root}/img.mrc\nstructureFile {root}/vol.mrc\noutputDir {root}/{out}\n"
                 "aPerPix 2.0\ndefocus 15000.0 14000.0 30.0\nangle_inc 20\nT_sample 0\n"
                 "arbThr 4.0\nmargin_pix 0\noptThr 5.0\ninc_degrees 2\n")
    return p


def test_batch_sizes_agree_and_path_map(tile):
    pm = f"/work={tile.as_posix()}"
    assert main([str(_par(tile, "b1")), "--device", "cpu", "--batch-size", "1", "--path-map", pm,
                 "--log", str(tile / "b1_run.log")]) == 0
    log = (tile / "b1_run.log").read_text(encoding="utf-8")
    assert "Done combining search output" in log and "starting optimization on" in log
    assert main([str(_par(tile, "b8x")), "--device", "cpu", "--batch-size", "8", "--path-map", pm,
                 "--output-dir", str(tile / "b8")]) == 0
    rep = compare_outputs(tile / "b1", tile / "b8")
    json.dumps(rep)                                  # --json must not fail on numpy scalars
    print(format_report(rep))
    assert rep["all_ok"], format_report(rep)
    assert rep["particles"]["n_paired"] >= 1
    st = json.loads((tile / "b8" / "search_status.json").read_text())
    prov = st["provenance"]
    assert prov["input_sha256"]["imageFile"]["sha256"] and prov["run_settings"]["batch_size"] == 8
    assert (tile / "b8" / "particles_schema.json").exists()


def test_compare_detects_differences(tile):
    if not (tile / "b1" / "particles.txt").exists():
        pytest.skip("depends on previous test")
    shutil.copytree(tile / "b1", tile / "b1_mod")
    lines = (tile / "b1_mod" / "particles.txt").read_text().splitlines()
    k, v = lines[1].split()
    lines[1] = f"{k}  {float(v) + 0.5:6.3f}"
    (tile / "b1_mod" / "particles.txt").write_text("\n".join(lines) + "\n")
    rep = compare_outputs(tile / "b1", tile / "b1_mod")
    assert not rep["checks"]["d_score_final"]["ok"]


def test_incomplete_run_is_not_marked_done(tile, capsys):
    rc = main([str(_par(tile, "inc")), "--device", "cpu", "--max-rotations", "30",
               "--path-map", f"/work={tile.as_posix()}"])
    out = capsys.readouterr().out
    assert rc == 2 and "Done combining search output" not in out
    assert not (tile / "inc" / "particles.txt").exists()
