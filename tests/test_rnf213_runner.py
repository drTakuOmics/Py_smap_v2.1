"""tools/rnf213/rnf213_port_tiles.py: parallel tree preparation for the RNF213 layouts (no searches)."""

import csv
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "rnf213_port_tiles", Path(__file__).resolve().parents[1] / "tools" / "rnf213" / "rnf213_port_tiles.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _tile(tree: Path, name: str, root_rel: str):
    d = tree / name
    (d / "output").mkdir(parents=True)
    (d / f"{name}.mrc").write_bytes(b"img")
    (d / "search.par").write_text(
        "function search_global\nnCores 1\n"
        f"imageFile /work/{root_rel}/{name}/{name}.mrc\nmodelFile /work/model_SP.mrc\n"
        f"outputDir /work/{root_rel}/{name}/output\naPerPix 1.072\naPerPix_search 1.5\n")


def _offsets(tree, names):
    with open(tree / "tile_offsets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "x0_zero_based", "y0_zero_based", "width", "height"])
        for i, n in enumerate(names):
            w.writerow([n, 640 * i, 0, 1024, 1024])


def test_prepare_single_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_SP.mrc").write_bytes(b"sp")
    _tile(tmp_path / "tiles_3p8", "tile_0001", "tiles_3p8")
    _tile(tmp_path / "tiles_3p8", "tile_0002", "tiles_3p8")
    _offsets(tmp_path / "tiles_3p8", ["tile_0001", "tile_0002"])
    runner.main(["prepare", "--source", "tiles_3p8", "--dest", "tiles_3p8_port", "--set", "refine_continuous=1"])
    par = (tmp_path / "tiles_3p8_port/tile_0002/search.par").read_text()
    assert "imageFile /work/tiles_3p8_port/tile_0002/tile_0002.mrc" in par
    assert "outputDir /work/tiles_3p8_port/tile_0002/output" in par
    assert "modelFile /work/model_SP.mrc" in par and par.rstrip().endswith("refine_continuous 1")
    assert (tmp_path / "tiles_3p8_port/tile_offsets.csv").read_text() == \
        (tmp_path / "tiles_3p8/tile_offsets.csv").read_text()
    assert (tmp_path / "tiles_3p8_port/tile_0001/tile_0001.mrc").read_bytes() == b"img"
    assert "/work/tiles_3p8/" in (tmp_path / "tiles_3p8/tile_0001/search.par").read_text()  # untouched
    with pytest.raises(SystemExit):   # typo guard
        runner.main(["prepare", "--source", "tiles_3p8", "--dest", "x", "--set", "refine_continuos=1"])
    with pytest.raises(SystemExit):   # never overwrite
        runner.main(["prepare", "--source", "tiles_3p8", "--dest", "tiles_3p8_port"])
    runner.main(["run", "--tiles", "tiles_3p8_port", "--dry-run"])


def test_prepare_batch_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_SP.mrc").write_bytes(b"sp")
    s = tmp_path / "batch_a/searches"
    rows = []
    for img in ("image_0001", "image_0002"):
        _tile(s / img, "tile_0001", f"batch_a/searches/{img}")
        _offsets(s / img, ["tile_0001"])
        (s / img / "micrograph_manifest.json").write_text("{}")
        d = s / img / "tile_0001"
        rows.append(dict(batch_image=img, tile="tile_0001", directory=f"batch_a/searches/{img}/tile_0001",
                         configuration_sha256=hashlib.sha256((d / "search.par").read_bytes()).hexdigest(),
                         tile_file="tile_0001.mrc", tile_sha256="x"))
    (s / "batch_search.json").write_text("{}")
    with open(s / "jobs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    runner.main(["prepare", "--source", "batch_a/searches", "--dest", "batch_a_port/searches"])
    d = tmp_path / "batch_a_port/searches"
    assert (d / "batch_search.json").exists() and (d / "image_0002/micrograph_manifest.json").exists()
    jobs = list(csv.DictReader(open(d / "jobs.csv")))
    assert jobs[1]["directory"] == "batch_a_port/searches/image_0002/tile_0001"
    par = d / "image_0002/tile_0001/search.par"
    assert jobs[1]["configuration_sha256"] == hashlib.sha256(par.read_bytes()).hexdigest()
    assert "outputDir /work/batch_a_port/searches/image_0002/tile_0001/output" in par.read_text()
    assert os.path.samefile(d / "image_0001/tile_0001/tile_0001.mrc", s / "image_0001/tile_0001/tile_0001.mrc")
    runner.main(["run", "--tiles", "batch_a_port/searches", "--images", "image_0002", "--dry-run"])
    with pytest.raises(SystemExit):
        runner.main(["run", "--tiles", "batch_a_port/searches", "--images", "image_0099", "--dry-run"])
