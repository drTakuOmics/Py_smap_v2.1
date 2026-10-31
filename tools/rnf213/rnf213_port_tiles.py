#!/usr/bin/env python3
"""Run the Python port on tiles already prepared for SMAP v2.1 (RNF213 project layout).

The v2.1 tree is never modified. ``prepare`` builds a parallel tree whose search.par files
differ from the v2.1 ones only in imageFile/outputDir (plus any --set options); tile
images are hard-linked (copied if linking fails). The new tree has the same layout
(tile_offsets.csv, tile_XXXX/{search.par, image, run.log, output/}), so the unchanged
rnf213_workflow_v3.py / rnf213_review.py / batch review scripts read it directly.

Run every command from the project root (the folder mounted as /work for Docker), e.g.
~/Documents/RNF213-2DTM/rnf213-2dtm, with the Python environment that has the port and
CUDA PyTorch installed (no sudo, no Docker).

  prepare  --source tiles_3p8 --dest tiles_3p8_port [--set KEY=VALUE ...]
  run      --tiles tiles_3p8_port [--only tile_0003 ...] [--images image_0001 ...]
           [--device cuda] [--batch-size 8]
  status   --tiles tiles_3p8_port
  compare  --a tiles_3p8 --b tiles_3p8_port [--out compare_port]

--source may be a single tile tree (tiles_3p8) or a batch's searches folder
(--source batch_noE3_3/searches --dest batch_noE3_3_port/searches): every folder below it
containing tile_offsets.csv is mirrored; jobs.csv is re-pointed to the new tiles (new
search.par hashes) and the other .json/.csv metadata are copied, so
rnf213_batch_review.py --batch batch_noE3_3_port works on the port outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MARKER = "Done combining search output"
FINAL_FILES = ("particles.txt", "particles.par", "particles.mrc", "templates.mrc", "list_final.txt")


# ------------------------------------------------------------------ helpers

def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def par_value(text: str, key: str):
    for line in text.splitlines():
        f = line.split("#", 1)[0].split(None, 1)
        if len(f) == 2 and f[0] == key:
            return f[1].strip()
    return None


def set_par(text: str, changes: dict) -> str:
    lines, seen = [], set()
    for line in text.splitlines():
        f = line.split("#", 1)[0].split()
        if f and f[0] in changes:
            if f[0] in seen:
                continue
            line = f"{f[0]} {changes[f[0]]}"
            seen.add(f[0])
        lines.append(line)
    lines += [f"{k} {v}" for k, v in changes.items() if k not in seen]
    return "\n".join(lines) + "\n"


def to_project(root: Path, value: str) -> Path:
    p = root / value[len("/work/"):] if value.startswith("/work/") else Path(value)
    return p if p.is_absolute() else root / p


def like_original(root: Path, original: str, new_path: Path) -> str:
    """Write the new path in the same style as the original (/work/... or absolute)."""
    if original.startswith("/work/"):
        return "/work/" + new_path.resolve().relative_to(root.resolve()).as_posix()
    return str(new_path.resolve())


def tile_trees(src: Path):
    return sorted(p.parent for p in src.rglob("tile_offsets.csv"))


def read_offsets(tree: Path):
    with open(tree / "tile_offsets.csv", newline="") as f:
        return list(csv.DictReader(f))


def complete(tile: Path) -> bool:
    """Same rule as rnf213_batch_workflow.complete()."""
    log = tile / "run.log"
    if not log.exists():
        return False
    st = tile / "run_state.json"
    if st.exists() and json.loads(st.read_text()).get("status") == "interrupted_or_failed":
        return False
    text = log.read_text(errors="replace")
    sh = tile / "output" / "search_SH.txt"
    if MARKER not in text or not sh.is_file() or sh.stat().st_size == 0:
        return False
    if "starting optimization on" in text:
        return all((tile / "output" / n).is_file() and (tile / "output" / n).stat().st_size > 0
                   for n in FINAL_FILES)
    return "nothing to optimize" in text


def all_tiles(tiles: Path):
    out = []
    for tree in tile_trees(tiles):
        for row in read_offsets(tree):
            out.append(tree / row["tile"])
    return out


def known_keys():
    try:
        from smap.io import SEARCH_GLOBAL_DEFAULTS
    except ImportError as e:
        raise SystemExit(f"cannot import the port ({e}); activate the environment with smap installed")
    return set(SEARCH_GLOBAL_DEFAULTS)


# ------------------------------------------------------------------ commands

def prepare(root: Path, src: Path, dest: Path, sets: list[str]):
    if dest.exists():
        raise SystemExit(f"{dest} exists; choose a new --dest (nothing is overwritten)")
    keys = known_keys()
    changes = {}
    for s in sets:
        if "=" not in s:
            raise SystemExit(f"--set expects KEY=VALUE, got {s!r}")
        k, v = s.split("=", 1)
        if k not in keys:
            raise SystemExit(f"unknown .par key {k!r} (typos would be ignored silently by SMAP)")
        changes[k] = v
    trees = tile_trees(src)
    if not trees:
        raise SystemExit(f"no tile_offsets.csv below {src}")
    plan = []
    for tree in trees:
        for row in read_offsets(tree):
            old = tree / row["tile"]
            text = (old / "search.par").read_text()
            img_val, out_val = par_value(text, "imageFile"), par_value(text, "outputDir")
            if not img_val or not out_val:
                raise SystemExit(f"{old}/search.par lacks imageFile/outputDir")
            img = old / Path(img_val).name
            if not img.is_file():
                raise SystemExit(f"tile image missing: {img}")
            for key in ("modelFile", "structureFile", "PDBFile", "rotationsFile", "maskFile"):
                v = par_value(text, key)
                if v and not to_project(root, v).is_file():
                    raise SystemExit(f"{old}/search.par: {key} {v} not found at {to_project(root, v)}")
            if int(float(par_value(text, "nCores") or 1)) != 1:
                raise SystemExit(f"{old}: nCores must be 1 for the RNF213 review scripts")
            plan.append((tree, row, old, text, img, img_val, out_val))
    n_link = 0
    manifest = []
    for tree, row, old, text, img, img_val, out_val in plan:
        new_tree = dest / tree.relative_to(src)
        new = new_tree / row["tile"]
        (new / "output").mkdir(parents=True)
        if not (new_tree / "tile_offsets.csv").exists():
            shutil.copy2(tree / "tile_offsets.csv", new_tree / "tile_offsets.csv")
        try:
            os.link(img, new / img.name)
            n_link += 1
        except OSError:
            shutil.copy2(img, new / img.name)
        par = set_par(text, {"imageFile": like_original(root, img_val, new / img.name),
                             "outputDir": like_original(root, out_val, new / "output"), **changes})
        (new / "search.par").write_text(par)
        manifest.append(dict(tile=str(new.relative_to(root) if new.is_relative_to(root) else new),
                             source_tile=str(old), image_sha256=sha256(img),
                             source_par_sha256=sha256(old / "search.par"),
                             par_sha256=sha256(new / "search.par")))
    # batch layout (rnf213_batch_workflow.py): copy metadata files and re-point jobs.csv so
    # rnf213_batch_review.py --batch <dest batch> accepts the port tree
    for tree in [src] + trees:
        for f in tree.iterdir():
            if f.is_file() and f.suffix in (".json", ".csv") and f.name != "jobs.csv":
                t = dest / f.relative_to(src)
                if not t.exists():
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, t)
    if (src / "jobs.csv").exists():
        with open(src / "jobs.csv", newline="") as f:
            jobs = list(csv.DictReader(f))
        for j in jobs:
            new = dest / Path(j["directory"]).relative_to(src)
            j["directory"] = new.as_posix()
            j["configuration_sha256"] = sha256(new / "search.par")
        with open(dest / "jobs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(jobs[0]))
            w.writeheader()
            w.writerows(jobs)
    try:
        from smap.provenance import code_commit
        commit = code_commit()
    except Exception:  # noqa: BLE001
        commit = None
    (dest / "port_manifest.json").write_text(json.dumps(dict(
        created_utc=stamp(), source=str(src), options=changes, port_commit=commit,
        tiles=manifest), indent=1))
    print(f"Prepared {len(plan)} tiles in {dest} ({n_link} images hard-linked); options {changes or 'none'}")


def run(root: Path, tiles: Path, only, device, batch_size, keep_going, dry, images=None):
    todo = [t for t in all_tiles(tiles) if (not only or t.name in only)
            and (not images or t.parent.name in images)]
    if images and not todo:
        raise SystemExit(f"--images not found below {tiles}: {images}")
    if only and len({t.name for t in todo}) != len(set(only)):
        raise SystemExit(f"--only names not found: {sorted(set(only) - {t.name for t in todo})}")
    done = 0
    for i, tile in enumerate(todo, 1):
        if complete(tile):
            done += 1
            print(f"skip completed {tile} [{i}/{len(todo)}]", flush=True)
            continue
        out = tile / "output"
        if (tile / "run.log").exists() or (out.exists() and any(out.iterdir())):
            arch = tile / "attempts" / stamp()
            arch.mkdir(parents=True)
            for n in ("output", "run.log", "run_state.json"):
                if (tile / n).exists():
                    (tile / n).rename(arch / n)
            out.mkdir()
            print(f"preserved unfinished attempt in {arch}", flush=True)
        cmd = [sys.executable, "-m", "smap.search_global", str(tile / "search.par"),
               "--path-map", f"/work={root.resolve().as_posix()}", "--batch-size", str(batch_size),
               "--log", str(tile / "run.log")]
        if device:
            cmd += ["--device", device]
        print(f"start {tile} [{i}/{len(todo)}]: {' '.join(cmd)}", flush=True)
        if dry:
            continue
        state = dict(status="running", started_utc=stamp(), command=cmd)
        (tile / "run_state.json").write_text(json.dumps(state, indent=1))
        t0 = time.monotonic()
        code = subprocess.run(cmd).returncode    # progress on screen; --log copies it to run.log
        state.update(ended_utc=stamp(), elapsed_seconds=round(time.monotonic() - t0, 1), exit_code=code)
        ok = code == 0 and complete(tile)
        state["status"] = "completed" if ok else "interrupted_or_failed"
        (tile / "run_state.json").write_text(json.dumps(state, indent=1))
        if ok:
            done += 1
            print(f"completed {tile} in {state['elapsed_seconds']:.0f} s ({done}/{len(todo)})", flush=True)
        else:
            print(f"FAILED {tile} (exit {code}); see {tile / 'run.log'}", flush=True)
            if not keep_going:
                raise SystemExit(1)
    print(f"{done}/{len(todo)} tiles complete")


def status(tiles: Path):
    n = n8 = c = 0
    for tile in all_tiles(tiles):
        n += 1
        if not complete(tile):
            continue
        c += 1
        p = tile / "output" / "particles.txt"
        if p.exists():
            for line in p.read_text().splitlines():
                f = line.split()
                if len(f) == 2 and not f[0].startswith("#"):
                    n8 += float(f[1]) >= 8
    print(f"{c}/{n} tiles complete; {n8} tile entries with final SNR >= 8 before overlap removal")


def compare(a: Path, b: Path, out: Path):
    from smap.compare import compare_outputs
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tb in all_tiles(b):
        ta = a / tb.relative_to(b)
        if not (complete(ta) and complete(tb)):
            continue
        rep = compare_outputs(ta / "output", tb / "output")
        ck = rep["checks"]
        ea = json.loads((ta / "run_state.json").read_text()).get("elapsed_seconds") \
            if (ta / "run_state.json").exists() else None
        eb = json.loads((tb / "run_state.json").read_text()).get("elapsed_seconds") \
            if (tb / "run_state.json").exists() else None
        rows.append(dict(tile=str(tb.relative_to(b)), n_a=rep["particles"]["n_reference"],
                         n_b=rep["particles"]["n_test"], n_paired=rep["particles"]["n_paired"],
                         ge8_a=ck["count_final_ge_8"]["value"][0], ge8_b=ck["count_final_ge_8"]["value"][1],
                         max_abs_d_score_final=ck["d_score_final"]["max_abs"],
                         max_angle_final_deg=ck["angle_final_deg"]["max_abs"],
                         search_mean_rel=ck.get("search_mean.mrc", {}).get("relative"),
                         all_within_tolerance=rep["all_ok"], seconds_a=ea, seconds_b=eb))
        (out / f"{rows[-1]['tile'].replace('/', '__')}.json").write_text(json.dumps(rep, indent=1))
    if not rows:
        raise SystemExit("no tile completed in both trees")
    with open(out / "per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sa = sum(r["ge8_a"] for r in rows)
    sb = sum(r["ge8_b"] for r in rows)
    ok = sum(bool(r["all_within_tolerance"]) for r in rows)
    print(f"{len(rows)} tiles compared; within smap.compare tolerances: {ok}; "
          f"entries >= 8 (before overlap removal): {a.name} {sa}, {b.name} {sb}; "
          f"details in {out / 'per_tile.csv'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="extra .par option, e.g. refine_continuous=1 or decoy_label=NAME")
    r = sub.add_parser("run")
    r.add_argument("--tiles", required=True)
    r.add_argument("--only", nargs="+", default=None, help="tile names, e.g. tile_0003")
    r.add_argument("--images", nargs="+", default=None,
                   help="batch layout: image folders to run, e.g. image_0001 image_0002")
    r.add_argument("--device", default=None)
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--keep-going", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="print the commands only")
    s = sub.add_parser("status")
    s.add_argument("--tiles", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="reference tree (v2.1)")
    c.add_argument("--b", required=True, help="test tree (port)")
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path.cwd()
    for k in ("source", "dest", "tiles", "a", "b"):
        v = getattr(a, k, None)
        if v and (Path(v).is_absolute() or ".." in Path(v).parts):
            raise SystemExit(f"use project-relative paths (run from the project root): {v}")
    if a.cmd == "prepare":
        prepare(root, Path(a.source), Path(a.dest), a.set)
    elif a.cmd == "run":
        run(root, Path(a.tiles), a.only, a.device, a.batch_size, a.keep_going, a.dry_run, a.images)
    elif a.cmd == "status":
        status(Path(a.tiles))
    else:
        compare(Path(a.a), Path(a.b), Path(a.out or f"{a.b}/review/compare_vs_{Path(a.a).name}"))


if __name__ == "__main__":
    main()
