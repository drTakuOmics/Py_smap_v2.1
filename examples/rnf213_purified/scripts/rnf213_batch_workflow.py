#!/usr/bin/env python3
"""Prepare and run the RNF213 domain-omission anchor batch sequentially.

prepare: use the project venv (numpy/mrcfile required); creates cropped images,
per-image CTF configurations, offsets and input fingerprints. No searches.
run: standard-library-only, intended for sudo /usr/bin/python3; starts one GPU
container at a time, skips verified completed jobs and preserves failed attempts.
status: standard-library-only progress and tile-entry counts before deduplication.

Run from the project directory. Existing pilot/review files are preserved.
"""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

DOCKER = "/usr/bin/docker"
MARKER = "Done combining search output"
SIZE, STRIDE = 1024, 640


def digest(path):
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_stamp():
    """UTC time stamp for folder names, to the microsecond."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def parameters(path):
    """{key: value} of a search.par ('#' starts a comment; a repeated key is an error)."""
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split("#", 1)[0].split(maxsplit=1)
        if len(fields) == 2:
            if fields[0] in result:
                raise ValueError(f"Duplicate parameter {fields[0]} in {path}.")
            result[fields[0]] = fields[1]
    return result


def replace_parameters(text, replacements):
    """The .par text with the given keys replaced (missing ones appended) and any
    rotationsFile line dropped, so the grid is generated from angle_inc."""
    lines, seen = [], set()
    for line in text.splitlines():
        fields = line.split("#", 1)[0].split()
        key = fields[0] if fields else ""
        if key == "rotationsFile":
            continue  # Generate the inherited 3.8-degree grid rather than a stale rotation cache.
        if key in replacements:
            line = key + " " + replacements[key]
            seen.add(key)
        lines.append(line)
    lines.extend(k + " " + v for k, v in replacements.items() if k not in seen)
    return "\n".join(lines) + "\n"


def project_path(root, value):
    """A '/work/...' or relative .par path as an absolute path inside the project (error outside it)."""
    path = root / value[6:] if str(value).startswith("/work/") else Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    path.relative_to(root)
    return path


def docker_path(root, path):
    """An absolute project path as the '/work/...' path seen inside the Docker container."""
    relative = Path(path).resolve().relative_to(root).as_posix()
    if any(c.isspace() for c in relative):
        raise ValueError("Project-relative search paths must not contain whitespace.")
    return "/work/" + relative


def write_csv(path, rows):
    """Write rows (dicts) as CSV; the keys of the first row are the header."""
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    """Rows of a CSV file."""
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, data):
    """Write JSON through a temporary file, so a crash never leaves a half-written file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def starts(length):
    """Start offsets of SIZE-px tiles every STRIDE px along one dimension; the last tile ends at the edge."""
    if length < SIZE:
        raise ValueError("Micrograph dimension is smaller than the tile size.")
    values = list(range(0, length - SIZE + 1, STRIDE))
    if values[-1] != length - SIZE:
        values.append(length - SIZE)
    return values


def prepare(root, batch, anchor):
    """'prepare': check the selection, the anchor pilot and its potential, crop every micrograph
    into tiles and write one search.par per tile, the offsets and the input fingerprints.
    No search is started."""
    import numpy as np
    import mrcfile
    import rnf213_batch_plan as planner
    import rnf213_batch_download as downloader

    destination = batch / "searches"
    if destination.exists():
        raise FileExistsError(f"{destination} already exists. Use run to resume the prepared jobs.")
    rows = downloader.selection(batch)
    report = json.loads((batch / "download_report.json").read_text())
    downloaded = {r["batch_image"]: r for r in report["images"]}
    loops = planner.read_loops(batch / "selected_micrographs_ctf.star")
    optics_loops = [
        r for _, labels, r in loops if "_rlnOpticsGroup" in labels and "_rlnMicrographName" not in labels
    ]
    optics = {r["_rlnOpticsGroup"]: r for r in optics_loops[0]} if len(optics_loops) == 1 else {}
    if len(optics_loops) > 1:
        raise ValueError("Ambiguous optics metadata.")
    micro_loops = [r for _, labels, r in loops if "_rlnMicrographName" in labels]
    if len(micro_loops) != 1:
        raise ValueError("Expected one selected micrograph STAR loop.")
    star_rows = {Path(r["_rlnMicrographName"]).name: r for r in micro_loops[0]}
    source_text = (anchor / "search.par").read_text()
    source = parameters(anchor / "search.par")
    if source.get("function") != "search_global" or source.get("nCores") != "1":
        raise ValueError("Use the completed, single-worker anchor pilot.")
    if not all(math.isclose(float(source[k]), 3.8) for k in ("angle_inc", "psi_inc")):
        raise ValueError("Both anchor angular increments must be 3.8 degrees.")
    if MARKER not in (anchor / "run.log").read_text():
        raise ValueError("Anchor pilot is not completed.")
    manifest = json.loads((anchor / "manifest.json").read_text())
    if (
        "domain-omission" not in manifest.get("template_type", "")
        or manifest.get("omitted_E3_author_residues") != [3589, 4926]
        or manifest.get("omitted_CTD_author_residues") != [4927, 5148]
        or not manifest.get("excluded_other_chains")
    ):
        raise ValueError("Expected the validated E3/CTD-omitted, chain-A-only anchor.")
    cif = anchor / manifest["fragment_CIF"]
    model = anchor / "output" / (cif.stem + "_SP.mrc")
    if "anchor_noE3_noCTD" not in model.name:
        raise ValueError("Unexpected anchor SP model name.")
    apix = float(source["aPerPix"])
    with mrcfile.mmap(model) as m:
        if m.data.ndim != 3 or len(set(m.data.shape)) != 1 or m.data.shape[0] % 2:
            raise ValueError("Expected an even, cubic anchor SP model.")
        if not np.allclose([m.voxel_size.x, m.voxel_size.y, m.voxel_size.z], apix, atol=1e-5):
            raise ValueError("Anchor SP voxel size differs from the working configuration.")
    model_hash = digest(model)
    inputs = []
    for row in rows:
        image = batch / "micrographs" / row["basename"]
        record = downloaded[row["batch_image"]]
        info = downloader.validate_image(image, float(row["pixel_size_A"]))
        if info["sha256"] != record["sha256"]:
            raise ValueError(f"Downloaded image changed: {image}.")
        if not math.isclose(float(row["pixel_size_A"]), apix, abs_tol=1e-5):
            raise ValueError("Batch and anchor pixel sizes differ.")
        sr = star_rows[row["basename"]]
        for sk, ck in [
            ("_rlnDefocusU", "defocus_u_A"),
            ("_rlnDefocusV", "defocus_v_A"),
            ("_rlnDefocusAngle", "astigmatism_angle_degrees"),
        ]:
            if not math.isclose(float(sr[sk]), float(row[ck]), abs_tol=1e-5):
                raise ValueError("CSV and selected STAR CTF parameters differ.")
        opt = optics.get(sr.get("_rlnOpticsGroup"), {})
        if opt:
            if not math.isclose(float(opt["_rlnVoltage"]) * 1000, float(source["V_acc"]), abs_tol=1):
                raise ValueError("Accelerating voltages differ.")
            if not math.isclose(
                float(opt["_rlnSphericalAberration"]) * 0.001, float(source["Cs"]), abs_tol=1e-8
            ):
                raise ValueError("Spherical aberration values differ.")
        inputs.append((row, image, info))
    jobs = []
    with tempfile.TemporaryDirectory(prefix=".anchor-batch-", dir=batch) as temp:
        stage = Path(temp) / "searches"
        stage.mkdir()
        for row, image, info in inputs:
            image_dir = stage / row["batch_image"]
            image_dir.mkdir()
            offsets = []
            with mrcfile.mmap(image) as micrograph:
                h, w = micrograph.data.shape
                index = 0
                for y0 in starts(h):
                    for x0 in starts(w):
                        index += 1
                        tile_name = f"tile_{index:04d}"
                        tile = image_dir / tile_name
                        tile.mkdir()
                        (tile / "output").mkdir()
                        final_tile = destination / row["batch_image"] / tile_name
                        tile_file = tile / f"tile_RNF213_{index:04d}.mrc"
                        with mrcfile.new(tile_file) as m:
                            m.set_data(
                                np.array(
                                    micrograph.data[y0 : y0 + SIZE, x0 : x0 + SIZE],
                                    dtype=np.float32,
                                    copy=True,
                                )
                            )
                            m.voxel_size = apix
                        df = " ".join(
                            f"{float(row[k]):.6f}"
                            for k in ("defocus_u_A", "defocus_v_A", "astigmatism_angle_degrees")
                        )
                        config = replace_parameters(
                            source_text,
                            dict(
                                imageFile=docker_path(root, final_tile / tile_file.name),
                                modelFile=docker_path(root, model),
                                structureFile=docker_path(root, model),
                                outputDir=docker_path(root, final_tile / "output"),
                                defocus=df,
                            ),
                        )
                        (tile / "search.par").write_text(config)
                        offsets.append(
                            dict(tile=tile_name, x0_zero_based=x0, y0_zero_based=y0, width=SIZE, height=SIZE)
                        )
                        jobs.append(
                            dict(
                                batch_image=row["batch_image"],
                                tile=tile_name,
                                directory=final_tile.relative_to(root).as_posix(),
                                configuration_sha256=digest(tile / "search.par"),
                                tile_file=tile_file.name,
                                tile_sha256=digest(tile_file),
                            )
                        )
            write_csv(image_dir / "tile_offsets.csv", offsets)
            write_json(
                image_dir / "micrograph_manifest.json",
                dict(
                    source_micrograph=str(image),
                    source_micrograph_name=row["micrograph_name"],
                    source_micrograph_sha256=info["sha256"],
                    shape_yx=info["shape_yx"],
                    pixel_size_A=apix,
                    ctf=row,
                    tiles=len(offsets),
                    coordinate_convention="Zero-based original pixel centers; offsets in tile_offsets.csv",
                    anchor_SP=str(model),
                    anchor_SP_sha256=model_hash,
                    anchor_CIF=str(cif),
                    anchor_CIF_sha256=digest(cif),
                    detection_SNR_cutoff=8,
                    biological_target="Held-out E3/CTD/E2 density",
                ),
            )
            print(
                f"Prepared {row['batch_image']}: {len(offsets)} tiles; defocus {df} (A, A, degrees)",
                flush=True,
            )
        write_csv(stage / "jobs.csv", jobs)
        write_json(
            stage / "batch_search.json",
            dict(
                project=str(root),
                batch=str(batch),
                angular_spacing_degrees=3.8,
                source_anchor=str(anchor),
                source_parameters_sha256=digest(anchor / "search.par"),
                anchor_SP=str(model),
                anchor_SP_sha256=model_hash,
                tile_size=SIZE,
                nominal_overlap=SIZE - STRIDE,
                total_images=len(inputs),
                total_tiles=len(jobs),
                final_SNR_cutoff=8,
                nCores=1,
                source_anchor_manifest=manifest,
            ),
        )
        stage.rename(destination)
    print(f"Prepared {len(jobs)} tiles across {len(inputs)} images. No searches started.")


def complete(tile):
    """True when a tile's search finished: completion line in run.log, a non-empty search_SH.txt
    and the refinement outputs (or 'nothing to optimize'); a failed attempt is never complete."""
    state = tile / "run_state.json"
    if state.exists() and json.loads(state.read_text()).get("status") == "interrupted_or_failed":
        return False
    log = tile / "run.log"
    if not log.exists():
        return False
    text = log.read_text(errors="replace")
    sh = tile / "output/search_SH.txt"
    if MARKER not in text or not sh.is_file() or sh.stat().st_size == 0:
        return False
    if "starting optimization on" in text:
        return all(
            (tile / "output" / name).is_file() and (tile / "output" / name).stat().st_size > 0
            for name in ("particles.txt", "particles.par", "particles.mrc", "templates.mrc", "list_final.txt")
        )
    return "nothing to optimize" in text


def preserve_attempt(tile):
    """Move the output and log of an unfinished attempt into attempts/<time stamp>/."""
    output, log = tile / "output", tile / "run.log"
    if log.exists() or (output.exists() and any(output.iterdir())):
        archive = tile / "attempts" / utc_stamp()
        archive.mkdir(parents=True)
        if output.exists():
            output.rename(archive / "output")
        if log.exists():
            log.rename(archive / "run.log")
        if (tile / "run_state.json").exists():
            (tile / "run_state.json").rename(archive / "run_state.json")
        print(f"Preserved unfinished attempt: {archive}", flush=True)
    output.mkdir(exist_ok=True)


def selected_jobs(root, batch, image, all_images):
    """Prepared jobs of one image (or all), after checking that no search.par changed since prepare."""
    jobs = read_csv(batch / "searches/jobs.csv")
    selected = jobs if all_images else [j for j in jobs if j["batch_image"] == image]
    if not selected:
        raise ValueError(f"No prepared jobs for {image}.")
    for job in selected:
        tile = project_path(root, job["directory"])
        if digest(tile / "search.par") != job["configuration_sha256"]:
            raise ValueError(f"Prepared configuration changed: {tile}.")
    return selected


def interrupt_handler(signum, frame):
    """Turn SIGTERM into KeyboardInterrupt, so an interrupted tile is recorded as such."""
    raise KeyboardInterrupt()


def run(root, batch, image, all_images):
    """'run': search the selected tiles one at a time in the SMAP v2.1 Docker image (needs root),
    skipping completed tiles and archiving failed attempts."""
    if os.geteuid() != 0:
        raise PermissionError(
            "Run this subcommand with sudo /usr/bin/python3 to avoid repeated sudo prompts."
        )
    details = json.loads((batch / "searches/batch_search.json").read_text())
    if Path(details["project"]) != root or digest(details["anchor_SP"]) != details["anchor_SP_sha256"]:
        raise ValueError("Project location or shared anchor SP has changed; prepare inputs again.")
    jobs = selected_jobs(root, batch, image, all_images)
    subprocess.run([DOCKER, "info", "--format", "{{.ServerVersion}}"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([DOCKER, "image", "inspect", "smap:2.1"], check=True, stdout=subprocess.DEVNULL)
    lock = (batch / "searches/.runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another runner is active for this batch.")
    signal.signal(signal.SIGTERM, interrupt_handler)
    completed = 0
    for number, job in enumerate(jobs, 1):
        tile = project_path(root, job["directory"])
        if digest(tile / job["tile_file"]) != job["tile_sha256"]:
            raise ValueError(f"Prepared tile image changed: {tile}.")
        name = (
            "rnf213-anchor-"
            + hashlib.sha256(str(batch).encode()).hexdigest()[:8]
            + "-"
            + job["batch_image"]
            + "-"
            + job["tile"]
        )
        existing = subprocess.run(
            [DOCKER, "container", "inspect", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if existing.returncode == 0:
            raise RuntimeError(
                f"Container {name} already exists; inspect or stop the previous job before resuming."
            )
        if complete(tile):
            completed += 1
            print(f"Skip completed {job['batch_image']}/{job['tile']} [{number}/{len(jobs)}]", flush=True)
            continue
        preserve_attempt(tile)
        state = dict(
            status="running",
            started_UTC=utc_stamp(),
            container=name,
            configuration_sha256=job["configuration_sha256"],
            tile_sha256=job["tile_sha256"],
        )
        write_json(tile / "run_state.json", state)
        command = [
            DOCKER,
            "run",
            "--rm",
            "--gpus",
            "all",
            "--name",
            name,
            "-v",
            str(root) + ":/work",
            "--entrypoint",
            "/opt/smap/smappoi_search_global",
            "smap:2.1",
            docker_path(root, tile / "search.par"),
            "1",
        ]
        print(f"Start {job['batch_image']}/{job['tile']} [{number}/{len(jobs)}]", flush=True)
        started = time.monotonic()
        process = None
        try:
            with (tile / "run.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    start_new_session=True,
                )
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                code = process.wait()
            if code != 0 or not complete(tile):
                raise RuntimeError(f"SMAP job failed or lacks complete output: {tile} (exit {code}).")
        except BaseException:
            if process is not None:
                subprocess.run(
                    [DOCKER, "stop", "--time", "10", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            state.update(status="interrupted_or_failed", ended_UTC=utc_stamp())
            write_json(tile / "run_state.json", state)
            raise
        completed += 1
        state.update(status="completed", ended_UTC=utc_stamp(), elapsed_seconds=time.monotonic() - started)
        write_json(tile / "run_state.json", state)
        print(
            f"Completed {job['batch_image']}/{job['tile']}; {completed}/{len(jobs)} selected jobs complete.",
            flush=True,
        )
    print("All selected tile searches completed. Final SNR filtering and overlap removal remain for review.")


def status(root, batch):
    """'status': per image, completed tiles and tile entries with final score >= 8 (before
    de-duplication)."""
    jobs = read_csv(batch / "searches/jobs.csv")
    for image in sorted({j["batch_image"] for j in jobs}):
        selected = [j for j in jobs if j["batch_image"] == image]
        finished = [
            project_path(root, j["directory"])
            for j in selected
            if complete(project_path(root, j["directory"]))
        ]
        count = 0
        for tile in finished:
            scores = tile / "output/particles.txt"
            if scores.exists():
                for line in scores.read_text().splitlines():
                    fields = line.split()
                    if len(fields) == 2 and not fields[0].startswith("#"):
                        count += float(fields[1]) >= 8
        print(
            f"{image}: {len(finished)}/{len(selected)} tiles complete; {count} tile entries at SNR >= 8 before overlap removal."
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "status"):
        command = sub.add_parser(name)
        command.add_argument("--batch", default="batch_noE3_3")
        if name == "prepare":
            command.add_argument("--anchor", default="anchor_noE3_pilot")
        if name == "run":
            group = command.add_mutually_exclusive_group(required=True)
            group.add_argument("--image", help="One prepared image ID, e.g. image_0001.")
            group.add_argument("--all-images", action="store_true")
    args = p.parse_args()
    root = Path.cwd().resolve()
    batch = project_path(root, args.batch)
    if args.command == "prepare":
        prepare(root, batch, project_path(root, args.anchor))
    elif args.command == "run":
        run(root, batch, args.image, args.all_images)
    else:
        status(root, batch)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run the same command to resume; completed tiles will be skipped.", file=sys.stderr)
        sys.exit(130)
