#!/usr/bin/env python3
"""Download only the processed EMPIAR-10711 images in a saved batch selection.

Run from the RNF213 project directory. Uses curl with resumable .part downloads,
validates each complete MRC, and records local SHA256 fingerprints. No tiling,
Docker calls, GPU searches, or modifications of the saved selection.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import quote

import mrcfile
import numpy as np


ARCHIVE = "https://ftp.ebi.ac.uk/empiar/world_availability/10711/data/micrographs_ctf_motioncorr/"


def validate_image(path, apix):
    """Check a downloaded micrograph (size, data type, axis order, file length, pixel size,
    finite and non-constant data) and return its description with the SHA-256."""
    with mrcfile.mmap(path, mode="r", permissive=False) as m:
        shape = tuple(int(v) for v in m.data.shape)
        if shape != (4092, 5760) or int(m.header.mode) != 2:
            raise ValueError(f"{path}: expected a 4092 x 5760 float32 processed image; got {shape}.")
        if tuple(int(m.header[k]) for k in ("mapc", "mapr", "maps")) != (1, 2, 3):
            raise ValueError(f"{path}: nonstandard MRC axis order.")
        expected = 1024 + int(m.header.nsymbt) + m.data.size * m.data.dtype.itemsize
        if path.stat().st_size != expected:
            raise ValueError(f"{path}: file length disagrees with its MRC header.")
        voxel = [float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)]
        if not np.isfinite(voxel).all() or any(v < 0 for v in voxel):
            raise ValueError(f"{path}: invalid header voxel size.")
        if any(v > 0 and not math.isclose(v, apix, abs_tol=1e-4) for v in voxel[:2]):
            raise ValueError(f"{path}: header pixel size differs from the selected STAR metadata ({apix} A).")
        if not np.isfinite(m.data).all() or float(np.std(m.data, dtype=np.float64)) <= 0:
            raise ValueError(f"{path}: nonfinite or constant image data.")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return dict(
        shape_yx=list(shape),
        bytes=expected,
        sha256=h.hexdigest(),
        header_voxel_size_A=voxel,
        pixel_size_A=apix,
    )


def selection(batch):
    """Rows of selected_micrographs.csv, after checking names, image ids and pixel size."""
    with (batch / "selected_micrographs.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("The batch selection is empty.")
    seen_names, seen_ids = set(), set()
    for row in rows:
        name, index = row["basename"], row["batch_image"]
        if not re.fullmatch(r"FoilHole_[0-9]+_Data_[0-9]+_[0-9]+_[0-9]{8}_[0-9]{6}_fractions\.mrc", name):
            raise ValueError(f"Unexpected processed micrograph basename: {name!r}.")
        if not re.fullmatch(r"image_[0-9]{4}", index) or name in seen_names or index in seen_ids:
            raise ValueError("Duplicate micrographs or invalid batch image IDs.")
        apix = float(row["pixel_size_A"])
        if not math.isfinite(apix) or not math.isclose(apix, 1.072, abs_tol=1e-4):
            raise ValueError("This downloader expects the established 1.072 A/pixel processed images.")
        seen_names.add(name)
        seen_ids.add(index)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", default="batch_noE3_3")
    p.add_argument(
        "--check-only", action="store_true", help="Validate existing downloads without network access."
    )
    args = p.parse_args()
    batch = Path(args.batch).resolve()
    rows = selection(batch)
    curl = shutil.which("curl")
    if not args.check_only and not curl:
        p.error("curl is required for downloading.")
    folder = batch / "micrographs"
    folder.mkdir(exist_ok=True)
    results = []
    for row in rows:
        name = row["basename"]
        apix = float(row["pixel_size_A"])
        target = folder / name
        url = ARCHIVE + quote(name, safe="")
        if not target.exists():
            if args.check_only:
                raise FileNotFoundError(target)
            part = target.with_name(target.name + ".part")
            complete = False
            if part.exists():
                try:
                    validate_image(part, apix)
                    complete = True
                except (ValueError, OSError, EOFError):
                    pass  # Incomplete .part file is resumed by curl.
            if not complete:
                print(f"Downloading {row['batch_image']}: {name}", flush=True)
                subprocess.run(
                    [
                        curl,
                        "--fail",
                        "--location",
                        "--show-error",
                        "--retry",
                        "3",
                        "--connect-timeout",
                        "30",
                        "--max-time",
                        "1800",
                        "--continue-at",
                        "-",
                        "--output",
                        str(part),
                        url,
                    ],
                    check=True,
                )
                validate_image(part, apix)
            part.rename(target)
        info = validate_image(target, apix)
        results.append(
            dict(
                batch_image=row["batch_image"], basename=name, local_path=str(target), source_url=url, **info
            )
        )
        print(
            f"Verified {row['batch_image']}: {info['shape_yx']}, "
            f"{info['bytes'] / 1024**2:.1f} MiB, {apix:g} A/pixel from STAR",
            flush=True,
        )
        if not all(v > 0 for v in info["header_voxel_size_A"][:2]):
            print("  Header pixel size missing; selected STAR metadata supplies the calibration.", flush=True)
    report = batch / "download_report.json"
    temporary = report.with_name(report.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(status="All selected processed images validated", images=results), indent=2) + "\n"
    )
    temporary.replace(report)
    print(f"Images ready: {len(results)}/{len(rows)}")
    print(f"Download folder: {folder}")
    print("No tiles prepared and no GPU searches started.")


if __name__ == "__main__":
    main()
