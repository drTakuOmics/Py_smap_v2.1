#!/usr/bin/env python3
"""Prepare a small, traceable RNF213 SMAP v2.1 pilot from one processed micrograph.

Install: python -m pip install numpy mrcfile

Example (paths are local to your Linux machine):
  python rnf213_smap_pilot.py prepare --micrograph /data/one.mrc \
    --star /data/micrographs_ctf.star --model /data/7oik.cif \
    --out /data/rnf213-pilot --center-x 2500 --center-y 1800 --size 1024 \
    --apix 1.072

Use the pixel size of the *processed MRC*, verified against its header/STAR;
0.536 A/px is the archive's raw movie pixel spacing, not necessarily this MRC.
The output search.par uses container paths /work/... . Run with:
  docker build -t smap:latest /path/to/smap_v2.1
  docker run --rm -it --gpus all -v /data/rnf213-pilot:/work \
    --entrypoint /bin/bash smap:latest
  cd /opt/smap && ./smap_run.sh /work/search.par

The SMAP launcher runs in the background. Keep the container shell open and
inspect /work/output/search.log and list_final.txt. On early exit, inspect
the logs before changing parameters. Never regard a coarse-grid hit as a
validated orientation without a denser follow-up search and independent check.
"""

import argparse
import json
import math
import shutil
from pathlib import Path


def read_star_loop(path, filename):
    """Find the row in a RELION 3.1 micrographs STAR loop matching a basename."""
    lines = Path(path).read_text().splitlines()
    columns = []
    matches = []
    in_loop = False
    for raw in lines:
        line = raw.strip()
        if line == "loop_":
            columns, in_loop = [], True
            continue
        if not line or line.startswith("#"):
            continue
        if line.startswith("data_"):
            columns, in_loop = [], False
            continue
        if line.startswith("_"):
            if in_loop:
                columns.append(line.split()[0])
            continue
        if in_loop and columns and "_rlnMicrographName" in columns:
            fields = line.split()
            if len(fields) < len(columns):
                continue
            row = dict(zip(columns, fields))
            if Path(row["_rlnMicrographName"]).name == filename:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected one STAR row for {filename}; found {len(matches)}")
    return matches[0]


def get_defocus(args, image_name):
    """Defocus U, V and angle from the STAR row of the image, or from the explicit options
    (one or the other, not both)."""
    if args.star:
        row = read_star_loop(args.star, image_name)
        needed = ["_rlnDefocusU", "_rlnDefocusV", "_rlnDefocusAngle"]
        missing = [k for k in needed if k not in row]
        if missing:
            raise ValueError(f"Missing STAR fields: {', '.join(missing)}")
        values = [float(row[k]) for k in needed]
        if args.defocus_u is not None:
            raise ValueError("Use --star OR explicit defocus fields, not both")
        return values, row
    if any(v is None for v in (args.defocus_u, args.defocus_v, args.defocus_angle)):
        raise ValueError("Provide --star or all three --defocus-u/v/angle values")
    return [args.defocus_u, args.defocus_v, args.defocus_angle], {}


def prepare(args):
    """'prepare': validate the inputs, cut the tile from the micrograph and write the pilot
    folder (tile image, search.par and a record of every setting)."""
    import mrcfile
    import numpy as np

    micrograph, model, out = Path(args.micrograph), Path(args.model), Path(args.out)
    if not micrograph.is_file() or not model.is_file():
        raise FileNotFoundError("The processed micrograph and atomic model must exist")
    if model.suffix.lower() not in (".pdb", ".cif"):
        raise ValueError("Use an atomic .pdb or .cif model (7OIK is a good first choice)")
    if args.size < 256 or args.size % 2:
        raise ValueError("--size must be even and at least 256 pixels")
    if args.apix <= 0 or args.search_apix < args.apix:
        raise ValueError("Pixel sizes must be positive; search pixel size >= input pixel size")
    if args.angle_inc <= 0 or args.df_inc <= 0 or args.thickness < 0:
        raise ValueError("Angle/defocus steps must be positive; thickness nonnegative")
    if args.margin >= args.size // 3:
        raise ValueError("Margin is too large for this crop")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out}")

    defocus, row = get_defocus(args, micrograph.name)
    if not all(math.isfinite(x) for x in defocus):
        raise ValueError("Defocus values must be finite")
    if not all(0 < x < 100000 for x in defocus[:2]):
        raise ValueError("SMAP expects positive underfocus magnitudes in angstroms")

    with mrcfile.open(micrograph, permissive=False, mode="r") as src:
        if src.data.ndim != 2:
            raise ValueError("SMAP requires a motion-corrected single-frame 2D MRC")
        height, width = src.data.shape
        x0, y0 = args.center_x - args.size // 2, args.center_y - args.size // 2
        if min(x0, y0) < 0 or x0 + args.size > width or y0 + args.size > height:
            raise ValueError(f"Crop exceeds the {width}x{height} micrograph")
        header_apix = float(src.voxel_size.x)
        if header_apix > 0 and abs(header_apix - args.apix) / args.apix > 0.01:
            raise ValueError(f"Pixel size conflict: MRC {header_apix:.4f}, --apix {args.apix:.4f}")
        crop = np.array(src.data[y0 : y0 + args.size, x0 : x0 + args.size], dtype=np.float32)
    if not np.isfinite(crop).all() or np.std(crop) == 0:
        raise ValueError("Crop has invalid pixels or no variance")

    out.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(out / "tile.mrc", overwrite=False) as dst:
        dst.set_data(crop)
        dst.voxel_size = args.apix
    shutil.copy2(model, out / model.name)
    (out / "output").mkdir()
    lines = [
        "function search_global",
        "nCores 1",
        "imageFile /work/tile.mrc",
        f"modelFile /work/{model.name}",
        "bFactor 0",
        "outputDir /work/output",
        f"aPerPix {args.apix:g}",
        f"defocus {defocus[0]:.3f} {defocus[1]:.3f} {defocus[2]:.3f}",
        f"aPerPix_search {args.search_apix:g}",
        f"angle_inc {args.angle_inc:g}",
        f"T_sample {args.thickness:g}",
        f"df_inc {args.df_inc:g}",
        f"V_acc {args.voltage_kv * 1000:g}",
        f"Cs {args.cs_mm / 1000:g}",
        "Cc 0.0027",
        "deltaE 0.7",
        "a_i 0.000050",
        "optThr 7.0",
        "qThr 10",
        "dThr 10",
        "range_degrees 2.0",
        "inc_degrees 0.5",
        "arbThr 6.0",
        "keep_scratch_flag 0",
        f"margin_pix {args.margin}",
    ]
    (out / "search.par").write_text("\n".join(lines) + "\n")
    manifest = dict(
        source_micrograph=str(micrograph.resolve()),
        model=str(model.resolve()),
        crop_origin_xy=[x0, y0],
        crop_center_xy=[args.center_x, args.center_y],
        crop_shape_yx=[args.size, args.size],
        source_shape_yx=[height, width],
        pixel_size_angstrom=args.apix,
        mrc_header_pixel_size_angstrom=header_apix,
        defocus_angstrom_angular_degrees=defocus,
        star_row=row,
        coordinate_note="original x,y = SMAP tile x,y + crop_origin_xy (check SMAP origin convention)",
        caveat="Coarse orientation search is for timing and candidate recovery only.",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {out.resolve()} | crop ({x0},{y0}) {args.size}x{args.size} | apix {args.apix:g}")
    print("Run SMAP within the GPU-enabled Docker container using /work/search.par")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare", help="Crop one MRC and write SMAP configuration")
    for name in ("micrograph", "model", "out"):
        s.add_argument(f"--{name}", required=True)
    s.add_argument("--star", help="RELION micrographs_ctf.star with this image's CTF row")
    for name in ("defocus-u", "defocus-v", "defocus-angle"):
        s.add_argument(f"--{name}", type=float)
    s.add_argument("--center-x", type=int, required=True, help="0-based x pixel in the MRC")
    s.add_argument("--center-y", type=int, required=True, help="0-based y pixel in the MRC")
    s.add_argument("--size", type=int, default=1024)
    s.add_argument("--margin", type=int, default=96)
    s.add_argument("--apix", type=float, required=True, help="Processed MRC Angstrom/pixel")
    s.add_argument("--search-apix", type=float, default=1.5)
    s.add_argument("--angle-inc", type=float, default=5.0, help="Coarse pilot only; refine later")
    s.add_argument("--thickness", type=float, default=50, help="nm; measured/estimated ice thickness")
    s.add_argument("--df-inc", type=float, default=50, help="nm defocus sampling step")
    s.add_argument("--voltage-kv", type=float, default=300)
    s.add_argument("--cs-mm", type=float, default=2.7)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)


if __name__ == "__main__":
    main()
