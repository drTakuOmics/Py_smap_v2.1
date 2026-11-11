#!/usr/bin/env python3
"""Select a small, reproducible RNF213 micrograph batch without launching searches.

Uses the existing micrographs_ctf.star, retains its optics metadata and chooses
images across the defocus range. This is a planning step: no downloading,
cropping, Docker calls or changes to previous results. Python standard library.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import tempfile


def read_loops(path):
    """Every loop of a RELION STAR file as (block name, labels, rows)."""
    lexer = shlex.shlex(Path(path).read_text(), posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    tokens = list(lexer)
    if any("\n" in t or t.startswith(";") for t in tokens):
        raise ValueError("This planner supports ordinary RELION loop STAR files, not multiline fields.")
    loops, block, i = [], "", 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("data_"):
            block = token
            i += 1
        elif token == "loop_":
            i += 1
            labels = []
            while i < len(tokens) and tokens[i].startswith("_"):
                labels.append(tokens[i])
                i += 1
            if not labels:
                raise ValueError("STAR loop has no columns.")
            values = []
            while i < len(tokens):
                t = tokens[i]
                if len(values) % len(labels) == 0 and (
                    t.startswith("data_") or t.startswith("_") or t in ("loop_", "stop_")
                ):
                    break
                values.append(t)
                i += 1
            if len(values) % len(labels):
                raise ValueError(f"Incomplete STAR row in {block}.")
            rows = [
                dict(zip(labels, values[j : j + len(labels)])) for j in range(0, len(values), len(labels))
            ]
            loops.append((block, labels, rows))
        elif token.startswith("_"):
            i += 2  # Scalar tags are not needed by this RELION loop planner.
        else:
            i += 1
    return loops


def numeric(row, key, fallback=None):
    """A finite number from a STAR row (the fallback, if given, when it is missing or invalid)."""
    try:
        value = float(row[key])
    except (ValueError, KeyError):
        if fallback is not None:
            return fallback
        raise ValueError(f"Missing or invalid {key} for {row.get('_rlnMicrographName', 'optics')}.")
    if not math.isfinite(value):
        if fallback is not None:
            return fallback
        raise ValueError(f"Non-finite {key}.")
    return value


def star_value(value):
    """A value quoted as STAR needs it (empty, white space, or a leading '#' or '_')."""
    if not value or any(c.isspace() for c in value) or value[0] in "#_":
        if "'" not in value:
            return "'" + value + "'"
        if '"' not in value:
            return '"' + value + '"'
        raise ValueError("STAR value requires unsupported multiline quoting.")
    return value


def write_loops(path, loops):
    """Write loops as a STAR file."""
    lines = ["# RELION micrograph batch subset; original optics and CTF rows retained."]
    for block, labels, rows in loops:
        lines.extend(["", block, "", "loop_"])
        lines.extend(f"{label} #{i}" for i, label in enumerate(labels, 1))
        lines.extend(" ".join(star_value(row[label]) for label in labels) for row in rows)
    Path(path).write_text("\n".join(lines) + "\n")


def choose_rows(rows, count):
    """count micrographs: equal-size strata of mean defocus, and within each the best CTF fit."""
    # Equal-count strata by mean defocus; best reported CTF fit within each.
    # A controlled quality-screened pilot, not an unbiased population sample.
    ordered = sorted(
        rows,
        key=lambda r: (
            (numeric(r, "_rlnDefocusU") + numeric(r, "_rlnDefocusV")) / 2,
            r["_rlnMicrographName"],
        ),
    )
    if len(ordered) < count:
        raise ValueError(f"Only {len(ordered)} eligible images; requested {count}.")
    selected = []
    for i in range(count):
        group = ordered[len(ordered) * i // count : len(ordered) * (i + 1) // count]
        selected.append(
            min(group, key=lambda r: (numeric(r, "_rlnCtfMaxResolution", math.inf), r["_rlnMicrographName"]))
        )
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--star", default="micrographs_ctf.star")
    p.add_argument("--review", default="review_current", help="Exclude the already reviewed micrograph.")
    p.add_argument("--source-tile", default="tiles_3p8/tile_0003")
    p.add_argument("--count", type=int, default=5)
    p.add_argument(
        "--exclude", action="append", default=[], help="Additional micrograph basename to exclude."
    )
    p.add_argument("--max-ctf-resolution", type=float, help="Optional maximum CTF-fit resolution, Angstroms.")
    p.add_argument("--out", default="batch_3p8")
    args = p.parse_args()
    if args.count < 1 or (args.max_ctf_resolution is not None and args.max_ctf_resolution <= 0):
        p.error("Use a positive count and optional positive CTF-resolution limit.")
    out = Path(args.out)
    if out.exists():
        p.error(f"{out} already exists; preserve it or choose a new --out.")
    loops = read_loops(args.star)
    micros = [loop for loop in loops if "_rlnMicrographName" in loop[1]]
    if len(micros) != 1:
        p.error("Expected exactly one micrograph loop containing _rlnMicrographName.")
    block, labels, rows = micros[0]
    required = ("_rlnMicrographName", "_rlnDefocusU", "_rlnDefocusV", "_rlnDefocusAngle")
    if not all(key in labels for key in required):
        p.error("Micrograph loop must include both defocus values and astigmatism angle.")
    excluded = {Path(name).name for name in args.exclude}
    settings = Path(args.review) / "review_settings.json"
    if settings.exists():
        name = json.loads(settings.read_text()).get("micrograph")
        if name:
            excluded.add(Path(name).name)
    eligible, seen = [], set()
    for row in rows:
        name = row["_rlnMicrographName"]
        if Path(name).name in excluded or name in seen:
            continue
        seen.add(name)
        if any(numeric(row, key) <= 0 for key in ("_rlnDefocusU", "_rlnDefocusV")):
            continue
        numeric(row, "_rlnDefocusAngle")
        if (
            args.max_ctf_resolution is not None
            and numeric(row, "_rlnCtfMaxResolution", math.inf) > args.max_ctf_resolution
        ):
            continue
        eligible.append(row)
    selected = choose_rows(eligible, args.count)
    optics = [loop for loop in loops if "_rlnOpticsGroup" in loop[1] and "_rlnMicrographName" not in loop[1]]
    if len(optics) > 1:
        p.error("Ambiguous optics loops.")
    if optics:
        lookup = {r["_rlnOpticsGroup"]: r for r in optics[0][2]}
        if any(r.get("_rlnOpticsGroup") not in lookup for r in selected):
            p.error("Selected micrographs refer to missing optics groups.")
    else:
        lookup = {}
    source = Path(args.source_tile) / "search.par"
    parameter_text = source.read_text()
    params = {}
    for line in parameter_text.splitlines():
        fields = line.split("#", 1)[0].split()
        if len(fields) >= 2:
            params[fields[0]] = fields[1:]
    if any(
        not math.isclose(float(params.get(k, ["nan"])[0]), 3.8, abs_tol=1e-6)
        for k in ("angle_inc", "psi_inc")
    ):
        p.error("Source tile must have angle_inc and psi_inc both 3.8.")
    source_apix = float(params["aPerPix"][0])
    manifest = []
    for index, row in enumerate(selected, 1):
        opt = lookup.get(row.get("_rlnOpticsGroup"), {})
        apix = numeric(opt, "_rlnMicrographPixelSize", numeric(row, "_rlnMicrographPixelSize", source_apix))
        ctf_res = numeric(row, "_rlnCtfMaxResolution", math.inf)
        manifest.append(
            dict(
                batch_image=f"image_{index:04d}",
                micrograph_name=row["_rlnMicrographName"],
                basename=Path(row["_rlnMicrographName"]).name,
                optics_group=row.get("_rlnOpticsGroup", ""),
                pixel_size_A=apix,
                defocus_u_A=numeric(row, "_rlnDefocusU"),
                defocus_v_A=numeric(row, "_rlnDefocusV"),
                astigmatism_angle_degrees=numeric(row, "_rlnDefocusAngle"),
                ctf_fit_resolution_A=ctf_res if math.isfinite(ctf_res) else "",
                already_present_in_project=Path(Path(row["_rlnMicrographName"]).name).is_file(),
                matches_source_pixel_size=math.isclose(apix, source_apix, abs_tol=1e-6),
            )
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after every input check; no partial output on validation failure.
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".batch-plan-") as temp:
        stage = Path(temp) / "plan"
        stage.mkdir()
        with (stage / "selected_micrographs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        write_loops(stage / "selected_micrographs_ctf.star", optics + [(block, labels, selected)])
        (stage / "source_search.par.txt").write_text(parameter_text)
        details = dict(
            angular_spacing_degrees=3.8,
            selected_images=len(selected),
            source_tile=str(Path(args.source_tile)),
            source_star=str(Path(args.star)),
            excluded_basenames=sorted(excluded),
            eligible_images=len(eligible),
            selection="Equal-count mean-defocus strata; best CTF-fit resolution in each stratum",
            caveat="Quality-screened feasibility pilot, not an unbiased population sample",
            status="Planned only; source_search.par.txt is a snapshot, not a runnable batch config",
            biological_control="Exclude UBE2L3 from the search template before claiming independent recovery of its density",
            next="Verify archive file paths, obtain these images, then prepare and run tiled searches sequentially",
            images=manifest,
        )
        (stage / "batch_plan.json").write_text(json.dumps(details, indent=2) + "\n")
        stage.rename(out)
    print(f"Selected {len(selected)} additional images from {len(eligible)} eligible micrographs.")
    for row in manifest:
        mean = (row["defocus_u_A"] + row["defocus_v_A"]) / 2 / 10000
        print(
            f"{row['batch_image']}: {row['basename']} | mean defocus {mean:.3f} µm | CTF {row['ctf_fit_resolution_A']} Å"
        )
    print(f"Plan: {(out / 'selected_micrographs.csv').resolve()}")
    print("Prepared only. No images downloaded and no GPU searches started.")


if __name__ == "__main__":
    main()
