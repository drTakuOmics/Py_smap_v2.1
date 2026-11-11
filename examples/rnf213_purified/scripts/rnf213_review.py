#!/usr/bin/env python3
"""Build one consistent, score-filtered RNF213 review from registered search entries.

Lines: cumulative unique threshold-qualified detections or positional reference
matches. Dots: final optimized, pixelwise flat-fielded SMAP SNR (z-scores).
Section assignment uses the representative kept during deduplication.
"""

import argparse
import csv
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import mrcfile
from scipy.optimize import linear_sum_assignment
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def read_csv(path):
    """Rows of a CSV file."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def params(path):
    """{key: [values]} of a search.par."""
    result = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            result[parts[0]] = parts[1:]
    return result


def boolean(value):
    """'true'/'1' or 'false'/'0' as a bool (error otherwise)."""
    if str(value).strip().lower() in ("true", "1"):
        return True
    if str(value).strip().lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid Boolean: {value!r}")


def completed_tiles(run):
    """Names of the tiles whose run.log reports a finished search."""
    return {
        r["tile"]
        for r in read_csv(run / "tile_offsets.csv")
        if (run / r["tile"] / "run.log").is_file()
        and "Done combining search output" in (run / r["tile"] / "run.log").read_text()
    }


def load_run(run, cutoff, dedup_radius, match_radius, selected_tiles=None):
    """Merged analysis of one run restricted to the given sections: registered candidates above
    the cutoff, de-duplicated within dedup_radius and matched to the reference picks within
    match_radius. Refuses runs with a patch registration that needs review."""
    folders = [run / "review/merged_v3", run / "review/merged"]
    folder = next(
        (p for p in folders if (p / "detected_particles.csv").is_file() and (p / "summary.txt").is_file()),
        None,
    )
    if folder is None:
        raise FileNotFoundError(f"No completed merged analysis in {run}. Merge it first.")
    offsets = read_csv(run / "tile_offsets.csv")
    entries = read_csv(folder / "all_candidates_registration.csv")
    reference = read_csv(folder / "reference_particles.csv")
    total_reference_count = len(reference)
    if selected_tiles is not None:
        offsets = [r for r in offsets if r["tile"] in selected_tiles]
        entries = [r for r in entries if r["tile"] in selected_tiles]
        if not offsets:
            raise ValueError("No common completed sections.")
        # Keep original reference IDs, restricting the denominator to the union
        # of the reviewed image-section footprints. This is geometric coverage,
        # not a guarantee of detectability within SMAP's excluded edge margin.
        reference = [
            r
            for r in reference
            if any(
                int(o["x0_zero_based"]) - 0.5
                <= float(r["x"])
                < int(o["x0_zero_based"]) + int(o["width"]) - 0.5
                and int(o["y0_zero_based"]) - 0.5
                <= float(r["y"])
                < int(o["y0_zero_based"]) + int(o["height"]) - 0.5
                for o in offsets
            )
        ]
    refs = np.array([[float(r["x"]), float(r["y"])] for r in reference]).reshape(-1, 2)
    if not len(refs):
        raise ValueError(f"No reference picks in {folder}")
    section = {row["tile"]: i + 1 for i, row in enumerate(offsets)}
    eligible = []
    for record in entries:
        if record["tile"] not in section:
            raise ValueError(f"Unknown tile: {record['tile']}")
        record["section"] = section[record["tile"]]
        record["score"] = float(record["score"])
        record["x"], record["y"] = float(record["x"]), float(record["y"])
        record["fully_contained"] = boolean(record["fully_contained"])
        if not all(np.isfinite(record[k]) for k in ("score", "x", "y")):
            raise ValueError("Non-finite score or coordinate.")
        quality = float(record["registration_ncc"])
        if not np.isfinite(quality) or quality < 0.5 or boolean(record["registration_boundary"]):
            raise ValueError("A patch registration needs review; refusing exports.")
        if record["score"] >= cutoff:
            eligible.append(record)
    # Filter first: an excluded representative must not suppress a qualifying overlap.
    records = []
    for record in sorted(eligible, key=lambda r: (r["fully_contained"], r["score"]), reverse=True):
        if any(
            record["tile"] != kept["tile"]
            and np.hypot(record["x"] - kept["x"], record["y"] - kept["y"]) <= dedup_radius
            for kept in records
        ):
            continue
        records.append(record)
    matches = {}
    if records:
        xy = np.array([[r["x"], r["y"]] for r in records])
        distances = np.linalg.norm(xy[:, None, :] - refs[None, :, :], axis=2)
        penalty = (len(records) + len(refs) + 1) * match_radius
        costs = np.full((len(records), len(refs) + len(records)), penalty)
        costs[:, : len(refs)] = np.where(distances <= match_radius, distances, penalty * 3)
        ii, jj = linear_sum_assignment(costs)
        matches = {
            int(i): int(j) for i, j in zip(ii, jj) if j < len(refs) and distances[i, j] <= match_radius
        }
    for i, r in enumerate(records):
        r["detection_id"] = i + 1
        r["reference_id"] = reference[matches[i]]["reference_id"] if i in matches else ""
        r["reference_distance_pixels"] = float(distances[i, matches[i]]) if i in matches else ""
        r["matched"] = i in matches
    angle = float(params(run / offsets[0]["tile"] / "search.par")["angle_inc"][0])
    return dict(
        run=run,
        folder=folder,
        offsets=offsets,
        records=records,
        angle=angle,
        refs=refs,
        raw_count=len(entries),
        eligible_count=len(eligible),
        total_reference_count=total_reference_count,
        reference_ids=[r["reference_id"] for r in reference],
    )


def csv_export(path, records, fields):
    """Write records as CSV with the given columns."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def star_export(path, records, micrograph):
    """Coordinate-only STAR file: micrograph, x, y (zero-based pixels) and final score."""
    with path.open("w") as handle:
        handle.write("# Zero-based original image pixel centers; final SMAP score.\n")
        handle.write(
            "data_particles\n\nloop_\n_rlnMicrographName #1\n"
            "_rlnCoordinateX #2\n_rlnCoordinateY #3\n_rlnAutopickFigureOfMerit #4\n"
        )
        for r in records:
            handle.write(f"{micrograph} {r['x']:.4f} {r['y']:.4f} {r['score']:.3f}\n")


def export_run(run, out, image, micrograph, args):
    """Outputs of one run in its own folder: detections as CSV and STAR (all, reference-matched,
    unmatched), the particle map figure and summary.txt."""
    folder = out / run["run"].name
    folder.mkdir()
    records = run["records"]
    fields = (
        list(records[0])
        if records
        else [
            "tile",
            "particle_id",
            "x",
            "y",
            "score",
            "section",
            "detection_id",
            "reference_id",
            "reference_distance_pixels",
            "matched",
        ]
    )
    csv_export(folder / "detected_particles.csv", records, fields)
    star_export(folder / "detected_particles.star", records, micrograph)
    for flag, name in [(True, "reference_matched_detections"), (False, "unmatched_detections")]:
        star_export(folder / f"{name}.star", [r for r in records if r["matched"] == flag], micrograph)
    refs = run["refs"]
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    low, high = np.percentile(image[::4, ::4], [1, 99])
    ax.imshow(
        image[::2, ::2],
        cmap="gray",
        vmin=low,
        vmax=high,
        origin="lower",
        extent=(-0.5, image.shape[1] - 0.5, -0.5, image.shape[0] - 0.5),
        interpolation="nearest",
    )
    ax.scatter(
        refs[:, 0],
        refs[:, 1],
        s=90,
        facecolors="none",
        edgecolors="#F0B43C",
        lw=1,
        label=f"Reference picks ({len(refs)})",
    )
    for flag, color, label in [
        (True, "#29B7E8", "Reference-matched candidates"),
        (False, "#E55F72", "Additional detections"),
    ]:
        selected = [r for r in records if r["matched"] == flag]
        if selected:
            xy = np.array([[r["x"], r["y"]] for r in selected])
            ax.scatter(
                xy[:, 0], xy[:, 1], s=28, c=color, marker="+", lw=1, label=f"{label} ({len(selected)})"
            )
    hits = sum(r["matched"] for r in records)
    ax.set(
        xlabel="X (original pixels)",
        ylabel="Y (original pixels)",
        title=f"RNF213 · {run['angle']:g}° · final SNR ≥ {args.score_cutoff:g}\n"
        f"{len(records)} threshold-qualified detections; {hits}/{len(refs)} reference picks matched",
    )
    ax.legend(fontsize=8, loc="upper right", facecolor="white", framealpha=0.85)
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"particle_map.{ext}", dpi=250)
    plt.close(fig)
    summary = (
        f"Angular spacing: {run['angle']:g} degrees\nReviewed sections: {len(run['offsets'])}\nFinal flat-fielded SNR threshold: >= {args.score_cutoff:g}\n"
        f"Reference picks within reviewed tile footprints: {len(refs)}/{run['total_reference_count']}\n"
        f"Original registered entries: {run['raw_count']}\n"
        f"Entries meeting cutoff before overlap removal: {run['eligible_count']}\n"
        f"Threshold-qualified detections after overlap removal: {len(records)}\n"
        f"Reference-position matches: {hits}/{len(refs)} ({hits / len(refs):.1%})\n"
        f"Unmatched candidates: {len(records) - hits}\n"
        f"Deduplication radius: {args.dedup_radius:g} px\nMatch radius: {args.match_radius:g} px\n"
        "Filtering precedes overlap removal; reference assignments are recomputed.\n"
        "Reference picks are selected, not exhaustive ground truth.\n"
        "Coordinates are zero-based original pixels; Frealign angles are preserved and unvalidated.\n"
        "SNR is SMAP's final optimized, pixelwise flat-fielded normalized correlation (z-score).\n"
        "Detection is defined by the stated SNR criterion; its Gaussian false-alarm model does not guarantee each identity.\n"
    )
    (folder / "summary.txt").write_text(summary)
    print(summary)


def publish_review(stage, out, archive_old):
    """Move the staged review to its final folder; a previous review is archived (archive_old)
    or replaced. A symlinked destination is refused."""
    root = Path.cwd().resolve()
    if out.is_symlink():
        raise ValueError("Refusing to replace a symlink review directory.")
    archive = None

    def archive_path(path):
        """Move a path into this run's time-stamped archive folder (created on first use)."""
        nonlocal archive
        if archive is None:
            archive = root / "archive_reviews" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            archive.mkdir(parents=True)
        destination = archive / path.name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.move(str(path), str(destination))
        print(f"Archived: {path.name} -> {archive.relative_to(root)}")

    if out.exists():
        archive_path(out)
    os.replace(stage, out)
    if archive_old:
        old = root / "comparison_5deg_3p8"
        if old.exists() and old != out:
            archive_path(old)
        for path in sorted(root.glob("rnf213_plot_review_v[0-9]*.py")):
            if path.is_file() and not path.is_symlink():
                archive_path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="review_current")
    parser.add_argument(
        "--snr-threshold",
        "--score-cutoff",
        dest="score_cutoff",
        type=float,
        default=8.0,
        help="Retain only final flat-fielded SNR >= threshold before overlap removal and matching.",
    )
    parser.add_argument("--dedup-radius", type=float, default=20)
    parser.add_argument("--match-radius", type=float, default=20)
    parser.add_argument(
        "--archive-old-reviews",
        action="store_true",
        help="Move the prior comparison folder and old plotting scripts into a dated archive.",
    )
    parser.add_argument(
        "--common-completed",
        action="store_true",
        help="Compare only sections completed in every run; recompute filtering, overlap removal and matches on that coverage.",
    )
    args = parser.parse_args()
    if not np.isfinite(args.score_cutoff):
        parser.error("--score-cutoff must be finite.")
    if not all(np.isfinite(v) and v > 0 for v in (args.dedup_radius, args.match_radius)):
        parser.error("Radii must be finite and positive.")
    root = Path.cwd().resolve()
    final_out = Path(args.out).resolve()
    if final_out.parent != root or final_out.name != "review_current":
        parser.error("--out must be the project-level review_current directory.")
    if final_out.exists() and not (final_out / "review_settings.json").is_file():
        parser.error("Existing review_current is not a recognized generated review; refusing replacement.")
    if len({Path(p).name for p in args.runs}) != len(args.runs):
        parser.error("Run directory names must be unique.")
    paths = [Path(path) for path in args.runs]
    geometry = [read_csv(path / "tile_offsets.csv") for path in paths]
    if any(g != geometry[0] for g in geometry[1:]):
        raise ValueError("Runs must have identical section geometry and order.")
    complete = [completed_tiles(path) for path in paths]
    selected_tiles = None
    if args.common_completed:
        selected_tiles = set.intersection(*complete)
        if not selected_tiles:
            raise ValueError("No sections completed in every run.")
        print(f"Common completed sections: {len(selected_tiles)}/{len(geometry[0])}")
        for path, names in zip(paths, complete):
            print(f"{path}: completed {len(names)}/{len(geometry[0])}")
    elif any(len(names) != len(g) for names, g in zip(complete, geometry)):
        raise ValueError("A run is incomplete. Use --common-completed for equal-coverage comparison.")
    runs = [
        load_run(path, args.score_cutoff, args.dedup_radius, args.match_radius, selected_tiles)
        for path in paths
    ]
    if any(r["offsets"] != runs[0]["offsets"] for r in runs[1:]):
        raise ValueError("Runs must have identical section geometry and order.")
    if any(not np.array_equal(r["refs"], runs[0]["refs"]) for r in runs[1:]):
        raise ValueError("Runs must use identical reference picks.")
    if any(r["reference_ids"] != runs[0]["reference_ids"] for r in runs[1:]):
        raise ValueError("Reference-pick identifiers differ between runs.")
    micrographs = list(root.glob("*_fractions.mrc"))
    if len(micrographs) != 1:
        parser.error("Expected exactly one original *_fractions.mrc micrograph in the project.")
    out = Path(tempfile.mkdtemp(prefix=".rnf213_review_build_", dir=root))
    with mrcfile.mmap(micrographs[0]) as m:
        if m.data.ndim != 2:
            raise ValueError("The original micrograph must be two-dimensional.")
        for run in runs:
            export_run(run, out, m.data, micrographs[0].name, args)
    n = len(runs[0]["offsets"])
    colors = ["#2878B5", "#D18A20", "#7564A6", "#348B72"]
    all_scores = [r["score"] for run in runs for r in run["records"]]
    score_low = np.floor(min(all_scores)) - 0.25 if all_scores else 0
    score_high = np.ceil(max(all_scores)) + 0.25 if all_scores else 1
    score_low = min(score_low, args.score_cutoff - 0.5)
    score_high = max(score_high, args.score_cutoff + 0.5)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True, layout="constrained")
    right_axes = [ax.twinx() for ax in axes]
    exports, dot_exports = [], []
    for panel, (ax, right) in enumerate(zip(axes, right_axes)):
        legends = []
        for run_index, run in enumerate(runs):
            color = colors[run_index % len(colors)]
            records = [r for r in run["records"] if panel == 0 or r["matched"]]
            counts = np.bincount(np.array([r["section"] for r in records], dtype=int), minlength=n + 1)[1:]
            cumulative = np.cumsum(counts)
            (line,) = ax.plot(
                np.arange(1, n + 1),
                cumulative,
                "-o",
                color=color,
                lw=1.9,
                ms=2.8,
                zorder=3,
                label=f"{run['angle']:g}° ({len(records)})",
            )
            legends.append(line)
            xs, ys = [], []
            for section in range(1, n + 1):
                local = sorted((r for r in records if r["section"] == section), key=lambda r: -r["score"])
                jitter = np.linspace(-0.18, 0.18, len(local)) if len(local) > 1 else [0]
                for r, shift in zip(local, jitter):
                    xs.append(section + shift)
                    ys.append(r["score"])
            right.scatter(xs, ys, s=13, alpha=0.40, color=color, edgecolors="none", zorder=1)
            if panel == 0:
                for section, count, total in zip(range(1, n + 1), counts, cumulative):
                    exports.append(
                        dict(
                            run=str(run["run"]),
                            angle_degrees=run["angle"],
                            section=section,
                            detected_particle_candidates=int(count),
                            cumulative_detected_particle_candidates=int(total),
                            cumulative_reference_matches=sum(
                                r["matched"] and r["section"] <= section for r in run["records"]
                            ),
                        )
                    )
                for r in records:
                    dot_exports.append(
                        dict(
                            run=str(run["run"]),
                            angle_degrees=run["angle"],
                            tile=r["tile"],
                            section=r["section"],
                            x=r["x"],
                            y=r["y"],
                            final_optimized_smap_score=r["score"],
                            reference_position_match=r["matched"],
                        )
                    )
        legends.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="#888888",
                markersize=4,
                label="Dots: final SNR of each detection (right axis)",
            )
        )
        right.axhline(args.score_cutoff, color="#555555", linestyle=":", lw=1.6, zorder=2)
        legends.append(
            Line2D(
                [],
                [],
                linestyle=":",
                color="#555555",
                lw=1.6,
                label=f"Detection SNR threshold: {args.score_cutoff:g} (right axis)",
            )
        )
        ax.legend(handles=legends, loc="upper left", frameon=False, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0.5, n + 0.5)
        ax.grid(axis="y", alpha=0.16)
        ax.spines["top"].set_visible(False)
        right.spines["top"].set_visible(False)
        right.set_ylim(score_low, score_high)
        right.set_ylabel("Final flat-fielded SMAP SNR", color="#666666")
        right.tick_params(axis="y", colors="#666666")
        ax.set_zorder(right.get_zorder() + 1)
        ax.patch.set_visible(False)
    coverage = f"{n} common completed sections" if args.common_completed else f"{n} sections"
    axes[0].set(
        title=f"RNF213 · {coverage} · final SNR ≥ {args.score_cutoff:g}",
        ylabel="Cumulative detected particle candidates",
    )
    axes[1].set(
        title="Threshold-qualified detections matching reference picks",
        ylabel="Cumulative reference-position matches",
        xlabel="Common section order (see section_tiles.csv)" if args.common_completed else "Section number",
    )
    fig.get_layout_engine().set(rect=(0, 0.055, 1, 1))
    fig.text(
        0.5,
        0.008,
        f"Overlap duplicates removed. Positions matched within {args.match_radius:g} px; orientations unvalidated.\n"
        "Accumulation follows representative sections; dots show final optimized flat-fielded SNR.\n"
        "All plotted detections meet the stated SNR criterion; the Gaussian noise model does not guarantee each identity.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        path = out / f"cumulative_candidates_with_scores.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    for name, rows in [
        ("cumulative_candidates_with_scores.csv", exports),
        ("candidate_score_dots.csv", dot_exports),
    ]:
        if rows:
            with (out / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    csv_export(
        out / "section_tiles.csv",
        [dict(section=i + 1, **offset) for i, offset in enumerate(runs[0]["offsets"])],
        ["section"] + list(runs[0]["offsets"][0]),
    )
    (out / "review_settings.json").write_text(
        json.dumps(
            dict(
                score_cutoff=args.score_cutoff,
                dedup_radius_pixels=args.dedup_radius,
                match_radius_pixels=args.match_radius,
                micrograph=micrographs[0].name,
                source_registered_tables=[
                    str((r["folder"] / "all_candidates_registration.csv").resolve()) for r in runs
                ],
                common_completed_only=args.common_completed,
                reviewed_tiles=[r["tile"] for r in runs[0]["offsets"]],
                reference_coverage="Union of reviewed tile image footprints; SMAP edge exclusions can reduce detectability",
                pipeline="final-score filter -> overlap removal -> one-to-one positional matching",
            ),
            indent=2,
        )
        + "\n"
    )
    (out / "README.txt").write_text(
        "Active review: all exported detections meet the final optimized flat-fielded SNR threshold.\n"
        "Each run has a filtered CSV, coordinate STAR files, particle map and summary.\n"
        "Source search files and registration tables are unchanged.\n"
        "Original Frealign angles and stack references are preserved in CSV.\n"
        "Positional reference matching does not validate orientations or all unmatched candidates.\n"
    )
    if args.common_completed:
        with (out / "README.txt").open("a") as handle:
            handle.write(
                "Equal-coverage comparison: only sections completed in every run are included.\n"
                "Reference recovery denominator includes reference centers within the union of reviewed tile footprints.\n"
                "Image footprints include edge regions; SMAP edge exclusions can reduce detectability.\n"
                "section_tiles.csv maps the common section order to original tile names and coordinates.\n"
            )
    publish_review(out, final_out, args.archive_old_reviews)
    print(f"Current review: {final_out}")
    print(f"Comparison: {final_out / 'cumulative_candidates_with_scores.png'}")


if __name__ == "__main__":
    main()
