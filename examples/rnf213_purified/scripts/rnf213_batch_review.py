#!/usr/bin/env python3
"""Review the completed no-E3/CTD batch; reuse existing registration helpers.

No searches or reconstruction. Final-score filtering precedes per-image overlap
removal. Anchor-centered coordinates/rotations are preserved for reconstruction;
parent-model centers are calculated only for reference matching and display.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import mrcfile
from scipy.special import erfc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import rnf213_batch_workflow as batch_tools
import rnf213_workflow_v3 as workflow
import rnf213_fragment_positions as positions
import rnf213_surface_overlay as overlay


FIELDS = [
    "batch_image",
    "micrograph",
    "tile",
    "particle_id",
    "score",
    "x",
    "y",
    "parent_x",
    "parent_y",
    "local_x",
    "local_y",
    "registration_ncc",
    "registration_boundary",
    "fully_contained",
    "frealign_psi",
    "frealign_theta",
    "frealign_phi",
    "defocus_u_A",
    "defocus_v_A",
    "astigmatism_angle_deg",
    "pixel_size_A",
    "particle_image",
    "template_image",
    "section",
]
FIELDS += [f"R{i}{j}" for i in range(3) for j in range(3)]
FINAL_FIELDS = FIELDS + ["detection_id", "reference_id", "reference_distance_pixels"]
COLORS = ["#2878B5", "#D47E32", "#43956B", "#886AB5"]


def write_csv(path, rows, fields):
    """Write rows as CSV with the given columns (other keys ignored)."""
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save(fig, folder, name):
    """Save a figure as PNG and PDF (300 dpi) and close it."""
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def deduplicate(entries, cutoff, radius):
    """Suppress overlaps from different tiles only, separately per micrograph."""
    eligible = [r for r in entries if r["score"] >= cutoff]
    kept, links = [], []
    for row in sorted(eligible, key=lambda r: (r["fully_contained"], r["score"]), reverse=True):
        near = [
            (i, float(np.hypot(row["parent_x"] - k["parent_x"], row["parent_y"] - k["parent_y"])))
            for i, k in enumerate(kept)
            if row["batch_image"] == k["batch_image"] and row["tile"] != k["tile"]
        ]
        near = [(i, d) for i, d in near if d <= radius]
        if near:
            i, distance = min(near, key=lambda t: t[1])
        else:
            i, distance = len(kept), 0.0
            kept.append(dict(row))
        links.append(
            dict(
                batch_image=row["batch_image"],
                tile=row["tile"],
                particle_id=row["particle_id"],
                representative_id=i + 1,
                distance_parent_pixels=distance,
            )
        )
    return eligible, kept, links


def reference_coordinates(path, names):
    """Stream one RELION particle loop; no assumption of exhaustive picks."""
    result = {name: [] for name in names}
    if not path.is_file():
        return result
    import shlex

    labels, in_loop, in_data = [], False, False
    required = ["_rlnMicrographName", "_rlnCoordinateX", "_rlnCoordinateY"]
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("data_") or line == "loop_":
                labels, in_loop, in_data = [], line == "loop_", False
                continue
            if in_loop and line.startswith("_") and not in_data:
                labels.append(line.split()[0])
                continue
            if not in_loop or not labels:
                continue
            in_data = True
            if not all(k in labels for k in required):
                continue
            fields = shlex.split(line) if any(c in line for c in "'\"") else line.split()
            if len(fields) != len(labels):
                raise ValueError(f"Unexpected STAR row width: {path}")
            i, x, y = [labels.index(k) for k in required]
            name = Path(fields[i]).name
            if name in result:
                result[name].append([float(fields[x]), float(fields[y])])
    return result


def register_image(root, directory, meta, offsets, convention, center_shift, out, cutoff=8.0):
    """Register the saved particle patches of every tile of one micrograph to the micrograph
    (rnf213_workflow_v3) and return one entry per candidate with its micrograph position."""
    entries = []
    for section, offset in enumerate(offsets, 1):
        tile = directory / offset["tile"]
        text = (tile / "run.log").read_text(errors="replace")
        scores = workflow.section_scores(tile, text)
        if not len(scores):
            print(f"{directory.name}/{tile.name}: completed, no candidates", flush=True)
            continue
        p = workflow.read_params(tile / "search.par")
        image = workflow.read_array(batch_tools.project_path(root, p["imageFile"][0]))
        seeds = workflow.matrix(tile / "output/list_final.txt")
        angles = workflow.matrix(tile / "output/particles.par", skiprows=1)
        patches = workflow.read_array(tile / "output/particles.mrc")
        if patches.ndim == 2:
            patches = patches[None]
        with mrcfile.mmap(tile / "output/templates.mrc") as m:
            nt = 1 if m.data.ndim == 2 else len(m.data)
        if not (len(scores) == len(seeds) == len(angles) == len(patches) == nt):
            raise ValueError(f"Particle stack/list lengths disagree: {tile}")
        with mrcfile.mmap(tile / "output/search_mean.mrc") as m:
            search_shape = np.array(m.data.shape[-2:])
        ratio = float(p["aPerPix_search"][0]) / float(p["aPerPix"][0])
        for j, (score, seed, angle, patch) in enumerate(zip(scores, seeds, angles, patches)):
            if int(score[0]) != j + 1 or int(angle[0]) != j + 1:
                raise ValueError(f"Unexpected particle indexing: {tile}")
            seed_rc = (
                np.floor((seed[1:3] - (search_shape // 2 + 1)) * ratio + np.array(image.shape) // 2 + 1 + 0.5)
                - 1
            )
            y, x, ncc, boundary = workflow.register_patch(image, patch, *seed_rc)
            margin = min(x, y, image.shape[1] - 1 - x, image.shape[0] - 1 - y)
            row = dict(
                batch_image=directory.name,
                micrograph=meta["source_micrograph_name"],
                tile=tile.name,
                particle_id=j + 1,
                score=float(score[1]),
                x=x + int(offset["x0_zero_based"]),
                y=y + int(offset["y0_zero_based"]),
                local_x=x,
                local_y=y,
                registration_ncc=ncc,
                registration_boundary=boundary,
                fully_contained=margin >= max(patch.shape) / 2,
                frealign_psi=float(angle[1]),
                frealign_theta=float(angle[2]),
                frealign_phi=float(angle[3]),
                defocus_u_A=float(angle[8]),
                defocus_v_A=float(angle[9]),
                astigmatism_angle_deg=float(angle[10]),
                pixel_size_A=float(meta["pixel_size_A"]),
                section=section,
                particle_image=f"{j + 1:06d}@{tile / 'output/particles.mrc'}",
                template_image=f"{j + 1:06d}@{tile / 'output/templates.mrc'}",
            )
            r = overlay.rotation(row, convention)
            shift = r @ center_shift
            row.update(parent_x=row["x"] + float(shift[0]), parent_y=row["y"] + float(shift[1]))
            row.update({f"R{i}{k}": float(r[i, k]) for i in range(3) for k in range(3)})
            entries.append(row)
        print(f"Registered {directory.name}/{tile.name}: {len(scores)} patches", flush=True)
    write_csv(out / "all_candidates_registration.csv", entries, FIELDS)
    bad = [
        r
        for r in entries
        if r["registration_ncc"] < 0.5
        or r["registration_boundary"]
        or not np.isfinite(
            [r[k] for k in ("x", "y", "parent_x", "parent_y", "score", "registration_ncc")]
        ).all()
    ]
    write_csv(out / "registration_failures.csv", bad, FIELDS)
    blocking = [r for r in bad if not np.isfinite(r["score"]) or r["score"] >= cutoff]
    if blocking:
        raise ValueError(
            f"{len(blocking)} registration failures affect candidates at/above SNR {cutoff:g} or with invalid scores; see {out}/registration_failures.csv"
        )
    if bad:
        print(
            f"{directory.name}: {len(bad)} registration failures recorded below SNR {cutoff:g}; excluded by the final score cutoff.",
            flush=True,
        )
    return entries


def coordinate_star(path, rows):
    """Coordinate-only STAR file (micrograph, x, y, final score); no Euler angles."""
    with path.open("w") as f:
        f.write(
            "# Anchor-SP-centered, zero-based original pixel centers.\n"
            "# Coordinate-only STAR; no conversion to RELION Euler angles.\n"
            "data_particles\n\nloop_\n_rlnMicrographName #1\n"
            "_rlnCoordinateX #2\n_rlnCoordinateY #3\n_rlnAutopickFigureOfMerit #4\n"
        )
        for r in rows:
            f.write(f'"{r["micrograph"]}" {r["x"]:.4f} {r["y"]:.4f} {r["score"]:.3f}\n')


def check_rotations(fvolume, rows, root, directory, base_parameters, convention):
    """Reuse one masked Fourier volume for all image checks."""
    names = ("A", "AT", "DAD", "DATD", "DA", "AD", "DAT", "ATD")
    tests = []
    for row in sorted(rows, key=lambda r: -r["score"])[:2]:
        p = overlay.parameters(directory / row["tile"] / "search.par")
        for key in ("mask_params", "aPerPix", "F_abs"):
            if p[key] != base_parameters[key]:
                raise ValueError(f"Template preparation differs between tiles: {key}")
        saved = overlay.stack_image(row["template_image"], root)
        if saved.shape != fvolume.shape[1:]:
            raise ValueError("Saved template and SP projection grids differ.")
        transfer = overlay.ctf(row, saved.shape[0], p)
        scores = {
            n: overlay.correlation(
                saved, overlay.template_projection(fvolume, overlay.rotation(row, n), transfer)
            )
            for n in names
        }
        if not np.isfinite(list(scores.values())).all():
            raise ValueError("Non-finite template rotation correlation.")
        tests.append(dict(tile=row["tile"], particle_id=row["particle_id"], **scores))
    if not tests:
        return None
    medians = {n: float(np.median([t[n] for t in tests])) for n in names}
    ranked = sorted(names, key=lambda n: -medians[n])
    if (
        ranked[0] != convention
        or medians[convention] < 0.97
        or min(t[convention] for t in tests) < 0.94
        or medians[convention] - medians[ranked[1]] < 0.02
    ):
        raise ValueError(f"Rotation conversion needs review: {directory.name}: {medians}")
    print(
        f"{directory.name}: rotation {convention}, median saved-template NCC {medians[convention]:.4f}",
        flush=True,
    )
    return dict(
        batch_image=directory.name,
        convention=convention,
        median_template_NCC=medians[convention],
        checks=tests,
    )


def image_figures(root, item, folder, cutoff):
    """Figures of one micrograph: detections and reference picks on the image, and saved
    particle / template pairs."""
    rows, refs, meta = item["kept"], np.array(item["refs"]).reshape(-1, 2), item["meta"]
    image = workflow.read_array(Path(meta["source_micrograph"]))
    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    lo, hi = np.percentile(image[::4, ::4], [1, 99])
    ax.imshow(
        image[::2, ::2],
        cmap="gray",
        vmin=lo,
        vmax=hi,
        origin="lower",
        extent=(-0.5, image.shape[1] - 0.5, -0.5, image.shape[0] - 0.5),
    )
    if len(refs):
        ax.scatter(
            *refs.T,
            s=75,
            facecolors="none",
            edgecolors="#F0B43C",
            lw=0.9,
            label=f"Reference picks ({len(refs)})",
        )
    for flag, color, label in [
        (True, "#29B7E8", "Reference-position matches"),
        (False, "#E55F72", "Other detected particle candidates"),
    ]:
        subset = [r for r in rows if bool(r["reference_id"]) == flag]
        if subset:
            xy = np.array([[r["parent_x"], r["parent_y"]] for r in subset])
            ax.scatter(*xy.T, marker="+", s=32, c=color, lw=1, label=f"{label} ({len(subset)})")
    ax.set(
        xlabel="X (original pixels)",
        ylabel="Y (original pixels)",
        title=f"{item['name']} · no-E3/CTD anchor · 3.8° · final SNR ≥ {cutoff:g}\n"
        f"{len(rows)} detected particle candidates after overlap removal",
    )
    if rows or len(refs):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    save(fig, folder, "particle_map")
    chosen = sorted(rows, key=lambda r: -r["score"])[:6]
    if chosen:
        fig, axes = plt.subplots(len(chosen), 2, figsize=(6, 2.25 * len(chosen)), squeeze=False)
        for pair, row in zip(axes, chosen):
            for ax, field in zip(pair, ("particle_image", "template_image")):
                data = overlay.stack_image(row[field], root)
                low, high = np.percentile(data, [1, 99])
                ax.imshow(data, cmap="gray", origin="lower", vmin=low, vmax=high)
                ax.axis("off")
            pair[0].set_title(
                f"{row['tile']} / particle {row['particle_id']} · SNR {row['score']:.2f}", fontsize=9
            )
            pair[1].set_title("Fitted anchor template (E3/CTD omitted)", fontsize=9)
        fig.suptitle("Saved particle / template pairs\nIndependent display contrast per panel", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, folder, "particle_template_pairs")


def batch_figures(items, out, cutoff):
    """Batch figures: candidates per tile with their scores, cumulative counts, and the
    viewing-direction coverage of the detections."""
    fig, axes = plt.subplots(len(items), 1, figsize=(9, 3 * len(items)), squeeze=False, layout="constrained")
    totals = [len(r["kept"]) for r in items]
    ymax = max(1, max(totals))
    allscores = [r["score"] for item in items for r in item["kept"]]
    snrmax = max([cutoff + 1] + allscores)
    progress = []
    for index, (item, ax) in enumerate(zip(items, axes[:, 0])):
        color = COLORS[index % len(COLORS)]
        counts = [sum(r["section"] == s for r in item["kept"]) for s in range(1, len(item["offsets"]) + 1)]
        matches = [
            sum(r["section"] == s and bool(r["reference_id"]) for r in item["kept"])
            for s in range(1, len(counts) + 1)
        ]
        x = np.arange(1, len(counts) + 1)
        cumulative, hits = np.cumsum(counts), np.cumsum(matches)
        xp = np.r_[0.5, x, len(counts) + 0.5]
        ax.step(
            xp,
            np.r_[0, cumulative, cumulative[-1]],
            where="post",
            c=color,
            lw=2,
            label=f"Detected particle candidates ({sum(counts)})",
        )
        if item["refs"]:
            ax.step(
                xp,
                np.r_[0, hits, hits[-1]],
                where="post",
                c="#606870",
                ls="--",
                label=f"Reference-position matches ({sum(matches)})",
            )
        score_ax = ax.twinx()
        score_ax.scatter(
            [r["section"] for r in item["kept"]],
            [r["score"] for r in item["kept"]],
            s=14,
            color=color,
            alpha=0.55,
        )
        score_ax.axhline(cutoff, c="#8A4545", ls=":", lw=1)
        score_ax.set(ylim=(cutoff - 0.5, snrmax + 0.5), ylabel="Final optimized SNR (dots)")
        ax.set(
            ylim=(0, ymax * 1.12),
            xlim=(0.5, len(counts) + 0.5),
            xlabel="Tile number",
            ylabel="Cumulative candidate count",
            title=item["name"],
        )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.15)
        for s, n, h, c, t in zip(x, counts, matches, cumulative, hits):
            progress.append(
                dict(
                    batch_image=item["name"],
                    tile=f"tile_{s:04d}",
                    detected_particle_candidates=int(n),
                    reference_matches=int(h),
                    cumulative_candidates=int(c),
                    cumulative_reference_matches=int(t),
                )
            )
    save(fig, out, "cumulative_candidates_with_scores")
    write_csv(out / "tile_progress.csv", progress, list(progress[0]))

    direction_histograms = []
    for item in items:
        directions = np.array(
            [
                np.array([[r[f"R{i}{j}"] for j in range(3)] for i in range(3)]).T @ np.array([0.0, 0.0, 1.0])
                for r in item["kept"]
            ]
        ).reshape(-1, 3)
        lon = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
        z = np.clip(directions[:, 2], -1, 1)
        direction_histograms.append(np.histogram2d(lon, z, bins=(24, 12), range=((-180, 180), (-1, 1))))
    peak_bin_count = max(1, max(float(h[0].max()) for h in direction_histograms))
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 4), squeeze=False, layout="constrained")
    for ax, item, histogram in zip(axes[0], items, direction_histograms):
        rows = item["kept"]
        counts, xx, yy = histogram
        heat = ax.pcolormesh(xx, yy, counts.T, cmap="Blues", shading="flat", vmin=0, vmax=peak_bin_count)
        fig.colorbar(heat, ax=ax, label="Candidates / equal-area direction bin", shrink=0.7)
        ax.set(
            xlim=(-180, 180),
            ylim=(-1, 1),
            xlabel="Viewing-direction longitude (degrees)",
            ylabel="Viewing-direction Z component",
            title=f"{item['name']} · {len(rows)} candidates",
        )
    fig.suptitle("Estimated viewing-direction coverage · in-plane angle is not shown")
    save(fig, out, "orientation_coverage")


def survival_figure(items, out):
    """Compare each image's summed retained CC tails, not particle detections."""
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 4), squeeze=False, layout="constrained")
    exported = []
    for ax, item in zip(axes[0], items):
        curves = []
        for o in item["offsets"]:
            curve = np.loadtxt(item["directory"] / o["tile"] / "output/search_SH.txt", ndmin=2)
            if curve.shape[1] != 3 or len(curve) < 2 or not np.isfinite(curve).all():
                raise ValueError("Invalid search_SH.txt")
            if np.any(np.diff(curve[:, 0]) < 0) or np.any(curve[:, 1:] < 0):
                raise ValueError("Invalid survival threshold/count ordering")
            curves.append(curve)
        left = max(c[0, 0] for c in curves)
        right = max(c[-1, 0] for c in curves)
        x = np.linspace(left, right, 1500)
        measured = np.zeros_like(x)
        ns = 0.0
        for c in curves:
            q = erfc(c[:, 0] / np.sqrt(2)) / 2
            valid = (q > 1e-250) & (c[:, 1] > 0)
            if not valid.any():
                raise ValueError("Cannot recover Gaussian search normalization")
            ns += float(np.median(c[valid, 1] / q[valid]))
            # Use first stored threshold at or above x: histogram-resolution
            # approximation; no linear interpolation or fractional particle counts.
            ix = np.searchsorted(c[:, 0], x, side="left")
            valid_x = ix < len(c)
            measured[valid_x] += c[ix[valid_x], 2]
        gaussian = ns * erfc(x / np.sqrt(2)) / 2
        for yy, color, label in [
            (gaussian, "#C44E52", "Gaussian search expectation"),
            (measured, "#2878B5", "Retained normalized CC tail"),
        ]:
            valid = yy > 0
            ax.plot(x[valid], yy[valid], c=color, label=label, lw=1.6)
        ax.set(
            yscale="log",
            xlabel="Flat-fielded search SNR threshold",
            ylabel="Surviving CC evaluations (overlapping tiles summed)",
            title=item["name"],
        )
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.15)
        exported.extend(
            dict(
                batch_image=item["name"],
                snr_threshold=float(a),
                gaussian_expected_CC=float(b),
                retained_CC=float(c),
            )
            for a, b, c in zip(x, gaussian, measured)
        )
    fig.suptitle(
        "Search tails · displayed only where every tile has stored coverage\n"
        "CC evaluations are correlated and are not unique particle counts",
        fontsize=11,
    )
    save(fig, out, "search_survival_by_image")
    write_csv(out / "search_survival_by_image.csv", exported, list(exported[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", default="batch_noE3_3")
    p.add_argument("--snr-threshold", type=float, default=8.0)
    p.add_argument("--dedup-radius", type=float, default=20.0)
    p.add_argument("--match-radius", type=float, default=20.0)
    p.add_argument("--reference", default="refined_particles.star")
    p.add_argument("--out", help="Default: BATCH/review; refuses existing output")
    a = p.parse_args()
    if (
        not np.isfinite([a.snr_threshold, a.dedup_radius, a.match_radius]).all()
        or min(a.dedup_radius, a.match_radius) <= 0
    ):
        p.error("Use a finite cutoff and positive radii.")
    root = Path.cwd().resolve()
    batch = Path(a.batch).resolve()
    details = json.loads((batch / "searches/batch_search.json").read_text())
    out = Path(a.out).resolve() if a.out else batch / "review"
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new --out to preserve this review.")
    anchor = Path(details["source_anchor"])
    center_file = anchor / "review/position_check/center_registration.json"
    center = json.loads(center_file.read_text())
    shift = np.array(center["full_SP_origin_in_fragment_SP_xyz_pixels"], float)
    convention = center["fragment_rotation_check"]["convention"]
    sp = Path(details["anchor_SP"])
    if batch_tools.digest(sp) != details["anchor_SP_sha256"]:
        raise ValueError("Batch anchor SP differs from prepared input.")
    anchor_manifest = json.loads((anchor / "manifest.json").read_text())
    anchor_cif = anchor / anchor_manifest["fragment_CIF"]
    if sp != (anchor / "output" / (anchor_cif.stem + "_SP.mrc")).resolve():
        raise ValueError("Saved center registration must describe this batch anchor SP.")
    if (
        Path(center["fragment"]["CIF"]).resolve() != anchor_cif.resolve()
        or shift.shape != (3,)
        or not np.isfinite(shift).all()
    ):
        raise ValueError("Center-registration provenance mismatch.")
    jobs = batch_tools.read_csv(batch / "searches/jobs.csv")
    for job in jobs:
        tile = batch_tools.project_path(root, job["directory"])
        if not batch_tools.complete(tile):
            raise RuntimeError(f"Incomplete search: {tile}")
        if (
            batch_tools.digest(tile / "search.par") != job["configuration_sha256"]
            or batch_tools.digest(tile / job["tile_file"]) != job["tile_sha256"]
        ):
            raise ValueError(f"Prepared input changed: {tile}")
    names = sorted({j["batch_image"] for j in jobs})
    manifests = {
        name: json.loads((batch / "searches" / name / "micrograph_manifest.json").read_text())
        for name in names
    }
    refs = reference_coordinates(
        Path(a.reference).resolve(), [Path(m["source_micrograph_name"]).name for m in manifests.values()]
    )
    out.mkdir(parents=True)
    volume = positions.load_sp(sp, float(manifests[names[0]]["pixel_size_A"]))
    first_tile = batch_tools.project_path(root, jobs[0]["directory"])
    base_parameters = overlay.parameters(first_tile / "search.par")
    fvolume = overlay.masked_fourier(volume, base_parameters)
    del volume
    items, allkept, summaries, checks = [], [], [], []
    for name in names:
        directory = batch / "searches" / name
        folder = out / name
        folder.mkdir()
        meta = manifests[name]
        if (
            batch_tools.digest(meta["source_micrograph"]) != meta["source_micrograph_sha256"]
            or meta["anchor_SP_sha256"] != details["anchor_SP_sha256"]
        ):
            raise ValueError(f"Source image/model provenance changed: {name}")
        if batch_tools.digest(meta["anchor_CIF"]) != meta["anchor_CIF_sha256"]:
            raise ValueError(f"Anchor CIF changed: {name}")
        offsets = batch_tools.read_csv(directory / "tile_offsets.csv")
        expected = {j["tile"] for j in jobs if j["batch_image"] == name}
        if len(offsets) != len(expected) or {o["tile"] for o in offsets} != expected:
            raise ValueError(f"Tile offsets and prepared jobs differ: {name}")
        entries = register_image(root, directory, meta, offsets, convention, shift, folder, a.snr_threshold)
        check = check_rotations(fvolume, entries, root, directory, base_parameters, convention)
        if check:
            checks.append(check)
        eligible, kept, links = deduplicate(entries, a.snr_threshold, a.dedup_radius)
        reference = refs[Path(meta["source_micrograph_name"]).name]
        matches = positions.assignments(
            [(r["parent_x"], r["parent_y"]) for r in kept], reference, a.match_radius
        )
        for i, row in enumerate(kept):
            row.update(
                detection_id=f"{name}_{i + 1:04d}",
                reference_id=matches[i][0] + 1 if i in matches else "",
                reference_distance_pixels=matches[i][1] if i in matches else "",
            )
        write_csv(folder / "detected_particles.csv", kept, FINAL_FIELDS)
        write_csv(
            folder / "overlap_assignments.csv",
            links,
            ["batch_image", "tile", "particle_id", "representative_id", "distance_parent_pixels"],
        )
        write_csv(
            folder / "reference_particles.csv",
            [dict(reference_id=i + 1, x=q[0], y=q[1]) for i, q in enumerate(reference)],
            ["reference_id", "x", "y"],
        )
        coordinate_star(folder / "anchor_center_coordinates.star", kept)
        item = dict(name=name, directory=directory, meta=meta, kept=kept, refs=reference, offsets=offsets)
        items.append(item)
        allkept.extend(kept)
        summary = dict(
            batch_image=name,
            tiles=len(offsets),
            refined_tile_entries=len(entries),
            entries_SNR_ge_cutoff=len(eligible),
            detected_particle_candidates=len(kept),
            candidates_with_fully_contained_tile_patch=sum(bool(r["fully_contained"]) for r in kept),
            reference_picks=len(reference) if reference else "",
            reference_position_matches=len(matches) if reference else "",
            reference_recovery_percent=100 * len(matches) / len(reference) if reference else "",
            minimum_registration_NCC=min([r["registration_ncc"] for r in entries], default=None),
        )
        summaries.append(summary)
        print(
            f"\n{name}: {len(entries)} entries; {len(eligible)} meet SNR ≥ {a.snr_threshold:g}; "
            f"{len(kept)} detected particle candidates after overlap removal; "
            + (
                f"{len(matches)}/{len(reference)} reference-position matches."
                if reference
                else "reference picks unavailable for this image."
            ),
            flush=True,
        )
        image_figures(root, item, folder, a.snr_threshold)
    del fvolume
    write_csv(out / "batch_summary.csv", summaries, list(summaries[0]))
    write_csv(out / "detected_particles.csv", allkept, FINAL_FIELDS)
    coordinate_star(out / "anchor_center_coordinates.star", allkept)
    batch_figures(items, out, a.snr_threshold)
    survival_figure(items, out)
    report = dict(
        batch=str(batch),
        final_SNR_cutoff=a.snr_threshold,
        dedup_radius_pixels=a.dedup_radius,
        match_radius_pixels=a.match_radius,
        total_detected_particle_candidates=len(allkept),
        images=summaries,
        center_registration=center,
        rotation_checks=checks,
        anchor_SP=str(sp),
        anchor_SP_sha256=details["anchor_SP_sha256"],
        coordinate_definition="x/y: anchor-SP center; parent_x/y: full-SP center for matching/display; all zero-based original pixels",
        rotation_definition="R maps anchor-SP object XYZ to image XYZ; original Frealign angles also retained",
        caveats=[
            "SNR criterion defines candidates, not independently confirmed identities.",
            "Reference picks are selected, not exhaustive ground truth; orientations not independently validated.",
            "Viewing-direction coverage is diagnostic, not a resolution prediction.",
            "Survival curves count correlated CC evaluations in overlapping tiles, not particles.",
            "No experimental reconstruction has been created by this review.",
        ],
    )
    (out / "review_report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"{s['batch_image']}: {s['detected_particle_candidates']} detected particle candidates at final SNR ≥ {a.snr_threshold:g}"
        for s in summaries
    ]
    lines += [
        f"Total: {len(allkept)} candidates after per-image overlap removal.",
        "Next: extract original-image patches around anchor-SP centers; preserve CTF and checked rotation matrices.",
        "Assess yield, image distribution and orientation coverage before reconstructing independent halves.",
        *report["caveats"],
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\nReview: {out}")


if __name__ == "__main__":
    main()
