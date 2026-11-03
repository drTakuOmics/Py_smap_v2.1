#!/usr/bin/env python3
"""Map SMAP patches to a micrograph, merge overlaps and compare RELION picks.

Coordinates are zero-based pixel centers in the original micrograph. Registration
to the actual saved patches avoids treating list_final.txt as final image coords.
Frealign angles are retained in CSV and are not relabelled as RELION angles.
"""
import argparse
import csv
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import mrcfile
from scipy.signal import fftconvolve
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_array(path):
    with mrcfile.open(path) as m:
        return np.array(m.data, dtype=np.float64, copy=True)


def matrix(path, skiprows=0):
    text = Path(path).read_text().splitlines()[skiprows:]
    lines = [x for x in text if x.strip() and not x.lstrip().startswith("#")]
    return np.loadtxt(lines, ndmin=2) if lines else np.empty((0, 0))


def section_scores(tile, log_text):
    """An explicitly completed empty search may have no particle score file."""
    path = tile / "output/particles.txt"
    if path.exists():
        return matrix(path)
    if ("Done combining search output" in log_text
            and "nothing to optimize" in log_text
            and "starting optimization on" not in log_text):
        return np.empty((0, 0))
    raise FileNotFoundError(f"Missing particle scores without an empty-search marker: {path}")


def read_params(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            result[fields[0]] = fields[1:]
    return result


def register_patch(image, patch, seed_y, seed_x, radius=24):
    """Find saved patch in a mean-padded window using normalized CC."""
    ph, pw = patch.shape
    hy, hx = ph // 2, pw // 2
    sy, sx = int(round(seed_y)), int(round(seed_x))
    padding = max(ph, pw) + radius + 4
    padded = np.pad(image, padding, mode="constant", constant_values=image.mean())
    y0, x0 = sy - hy - radius, sx - hx - radius
    window = padded[y0+padding:y0+padding+ph+2*radius,
                    x0+padding:x0+padding+pw+2*radius]
    if window.shape != (ph+2*radius, pw+2*radius):
        raise ValueError("Seed is outside image bounds; check grid conversion.")
    template = patch - patch.mean()
    ones = np.ones(patch.shape)
    numerator = fftconvolve(window, template[::-1, ::-1], mode="valid")
    total = fftconvolve(window, ones, mode="valid")
    total2 = fftconvolve(window**2, ones, mode="valid")
    variance = np.maximum(total2 - total**2 / patch.size, 0)
    denominator = np.sqrt(variance * np.sum(template**2))
    ncc = np.divide(numerator, denominator, out=np.full_like(numerator, -1),
                    where=denominator > 1e-12)
    iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape)

    def refine(axis, pos):
        if pos == 0 or pos == ncc.shape[axis]-1:
            return 0.0
        a, b, c = (ncc[iy-1:iy+2, ix] if axis == 0
                   else ncc[iy, ix-1:ix+2])
        curvature = a - 2*b + c
        return float(np.clip(0.5*(a-c)/curvature, -0.5, 0.5)) if curvature < 0 else 0.0

    return (sy-radius+iy+refine(0, iy), sx-radius+ix+refine(1, ix),
            float(ncc[iy, ix]), iy in (0, 2*radius) or ix in (0, 2*radius))


def read_reference(path, micrograph):
    """Read the relevant RELION particle loop without loading all micrographs."""
    labels, coordinates = [], []
    in_loop, in_data = False, False
    with Path(path).open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("data_") or line == "loop_":
                labels = []
                in_loop, in_data = line == "loop_", False
                continue
            if in_loop and line.startswith("_") and not in_data:
                labels.append(line.split()[0])
                continue
            if not in_loop or not labels:
                continue
            in_data = True
            required = ("_rlnMicrographName", "_rlnCoordinateX", "_rlnCoordinateY")
            if not all(k in labels for k in required):
                continue
            fields = shlex.split(line) if "'" in line or '"' in line else line.split()
            if len(fields) != len(labels):
                raise ValueError(f"Unexpected STAR row length in {path}")
            indices = [labels.index(k) for k in required]
            if Path(fields[indices[0]]).name == micrograph:
                coordinates.append([float(fields[indices[1]]), float(fields[indices[2]])])
    if not coordinates:
        raise ValueError(f"No reference coordinates found for {micrograph}")
    return np.asarray(coordinates)


def write_csv(path, records):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_star(path, records, micrograph):
    with Path(path).open("w") as handle:
        handle.write("# Coordinates: zero-based pixels; scores: final SMAP CC.\n")
        handle.write("data_particles\n\nloop_\n_rlnMicrographName #1\n")
        handle.write("_rlnCoordinateX #2\n_rlnCoordinateY #3\n")
        handle.write("_rlnAutopickFigureOfMerit #4\n")
        for r in records:
            handle.write(f"{micrograph} {r['x']:.4f} {r['y']:.4f} {r['score']:.3f}\n")


def plot_candidate_progress(entries, kept, offsets, out):
    ordered = sorted(kept, key=lambda r:(r['tile'], -float(r['score'])))
    index = np.arange(1, len(ordered)+1)
    score = np.array([float(r['score']) for r in ordered])
    matched = np.array([str(r.get('reference_id', '')).strip() != '' for r in ordered])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7), layout='constrained')
    for flag, color, label in [(False, '#9CA8B0', 'Unmatched candidate'),
                               (True, '#2878B5', 'Reference-position match')]:
        mask = matched == flag
        axes[0].scatter(index[mask], score[mask], s=17, c=color,
                        alpha=0.8, edgecolors='none', label=f'{label} ({mask.sum()})')
    axes[0].set(xlabel='Cumulative merged candidate number (ordered by section)',
                ylabel='Final optimized CC (SMAP score)',
                title='One dot per merged candidate')
    axes[0].legend(frameon=False, fontsize=8)
    rows = []
    raw_total = merged_total = matched_total = 0
    for i, offset in enumerate(offsets):
        name = offset['tile']
        raw = sum(r['tile'] == name for r in entries)
        selected = [r for r in kept if r['tile'] == name]
        hits = sum(str(r.get('reference_id', '')).strip() != '' for r in selected)
        raw_total += raw
        merged_total += len(selected)
        matched_total += hits
        rows.append(dict(tile=name, raw_entries=raw, merged_representatives=len(selected),
                         reference_matches=hits, cumulative_raw=raw_total,
                         cumulative_merged=merged_total, cumulative_matches=matched_total))
    tile_index = np.arange(1, len(rows)+1)
    for key,color,label in [('cumulative_raw','#9CA8B0','Entries including overlaps'),
                            ('cumulative_merged','#2878B5','Merged candidates'),
                            ('cumulative_matches','#D18A20','Reference-position matches')]:
        values = [r[key] for r in rows]
        axes[1].plot(tile_index, values, '-o', color=color, lw=1.3,
                     ms=2.5, label=f'{label} ({values[-1]})')
    axes[1].set(xlabel='Section number', ylabel='Cumulative count',
                title='Counts accumulated by representative section')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=0.15)
    for ext in ('png','pdf'):
        fig.savefig(out / f'candidate_progress.{ext}',dpi=300)
    plt.close(fig)
    write_csv(out / 'candidate_progress.csv', rows)


def prepare_tiles(source, destination, angle):
    if angle <= 0:
        raise ValueError('Angular spacing must be positive.')
    offsets = list(csv.DictReader((source / 'tile_offsets.csv').open()))
    # Validate source paths before creating a new configuration tree.
    prepared = []
    for row in offsets:
        old = source / row['tile']
        text = (old / 'search.par').read_text()
        params = read_params(old / 'search.par')
        image_name = Path(params['imageFile'][0]).name
        if not (old / image_name).is_file():
            raise FileNotFoundError(old / image_name)
        prepared.append((row['tile'],text,image_name,old))
    destination.mkdir(exist_ok=False)
    shutil.copy2(source / 'tile_offsets.csv', destination / 'tile_offsets.csv')
    for name,text,image_name,old in prepared:
        new = destination / name
        new.mkdir()
        (new / 'output').mkdir()
        os.link(old / image_name, new / image_name)
        folder = f'/work/{destination.as_posix()}/{name}'
        changes = dict(nCores='1',imageFile=f'{folder}/{image_name}',
                       outputDir=f'{folder}/output',angle_inc=str(angle),
                       psi_inc=str(angle),inc_degrees='0.5')
        lines, seen = [], set()
        for line in text.splitlines():
            fields = line.split()
            key = fields[0] if fields else ''
            if key in changes:
                line = f'{key} {changes[key]}'
                seen.add(key)
            lines.append(line)
        lines.extend(f'{k} {v}' for k,v in changes.items() if k not in seen)
        (new / 'search.par').write_text('\n'.join(lines)+'\n')
    print(f'Prepared {len(offsets)} sections at {angle:g} degrees in {destination}')


def compare_runs(run_paths, out, reference, dedup_radius, match_radius):
    """Ensure both runs are merged, then plot two angular-setting lines."""
    summaries = []
    for path in run_paths:
        run = Path(path)
        folder = run / 'review/merged'
        if not ((folder / 'detected_particles.csv').exists()
                and (folder / 'summary.txt').exists()):
            folder = run / 'review/merged_v3'
            if not ((folder / 'detected_particles.csv').exists()
                    and (folder / 'summary.txt').exists()):
                subprocess.run([
                    sys.executable, str(Path(__file__).resolve()), '--tiles', str(run),
                    '--reference', str(reference), '--out', str(folder),
                    '--dedup-radius', str(dedup_radius), '--match-radius', str(match_radius)
                ], check=True)
        rows = list(csv.DictReader((folder / 'detected_particles.csv').open()))
        offsets = list(csv.DictReader((run / 'tile_offsets.csv').open()))
        first = run / offsets[0]['tile'] / 'search.par'
        angle = float(read_params(first)['angle_inc'][0])
        merged = np.array([sum(r['tile']==o['tile'] for r in rows) for o in offsets])
        matches = np.array([sum(r['tile']==o['tile'] and bool(r['reference_id'].strip())
                                for r in rows) for o in offsets])
        raw = list(csv.DictReader((folder / 'all_candidates_registration.csv').open()))
        plot_candidate_progress(raw, rows, offsets, folder)
        summaries.append((run, angle, np.cumsum(merged), np.cumsum(matches), folder))
        print(f'\n{angle:g} degree run:\n'+(folder/'summary.txt').read_text(),flush=True)
    if len(summaries[0][2]) != len(summaries[1][2]):
        raise ValueError('Runs have different numbers of sections.')
    # Verify the comparison uses exactly the same section geometry/order.
    geometry = [list(csv.DictReader((Path(p)/'tile_offsets.csv').open())) for p in run_paths]
    if geometry[0] != geometry[1]:
        raise ValueError('Section geometry differs between runs.')
    out.mkdir(parents=True,exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for (run,angle,merged,matches,folder),color in zip(summaries,['#2878B5','#D18A20']):
        x = np.arange(1,len(merged)+1)
        axes[0].plot(x,merged,'-o',ms=3,lw=1.6,color=color,
                     label=f'{angle:g} degrees ({merged[-1]})')
        axes[1].plot(x,matches,'-o',ms=3,lw=1.6,color=color,
                     label=f'{angle:g} degrees ({matches[-1]})')
    axes[0].set(title='Merged candidate counts',ylabel='Cumulative merged candidates')
    axes[1].set(title='Reference-position matches',ylabel='Cumulative reference matches')
    for ax in axes:
        ax.set_xlabel('Section number')
        ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=0.15)
        ax.legend(frameon=False,fontsize=9)
    for ext in ('png','pdf'):
        fig.savefig(out/f'cumulative_angular_comparison.{ext}',dpi=300)
    plt.close(fig)
    data = []
    for i in range(len(summaries[0][2])):
        row = {'section_number':i+1}
        for _,angle,merged,matches,_ in summaries:
            row[f'cumulative_candidates_{angle:g}deg']=int(merged[i])
            row[f'cumulative_matches_{angle:g}deg']=int(matches[i])
        data.append(row)
    write_csv(out/'cumulative_angular_comparison.csv',data)
    print(f'Comparison saved: {(out/"cumulative_angular_comparison.png").resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", default="tiles_5deg")
    parser.add_argument("--reference", default="refined_particles.star")
    parser.add_argument("--out", default="tiles_5deg/review/merged")
    parser.add_argument("--dedup-radius", type=float, default=20)
    parser.add_argument("--match-radius", type=float, default=20)
    parser.add_argument("--plot-existing", action="store_true",
                        help="Plot an existing merged output without rerunning registration.")
    parser.add_argument("--completed-only", action="store_true",
                        help="Use only sections with the final completion marker; omit interrupted or unstarted sections.")
    parser.add_argument("--prepare-angle", type=float,
                        help="Prepare a new --tiles tree at this angular spacing, then exit.")
    parser.add_argument("--source-tiles", default="tiles_5deg")
    parser.add_argument("--compare-runs", nargs=2, metavar=('RUN_A','RUN_B'),
                        help="Merge if needed and overlay cumulative counts for two runs.")
    args = parser.parse_args()
    root = Path.cwd()
    tiles = Path(args.tiles)
    out = Path(args.out)
    if args.compare_runs:
        compare_runs(args.compare_runs, out, args.reference,
                     args.dedup_radius, args.match_radius)
        return
    if args.prepare_angle is not None:
        if tiles.is_absolute() or '..' in tiles.parts:
            raise ValueError('Use a project-relative --tiles path for Docker mounting.')
        prepare_tiles(Path(args.source_tiles), tiles, args.prepare_angle)
        return
    if args.plot_existing:
        entries = list(csv.DictReader((out / 'all_candidates_registration.csv').open()))
        kept = list(csv.DictReader((out / 'detected_particles.csv').open()))
        offsets = list(csv.DictReader((tiles / 'tile_offsets.csv').open()))
        plot_candidate_progress(entries, kept, offsets, out)
        print(f'Saved cumulative candidate plots in {out.resolve()}')
        return
    if args.dedup_radius <= 0 or args.match_radius <= 0:
        raise ValueError("Radii must be positive.")
    if out.exists():
        raise FileExistsError(f"Output exists: {out}; use a new --out to preserve it.")
    offsets = list(csv.DictReader((tiles / "tile_offsets.csv").open()))
    tile_status = []
    for offset in offsets:
        log = tiles / offset['tile'] / 'run.log'
        complete = log.is_file() and 'Done combining search output' in log.read_text()
        tile_status.append(dict(tile=offset['tile'], completed=complete))
    incomplete = [r['tile'] for r in tile_status if not r['completed']]
    if incomplete and not args.completed_only:
        raise RuntimeError(f"Incomplete sections: {', '.join(incomplete)}. Use --completed-only to review a partial batch.")
    completed_names = {r['tile'] for r in tile_status if r['completed']}
    offsets = [r for r in offsets if r['tile'] in completed_names]
    if not offsets:
        raise RuntimeError('No completed sections to merge.')
    # The source micrograph is the only fractions MRC in the project.
    candidates = sorted(root.glob("*_fractions.mrc"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one *_fractions.mrc in current directory.")
    micrograph = candidates[0]
    refs = read_reference(args.reference, micrograph.name)
    out.mkdir(parents=True)
    write_csv(out/'tile_completion.csv', tile_status)
    (out/'completed_tiles.txt').write_text('\n'.join(r['tile'] for r in offsets)+'\n')
    entries = []
    empty_sections = 0
    for offset in offsets:
        tile = tiles / offset["tile"]
        log = tile / "run.log"
        log_text = log.read_text() if log.exists() else ""
        if "Done combining search output" not in log_text:
            raise RuntimeError(f"Incomplete section: {tile.name}")
        scores = section_scores(tile, log_text)
        if len(scores) == 0:
            empty_sections += 1
            print(f"Completed {tile.name}: no candidates", flush=True)
            continue
        params = read_params(tile / "search.par")
        image = read_array(tile / Path(params["imageFile"][0]).name)
        seeds = matrix(tile / "output/list_final.txt")
        angles = matrix(tile / "output/particles.par", skiprows=1)
        patches = read_array(tile / "output/particles.mrc")
        templates_path = tile / "output/templates.mrc"
        with mrcfile.open(templates_path) as t:
            ntemplates = 1 if t.data.ndim == 2 else len(t.data)
        if patches.ndim == 2:
            patches = patches[None]
        if not (len(scores) == len(seeds) == len(angles) == len(patches) == ntemplates):
            raise ValueError(f"Particle file row counts differ: {tile.name}")
        if int(params["nCores"][0]) != 1:
            raise ValueError("This exporter requires the single-worker runs used here.")
        ratio = float(params["aPerPix_search"][0]) / float(params["aPerPix"][0])
        # search_mean.mrc gives the actual padded search-grid dimensions.
        with mrcfile.open(tile / "output/search_mean.mrc") as m:
            search_shape = np.array(m.data.shape[-2:])
        center_search = search_shape // 2 + 1
        center_original = np.array(image.shape) // 2 + 1
        for j, (score_row, seed, angle, patch) in enumerate(zip(scores, seeds, angles, patches)):
            if int(score_row[0]) != j+1 or int(angle[0]) != j+1:
                raise ValueError(f"Unexpected stack indexing in {tile.name}")
            seed_rc = np.floor((seed[1:3]-center_search)*ratio + center_original + 0.5)-1
            y, x, quality, boundary = register_patch(image, patch, *seed_rc)
            margin = min(x, y, image.shape[1]-1-x, image.shape[0]-1-y)
            entries.append(dict(
                tile=tile.name, particle_id=j+1,
                x=float(x+int(offset["x0_zero_based"])),
                y=float(y+int(offset["y0_zero_based"])), score=float(score_row[1]),
                local_x=float(x), local_y=float(y), registration_ncc=quality,
                registration_boundary=bool(boundary),
                fully_contained=bool(margin >= max(patch.shape)/2),
                frealign_psi=float(angle[1]), frealign_theta=float(angle[2]),
                frealign_phi=float(angle[3]), defocus_u_A=float(angle[8]),
                defocus_v_A=float(angle[9]), astigmatism_angle_deg=float(angle[10]),
                particle_image=f"{j+1:06d}@{tile / 'output/particles.mrc'}",
                template_image=f"{j+1:06d}@{templates_path}",
            ))
        print(f"Registered {tile.name}: {len(scores)} patches", flush=True)
    if not entries:
        raise RuntimeError("No candidates found.")
    write_csv(out / "all_candidates_registration.csv", entries)
    bad = [r for r in entries if r["registration_ncc"] < 0.5 or r["registration_boundary"]]
    if bad:
        raise RuntimeError(f"{len(bad)} registrations need review. See all_candidates_registration.csv; no detection STAR exported.")
    # Prefer complete patches, then strongest final score. Only suppress entries
    # from a different section; nearby entries from one section are retained.
    kept, assignments = [], []
    for r in sorted(entries, key=lambda t:(t["fully_contained"],t["score"]), reverse=True):
        close = [(i,np.hypot(r["x"]-k["x"],r["y"]-k["y"]))
                 for i,k in enumerate(kept) if r["tile"] != k["tile"]]
        close = [(i,d) for i,d in close if d <= args.dedup_radius]
        if close:
            i,d = min(close, key=lambda p:p[1])
            assignments.append(dict(tile=r["tile"],particle_id=r["particle_id"],
                                    representative_id=i+1,distance_pixels=float(d)))
        else:
            kept.append(r)
            assignments.append(dict(tile=r["tile"],particle_id=r["particle_id"],
                                    representative_id=len(kept),distance_pixels=0.0))
    xy = np.array([[r["x"],r["y"]] for r in kept])
    distances = np.linalg.norm(xy[:,None,:]-refs[None,:,:], axis=2)
    # Dummy columns allow unmatched detections. The large unmatched cost favors
    # maximum cardinality among assignments inside the chosen match radius.
    penalty = (len(kept)+len(refs)+1)*args.match_radius
    cost = np.full((len(kept),len(refs)+len(kept)), penalty)
    cost[:,:len(refs)] = np.where(distances <= args.match_radius, distances, penalty*3)
    row_idx,col_idx = linear_sum_assignment(cost)
    matches = {int(i):int(j) for i,j in zip(row_idx,col_idx) if j < len(refs) and distances[i,j] <= args.match_radius}
    for i,r in enumerate(kept):
        r["detection_id"] = i+1
        r["reference_id"] = matches[i]+1 if i in matches else ""
        r["reference_distance_pixels"] = float(distances[i,matches[i]]) if i in matches else ""
    write_csv(out / "detected_particles.csv", kept)
    write_csv(out / "overlap_assignments.csv", assignments)
    write_star(out / "detected_particles.star", kept, micrograph.name)
    matched = [r for r in kept if r["reference_id"] != ""]
    if matched:
        write_star(out / "reference_matched_detections.star", matched, micrograph.name)
    unmatched = [r for r in kept if r["reference_id"] == ""]
    if unmatched:
        write_star(out / "unmatched_detections.star", unmatched, micrograph.name)
    selected_refs = [dict(reference_id=i+1,x=float(p[0]),y=float(p[1]),
                          matched=bool(i in matches.values())) for i,p in enumerate(refs)]
    write_csv(out / "reference_particles.csv", selected_refs)
    plot_candidate_progress(entries, kept, offsets, out)
    ref_star = [dict(x=p[0],y=p[1],score=0.0) for p in refs]
    write_star(out / "reference_particles.star", ref_star, micrograph.name)
    image = read_array(micrograph)
    fig,ax = plt.subplots(figsize=(11,8),layout="constrained")
    low,high = np.percentile(image[::4,::4],[1,99])
    ax.imshow(image[::2,::2],cmap="gray",vmin=low,vmax=high,origin="lower",
              extent=(-0.5,image.shape[1]-0.5,-0.5,image.shape[0]-0.5),interpolation="nearest")
    ax.scatter(refs[:,0],refs[:,1],s=90,facecolors="none",edgecolors="#F0B43C",lw=1,label=f"Reference picks ({len(refs)})")
    for selected,color,label in [(matched,"#29B7E8","Reference-matched detections"),
                                 (unmatched,"#E55F72","Unmatched detections")]:
        if selected:
            coords = np.array([[r["x"],r["y"]] for r in selected])
            ax.scatter(coords[:,0],coords[:,1],s=28,c=color,marker="+",lw=1,label=f"{label} ({len(selected)})")
    ax.set(xlabel="X (original pixels)",ylabel="Y (original pixels)",
           title=f"RNF213: {len(kept)} merged candidates; {len(matches)}/{len(refs)} reference picks matched\nDeduplication and matching radii: {args.dedup_radius:g}, {args.match_radius:g} pixels")
    ax.legend(fontsize=8,loc="upper right",facecolor="white",framealpha=0.85)
    for ext in ("png","pdf"):
        fig.savefig(out / f"particle_map.{ext}",dpi=250)
    text = (f"Sections: {len(offsets)}\nSkipped incomplete sections: {len(incomplete)}\nEmpty sections: {empty_sections}\nCandidate entries: {len(entries)}\n"
            f"Merged candidates: {len(kept)}\nReference picks for this micrograph: {len(refs)}\n"
            f"One-to-one reference matches within {args.match_radius:g} px: {len(matches)}\n"
            f"Reference-pick recovery: {len(matches)/len(refs):.1%}\n"
            f"Unmatched candidates: {len(unmatched)} (not automatically false positives)\n"
            f"Minimum patch registration NCC: {min(r['registration_ncc'] for r in entries):.3f}\n"
            "Coordinates: zero-based pixels, approximately subpixel registered to saved patches.\n"
            "Reference picks are a selected set, not an exhaustive ground truth.\n"
            "Frealign angles are preserved in CSV; orientations have not been validated.\n")
    (out / "summary.txt").write_text(text)
    print("\n"+text+f"Outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
