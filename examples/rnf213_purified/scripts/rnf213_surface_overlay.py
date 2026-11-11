#!/usr/bin/env python3
"""Render searched RNF213 surfaces at physical scale on the original micrograph.

Run from the RNF213 project directory. Uses only accepted review detections.
Before rendering, reproduces SMAP's final, unwhitened templates to check the
Frealign-to-model rotation conversion. This checks display conventions, not
the biological correctness of fitted orientations. No search inputs change.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import mrcfile
from scipy import fft, ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# Ahel et al., eLife 2020, doi:10.7554/eLife.56185. Author residue numbers,
# not mmCIF label_seq_id (which can include expression-tag offsets).
DOMAIN_COLORS = {
    "N-arm": "#28b9d0",
    "Linker / hinge": "#a9b0bc",
    "AAA core": "#efa440",
    "E3 module": "#9775cf",
    "RING": "#db668c",
    "CTD": "#63b995",
    "UBE2L3": "#edcf55",
    "Other": "#dddddd",
}


def domain_label(chain, residue):
    """Domain of a residue: UBE2L3 for chain B, otherwise the RNF213 domain by residue number."""
    if chain == "B":
        return "UBE2L3"
    if chain != "A" or residue < 1 or residue > 5148:
        return "Other"
    if residue <= 1290:
        return "N-arm"
    if residue <= 1774 or 3406 <= residue <= 3588:
        return "Linker / hinge"
    if residue <= 3405:
        return "AAA core"
    if 3940 <= residue <= 3999:
        return "RING"
    return "E3 module" if residue <= 4926 else "CTD"


def cif_atoms(path):
    """Atom coordinates, weights and domain labels of the model (requires gemmi)."""
    import gemmi

    block = gemmi.cif.read_file(str(path)).sole_block()
    table = block.find(
        "_atom_site.",
        [
            "group_PDB",
            "type_symbol",
            "auth_asym_id",
            "auth_seq_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "pdbx_PDB_model_num",
        ],
    )
    if len(table) == 0:
        raise ValueError("The CIF atom table lacks required author numbering / model fields.")
    points, weights, labels = [], [], []
    for row in table:
        # SMAP read_cif_file uses ATOM records, including all modeled chains.
        if row[0] != "ATOM" or row[7] != "1":
            continue
        points.append([float(row[j]) for j in (4, 5, 6)])
        weights.append(float(gemmi.Element(row[1]).atomic_number))
        labels.append(domain_label(row[2], int(row[3])))
    if not points or "AAA core" not in labels or "E3 module" not in labels:
        raise ValueError("Expected chain A RNF213 with author-numbered AAA and E3 residues.")
    return np.array(points), np.array(weights), np.array(labels)


def atom_density(points, weights, shape, bin_factor):
    """Atoms as a smoothed density on a binned grid, for registration to the potential."""
    density = np.zeros(shape, dtype=np.float32)
    original_center = (np.array(shape) * bin_factor) // 2
    # Centers of binned voxels: factor*i+(factor-1)/2, relative to original center.
    xyz = (points + original_center[::-1] - (bin_factor - 1) / 2) / bin_factor
    xyz = xyz[:, ::-1]
    base = np.floor(xyz).astype(int)
    fraction = xyz - base
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                offset = np.array([dz, dy, dx])
                index = base + offset
                valid = np.all((index >= 0) & (index < np.array(shape)), axis=1)
                w = weights * np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
                np.add.at(density, tuple(index[valid].T), w[valid])
    return ndimage.gaussian_filter(density, 1.5)


def register_atoms(volume, points_A, weights, apix):
    """Bring the atoms into the frame of the template potential: translation and a small
    grid-scale correction found by correlating their density with the volume. Returns
    (aligned points, info)."""
    # Translation / small grid-scale correction only: CIF and SP retain the same
    # axes. SMAP centers its final SP by density mass, not by atom count.
    centered = (points_A - np.average(points_A, axis=0, weights=weights)) / apix
    factor = 4
    n = volume.shape[0]
    if n % factor:
        raise ValueError("CIF registration currently requires model size divisible by four.")
    v = np.maximum(volume - mode_value(volume), 0)
    target = v.reshape(n // 4, 4, n // 4, 4, n // 4, 4).mean(axis=(1, 3, 5))
    target = ndimage.gaussian_filter(target, 1.5)
    target = target - target.mean()
    target_fft = fft.rfftn(target)
    target_norm = np.linalg.norm(target)
    candidates = []
    for scale in (0.98, 0.99, 1.0, 1.01, 1.02):
        density = atom_density(centered * scale, weights, target.shape, factor)
        density -= density.mean()
        cc = fft.irfftn(target_fft * np.conj(fft.rfftn(density)), s=target.shape).real
        cc /= target_norm * np.linalg.norm(density)
        # Limit permitted translation to +/- three coarse voxels (12 original px).
        lags = np.arange(-3, 4)
        indices = lags % target.shape[0]
        small = cc[np.ix_(indices, indices, indices)]
        peak = np.unravel_index(np.argmax(small), small.shape)
        integer_peak = lags[np.array(peak)]
        shift = integer_peak.astype(float)
        boundary = any(p in (0, 6) for p in peak)
        for axis in range(3):
            low, high = integer_peak.copy(), integer_peak.copy()
            low[axis] -= 1
            high[axis] += 1
            left, right = cc[tuple(low % target.shape[0])], cc[tuple(high % target.shape[0])]
            middle = small[peak]
            denominator = left - 2 * middle + right
            if denominator < -1e-9:
                shift[axis] += np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)
        candidates.append((float(small[peak]), scale, shift, boundary))
    score, scale, shift, boundary = max(candidates, key=lambda r: r[0])
    info = dict(
        coarse_density_NCC=score,
        grid_scale_correction=scale,
        translation_original_px_xyz=(shift[::-1] * factor).tolist(),
        registration_boundary=bool(boundary),
        method="CIF atomic-number density versus SP; translation and small scale correction only",
    )
    if score < 0.80 or boundary:
        raise RuntimeError(
            f"CIF-to-SP registration failed (NCC={score:.3f}, boundary={boundary}); no domain colors assigned."
        )
    return centered * scale + shift[::-1] * factor, info


def color_surface(mesh, volume, structure, apix, out):
    """Colour every surface vertex by the domain of the nearest registered atom."""
    points, weights, labels = cif_atoms(structure)
    aligned, info = register_atoms(volume, points, weights, apix)
    _, index = cKDTree(aligned).query(mesh.points)
    names = labels[index]
    from matplotlib.colors import to_rgb

    mesh.point_data["domain_rgb"] = np.array(
        [np.array(to_rgb(DOMAIN_COLORS[name])) * 255 for name in names], dtype=np.uint8
    )
    info.update(
        source="https://elifesciences.org/articles/56185",
        numbering="7OIK author residue numbers; chain A RNF213, chain B UBE2L3",
        assignment="Nearest aligned modeled atom to each SP surface vertex; broad domain visualization",
        colors=DOMAIN_COLORS,
        ranges={
            "N-arm": [1, 1290],
            "linker": [1291, 1774],
            "AAA core": [1775, 3405],
            "hinge": [3406, 3588],
            "E3 module": [3589, 4926],
            "RING": [3940, 3999],
            "CTD": [4927, 5148],
        },
    )
    (out / "domain_coloring.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"CIF-to-SP coloring registration: NCC={info['coarse_density_NCC']:.3f}", flush=True)
    return info


def read_csv(path):
    """Rows of a CSV file."""
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    """Write rows (dicts) as CSV; the keys of the first row are the header."""
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def parameters(path):
    """Settings of a search.par, with v2.1 defaults for the keys it does not set."""
    p = dict(
        aPerPix=1.072,
        Cs=0.0027,
        Cc=0.0027,
        V_acc=300000.0,
        deltaE=0.7,
        a_i=50e-6,
        F_abs=0.0,
        mask_params=[0.0, 3.0],
    )
    for line in Path(path).read_text().splitlines():
        t = line.split("#", 1)[0].split()
        if len(t) > 1 and t[0] in p:
            if t[0] == "mask_params":
                p[t[0]] = [float(v) for v in t[1:]]
            else:
                p[t[0]] = float(t[1])
    return p


def stack_image(ref, root):
    """One image 'j@stack' (1-based) of an .mrc stack."""
    index, name = ref.split("@", 1)
    path = Path(name)
    if name.startswith("/work/"):
        path = root / name[6:]
    elif not path.is_absolute():
        path = root / path
    with mrcfile.mmap(path) as m:
        return np.array(m.data if m.data.ndim == 2 else m.data[int(index) - 1], dtype=np.float32, copy=True)


def rotation(row, convention):
    """Rotation matrix of a candidate under one of the tested conventions for the saved angles."""
    # smap2frealign reverses its ZYZ Euler triplet. Check transpose and Rx(pi)
    # offsets numerically against SMAP outputs, rather than assuming conventions.
    a = Rotation.from_euler(
        "ZYZ",
        [float(row["frealign_phi"]), float(row["frealign_theta"]), float(row["frealign_psi"])],
        degrees=True,
    ).as_matrix()
    d = np.diag([1.0, -1.0, -1.0])
    matrices = {
        "A": a,
        "AT": a.T,
        "DAD": d @ a @ d,
        "DATD": d @ a.T @ d,
        "DA": d @ a,
        "AD": a @ d,
        "DAT": d @ a.T,
        "ATD": a.T @ d,
    }
    return matrices[convention]


def mode_value(volume):
    """Most frequent value of a volume (the background removed as SMAP does)."""
    # NumPy's sorted unique counts reproduce MATLAB mode for float32 input;
    # most SP volumes have a large exactly constant background population.
    values, counts = np.unique(volume, return_counts=True)
    return float(values[np.argmax(counts)])


def masked_fourier(volume, p):
    """Centred Fourier transform of the potential after SMAP's background removal and mask."""
    bg = mode_value(volume)
    v = volume.astype(np.float32, copy=True) - bg
    near, width = p["mask_params"]
    if width <= 0:
        raise ValueError("This helper requires a positive cosine mask width.")
    threshold = np.std(np.abs(v), ddof=1, dtype=np.float64)
    distance = ndimage.distance_transform_edt(np.abs(v) <= threshold)
    distance -= near
    np.clip(distance, 0, width, out=distance)
    distance *= np.pi / width
    v *= (0.5 + 0.5 * np.cos(distance)).astype(np.float32)
    del distance
    v += bg
    f = fft.fftshift(fft.fftn(fft.ifftshift(v), workers=2))
    center = tuple(s // 2 for s in v.shape)
    f[center] = 0
    return f


def ctf(row, n, p):
    """Complex CTF (N x N, corner origin) of one candidate's defocus, with SMAP's constants."""
    # Same constants and complex CTF as smap.def_consts / smap.ctf.
    h, me, e, c = 6.636e-34, 9.10938215e-31, 1.602e-19, 2.99792458e8
    voltage = p["V_acc"]
    wavelength = h / np.sqrt(2 * me * e * voltage * (1 + e * voltage / (2 * me * c * c)))
    coord = np.arange(n, dtype=np.float64) - n // 2
    x, y = np.meshgrid(coord, coord)
    alpha = np.arctan2(-y, x)
    alpha[n // 2, n // 2] = 0
    freq = np.hypot(x, y) / (2 * (n // 2) * p["aPerPix"]) * 1e10
    df1, df2 = float(row["defocus_u_A"]) * 1e-10, float(row["defocus_v_A"]) * 1e-10
    ast = -np.deg2rad(float(row["astigmatism_angle_deg"]))
    df = 0.5 * (df1 + df2 + (df1 - df2) * np.cos(2 * (alpha - ast)))
    chi = np.pi * wavelength * freq**2 * (df - 0.5 * p["Cs"] * wavelength**2 * freq**2)
    envelope = np.exp(
        -(((np.pi * wavelength * freq**2 * p["Cc"] * p["deltaE"]) / (4 * voltage * np.sqrt(np.log(2)))) ** 2)
        - ((np.pi * p["Cs"] * wavelength**2 * freq**3 - np.pi * df * freq) * p["a_i"]) ** 2 / np.log(2)
    )
    f = p["F_abs"]
    result = (
        (f * np.sin(chi) - (1 - f) * np.cos(chi)) + 1j * (-f * np.cos(chi) - (1 - f) * np.sin(chi))
    ) * envelope
    return fft.ifftshift(result)


def template_projection(fvolume, forward_rotation, transfer):
    """Template as SMAP's final optimisation makes it: projected potential as a phase object,
    the CTF applied to the wave, intensity |wave|^2."""
    n = fvolume.shape[0]
    cp = n // 2
    x, y = np.meshgrid(np.arange(n) - cp, np.arange(n) - cp)
    xyz = np.array([x.ravel(), y.ravel(), np.zeros(n * n)])
    # SMAP uses RM to sample Fourier coordinates; the object rotates by RM.T.
    k = forward_rotation.T @ xyz
    coordinates = np.array([k[2] + cp, k[1] + cp, k[0] + cp])
    real = ndimage.map_coordinates(
        fvolume.real, coordinates, order=1, mode="constant", cval=0, prefilter=False
    )
    imag = ndimage.map_coordinates(
        fvolume.imag, coordinates, order=1, mode="constant", cval=0, prefilter=False
    )
    potential = (real + 1j * imag).reshape(n, n)
    wave = np.exp(1j * fft.ifft2(fft.ifftshift(potential)))
    detected = fft.fftshift(fft.ifft2(fft.fft2(wave) * transfer))
    return np.abs(detected) ** 2


def correlation(a, b):
    """Normalised correlation of two arrays (NaN if either is constant)."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def checks_for(rows, count):
    """Candidates for the convention check: reference matches and other detections, with whole
    patches, up to count."""
    selected, used = [], set()
    # Include both reference matches and additional detections, with whole patches.
    for flag in (True, False):
        group = [
            r
            for r in rows
            if bool(r.get("reference_id", "").strip()) == flag
            and str(r.get("fully_contained", "True")).lower() in ("true", "1")
        ]
        group.sort(key=lambda r: -float(r["score"]))
        taken = 0
        for row in group:
            if row["tile"] in used:
                continue
            selected.append(row)
            used.add(row["tile"])
            taken += 1
            if taken >= max(1, count // 2):
                break
    return selected[:count]


def verify_convention(volume, rows, root, run, out):
    """Compare the saved templates with projections under every candidate convention and keep
    the one that reproduces them; the figure and table go to out."""
    names = ("A", "AT", "DAD", "DATD", "DA", "AD", "DAT", "ATD")
    selected = checks_for(rows, 4)
    if len(selected) < 2:
        raise ValueError("Need at least two complete patches for rotation checks.")
    base_p = parameters(root / run / selected[0]["tile"] / "search.par")
    print("Checking rotations against saved SMAP templates...", flush=True)
    fvolume = masked_fourier(volume, base_p)
    report, comparisons = [], []
    for row in selected:
        p = parameters(root / run / row["tile"] / "search.par")
        if p["mask_params"] != base_p["mask_params"]:
            raise ValueError("Mask parameters differ between check sections.")
        saved = stack_image(row["template_image"], root)
        if saved.shape != volume.shape[1:]:
            raise ValueError(f"Saved template shape {saved.shape} differs from the model.")
        transfer = ctf(row, saved.shape[0], p)
        record = dict(tile=row["tile"], particle_id=row["particle_id"], SNR=row["score"])
        for name in names:
            predicted = template_projection(fvolume, rotation(row, name), transfer)
            record[name] = correlation(saved, predicted)
        report.append(record)
        comparisons.append((row, saved, transfer))
        best = max(names, key=lambda name: record[name])
        print(f"  {row['tile']} particle {row['particle_id']}: {best}, NCC={record[best]:.4f}", flush=True)
    medians = {name: float(np.median([r[name] for r in report])) for name in names}
    ordered = sorted(names, key=lambda name: -medians[name])
    convention = ordered[0]
    passed = (
        medians[convention] >= 0.97
        and min(r[convention] for r in report) >= 0.94
        and medians[convention] - medians[ordered[1]] >= 0.02
    )
    write_csv(out / "rotation_check.csv", report)
    details = dict(
        convention=convention,
        median_template_NCC=medians[convention],
        alternative_medians=medians,
        passed=bool(passed),
        scope="Display conversion check against SMAP templates; fitted poses remain unvalidated.",
    )
    (out / "rotation_check.json").write_text(json.dumps(details, indent=2) + "\n")
    if not passed:
        raise RuntimeError(
            "Rotation check did not pass. No surfaces rendered. Send the printed NCC values and rotation_check.csv."
        )
    projected = [
        template_projection(fvolume, rotation(row, convention), transfer)
        for row, saved, transfer in comparisons
    ]
    del fvolume
    return convention, comparisons, projected, details


def make_surface(volume, apix, resolution, mass_fraction):
    """Isosurface mesh of the potential enclosing mass_fraction of its density, low-passed to
    resolution (requires pyvista)."""
    import pyvista as pv

    # Bin only the display mesh. Binned voxel i represents original position 2*i+.5.
    n = volume.shape[0]
    if n % 2:
        raise ValueError("Display binning requires an even model dimension.")
    b = volume.reshape(n // 2, 2, n // 2, 2, n // 2, 2).mean(axis=(1, 3, 5))
    b = ndimage.gaussian_filter(b, resolution / (2.355 * 2 * apix))
    b = np.maximum(b - mode_value(volume), 0)
    values = np.sort(b.ravel())[::-1]
    mass = np.cumsum(values, dtype=np.float64)
    if mass[-1] <= 0:
        raise ValueError("The model has no positive scattering-potential mass.")
    level = float(values[min(np.searchsorted(mass, mass_fraction * mass[-1]), len(values) - 1)])
    if level <= 0:
        raise ValueError("Invalid isosurface level; reduce --mass-fraction.")
    origin = (0.5 - n // 2,) * 3
    grid = pv.ImageData(dimensions=b.shape[::-1], spacing=(2.0,) * 3, origin=origin)
    grid.point_data["potential"] = b.ravel(order="C")
    surface = grid.contour([level], scalars="potential").triangulate()
    if surface.n_points == 0:
        raise ValueError("The selected contour produced no surface.")
    surface = surface.decimate_pro(0.65, preserve_topology=True)
    surface = surface.compute_normals(auto_orient_normals=True)
    return surface, level


def render_surfaces(mesh, rows, convention, out):
    """Render the surface at every candidate's pose, orthographic, one raster pixel per
    micrograph pixel. Returns (RGBA stamps, stamp size)."""
    import pyvista as pv

    side = int(np.ceil(2 * np.max(np.linalg.norm(mesh.points, axis=1)) + 12))
    side += side % 2
    # One raster pixel is one original micrograph pixel. Orthographic, no zoom.
    plotter = pv.Plotter(off_screen=True, window_size=(side, side), lighting="none")
    plotter.set_background("white")
    if "domain_rgb" in mesh.point_data:
        actor = plotter.add_mesh(
            mesh,
            scalars="domain_rgb",
            rgb=True,
            smooth_shading=True,
            ambient=0.16,
            diffuse=0.84,
            specular=0.15,
            specular_power=25,
            show_scalar_bar=False,
        )
    else:
        actor = plotter.add_mesh(
            mesh,
            color="#27c9de",
            smooth_shading=True,
            ambient=0.16,
            diffuse=0.84,
            specular=0.15,
            specular_power=25,
        )
    plotter.add_light(
        pv.Light(
            position=(-side, side, 2 * side),
            focal_point=(0, 0, 0),
            color="white",
            intensity=1.0,
            light_type="scene light",
        )
    )
    plotter.add_light(
        pv.Light(
            position=(side, -0.3 * side, side),
            focal_point=(0, 0, 0),
            color="white",
            intensity=0.35,
            light_type="scene light",
        )
    )
    # Initialize the off-screen window once; subsequent screenshots otherwise
    # can reuse the previous framebuffer after only changing actor.user_matrix.
    plotter.show(auto_close=False, interactive=False)
    plotter.camera.position = (0, 0, side * 3)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 1, 0)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = side / 2
    plotter.camera.clipping_range = (0.1, side * 6)
    result, records, matrices = [], [], []
    for i, row in enumerate(rows):
        transform = np.eye(4)
        transform[:3, :3] = rotation(row, convention)
        actor.user_matrix = transform
        plotter.render()
        image = np.array(plotter.screenshot(transparent_background=True, return_img=True), copy=True)
        result.append(image)
        matrices.append(transform[:3, :3])
        records.append(
            dict(
                tile=row["tile"],
                particle_id=row["particle_id"],
                psi=row["frealign_psi"],
                theta=row["frealign_theta"],
                phi=row["frealign_phi"],
                rendered_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            )
        )
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"Rendered {i + 1}/{len(rows)} accepted detections", flush=True)
    plotter.close()
    unique_frames = len({r["rendered_sha256"] for r in records})
    unique_poses = len({tuple(np.round(m, 5).ravel()) for m in matrices})
    print(
        f"Render check: {unique_frames} distinct images for {unique_poses} distinct fitted rotations",
        flush=True,
    )
    write_csv(out / "render_check.csv", records)
    if unique_poses > 1 and unique_frames <= 1:
        raise RuntimeError(
            "All surface screenshots are identical despite different poses; redraw check failed."
        )
    return result, side


def rois_for(rows, offsets, count=3):
    """count regions of interest in different sections, centred on dense groups of candidates."""
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    candidates = []
    for o in offsets:
        x, y, w, h = (int(o[k]) for k in ("x0_zero_based", "y0_zero_based", "width", "height"))
        inside = (
            (xy[:, 0] >= x + 170)
            & (xy[:, 0] < x + w - 170)
            & (xy[:, 1] >= y + 170)
            & (xy[:, 1] < y + h - 170)
        )
        candidates.append((int(inside.sum()), (x, y, w, h)))
    chosen = []
    for score, box in sorted(candidates, key=lambda pair: -pair[0]):
        center = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
        if all(np.linalg.norm(center - np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])) > 1000 for b in chosen):
            chosen.append(box)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise ValueError("Not enough separated sections for zoom panels.")
    return chosen


def scale_bar(ax, box, apix, nm):
    """Scale bar of nm nanometres in the lower part of a box."""
    x, y, w, h = box
    start = x + 0.06 * w
    ypos = y + 0.07 * h
    length = nm * 10 / apix
    ax.plot([start, start + length], [ypos, ypos], color="white", lw=3, solid_capstyle="butt")
    ax.text(start + length / 2, ypos + 0.035 * h, f"{nm:g} nm", color="white", ha="center", fontsize=9)


def reviewed_offsets(run, settings):
    """Tile offsets of the sections covered by the review (all tiles if the review covered all)."""
    offsets = read_csv(run / "tile_offsets.csv")
    selected = settings.get("reviewed_tiles")
    if selected is None:
        return offsets, len(offsets)
    selected = set(selected)
    available = {r["tile"] for r in offsets}
    if not selected or not selected <= available:
        raise ValueError("Review coverage does not match this run's tile geometry.")
    return [r for r in offsets if r["tile"] in selected], len(offsets)


def footprint_mask(shape, offsets, stride=4):
    """Union of reviewed IMAGE footprints, not SMAP's edge-excluded search area."""
    h, w = shape
    xs = np.arange(0, w, stride)
    ys = np.arange(0, h, stride)
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    for row in offsets:
        x, y, bw, bh = (int(row[k]) for k in ("x0_zero_based", "y0_zero_based", "width", "height"))
        if x < 0 or y < 0 or bw <= 0 or bh <= 0 or x + bw > w or y + bh > h:
            raise ValueError("Reviewed tile footprint is outside the original image.")
        mask |= (ys[:, None] >= y) & (ys[:, None] < y + bh) & (xs[None, :] >= x) & (xs[None, :] < x + bw)
    return mask


def display_background(micrograph, box, overview=False):
    """The micrograph region of box, low-passed and binned for display, with its contrast limits."""
    x, y, bw, bh = box
    stride = 4 if overview else 1
    crop = micrograph[y : y + bh, x : x + bw]
    image = ndimage.gaussian_filter(crop[::stride, ::stride], 0.6 if overview else 1.0)
    low, high = np.percentile(image, [1, 99])
    extent = [
        x - stride / 2,
        x + (image.shape[1] - 0.5) * stride,
        y - stride / 2,
        y + (image.shape[0] - 0.5) * stride,
    ]
    return image, float(low), float(high), extent


def draw_image_panel(ax, background, box, rows, stamps, side, overlay=True):
    """Micrograph panel with the rendered surfaces of the candidates inside it; returns how many
    were drawn."""
    image, low, high, extent = background
    ax.imshow(image, cmap="gray", origin="lower", vmin=low, vmax=high, extent=extent)
    x, y, bw, bh = box
    shown = 0
    for row, stamp in zip(rows, stamps):
        px, py = float(row["x"]), float(row["y"])
        if x - side / 2 <= px < x + bw + side / 2 and y - side / 2 <= py < y + bh + side / 2:
            if overlay:
                ax.imshow(
                    stamp,
                    origin="upper",
                    alpha=0.97,
                    extent=[px - side / 2, px + side / 2, py - side / 2, py + side / 2],
                )
            shown += int(x <= px < x + bw and y <= py < y + bh)
    ax.set(xlim=(x - 0.5, x + bw - 0.5), ylim=(y - 0.5, y + bh - 0.5), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return shown


def add_domain_legend(fig):
    """Legend of the domain colours."""
    from matplotlib.lines import Line2D

    legend = [
        Line2D([], [], marker="s", linestyle="", color=color, label=name, markersize=8)
        for name, color in DOMAIN_COLORS.items()
        if name != "Other"
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=7, frameon=False, fontsize=9)


def make_figures(
    micrograph,
    rows,
    stamps,
    side,
    boxes,
    comparisons,
    projected,
    convention,
    apix,
    cutoff,
    run,
    out,
    offsets,
    total_tiles,
):
    """Overview of the micrograph with every accepted candidate drawn, and the ROI close-ups."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    h, w = micrograph.shape
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    panels = [(0, 0, w, h)] + boxes
    for index, (ax, box) in enumerate(zip(axes.ravel(), panels)):
        x, y, bw, bh = box
        background = display_background(micrograph, box, overview=index == 0)
        shown = draw_image_panel(ax, background, box, rows, stamps, side)
        if index == 0:
            mask = footprint_mask(micrograph.shape, offsets)
            # Purple shading is a categorical coverage annotation, not density.
            shade = np.zeros((*mask.shape, 4))
            shade[~mask] = (0.30, 0.12, 0.40, 0.55)
            ax.imshow(shade, origin="lower", extent=background[3], interpolation="nearest")
            if mask.any() and not mask.all():
                ax.contour(
                    np.arange(0, w, 4),
                    np.arange(0, h, 4),
                    mask.astype(float),
                    levels=[0.5],
                    colors=["#83dfde"],
                    linewidths=0.9,
                )
            for j, (rx, ry, rw, rh) in enumerate(boxes, 1):
                ax.add_patch(Rectangle((rx - 0.5, ry - 0.5), rw, rh, fill=False, edgecolor="#ffdb6e", lw=1.2))
                ax.text(rx + 20, ry + rh - 70, str(j), color="#ffdb6e", weight="bold")
            ax.set_title(
                f"Full micrograph · {len(offsets)}/{total_tiles} sections reviewed\n"
                f"{len(rows)} detected particle candidates · final SNR ≥ {cutoff:g}",
                fontsize=11,
            )
            ax.text(
                0.99,
                0.02,
                "Purple: outside reviewed image footprints",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
            )
            scale_bar(ax, box, apix, 100)
        else:
            ax.set_title(f"ROI {index} · {shown} candidate centers · physical scale", fontsize=11)
            scale_bar(ax, box, apix, 20)
    fig.suptitle(f"RNF213 · {run} · fitted 7OIK surfaces", fontsize=15)
    add_domain_legend(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"rnf213_surface_overlay.{ext}", dpi=250)
    plt.close(fig)


def make_raw_overlay_pairs(micrograph, rows, stamps, side, boxes, apix, cutoff, run, out):
    """For each ROI: the micrograph alone and with the surfaces, same contrast and scale."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(boxes), 2, figsize=(11, 5 * len(boxes)), squeeze=False, layout="constrained")
    records = []
    for i, box in enumerate(boxes):
        # Compute once, share the SAME array, limits, filtering and extent.
        background = display_background(micrograph, box)
        for col, ax in enumerate(axes[i]):
            shown = draw_image_panel(ax, background, box, rows, stamps, side, overlay=col == 1)
            title = "Experimental image" if col == 0 else f"Fitted surfaces · {shown} candidate centers"
            ax.set_title(f"ROI {i + 1} · {title}", fontsize=11)
            scale_bar(ax, box, apix, 20)
        records.append(
            dict(
                roi=i + 1,
                box_xywh=list(box),
                vmin=background[1],
                vmax=background[2],
                extent=background[3],
                gaussian_sigma_original_pixels=1.0,
            )
        )
    fig.suptitle(
        f"RNF213 · {run} · final SNR ≥ {cutoff:g}\n"
        "Paired views share contrast and scale; surfaces show fitted poses",
        fontsize=14,
    )
    add_domain_legend(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"rnf213_raw_overlay_pairs.{ext}", dpi=220)
    plt.close(fig)
    return records


def orientation_gallery(rows, stamps, convention, out):
    """Surfaces of detections with different fitted rotations."""
    import matplotlib.pyplot as plt

    matrices = np.array([rotation(r, convention) for r in rows])
    # Farthest rotations give a useful visual check, rather than neighboring
    # candidates that might truly have similar preferred views.
    selected = [int(np.argmax([float(r["score"]) for r in rows]))]
    while len(selected) < min(8, len(rows)):
        closest = np.full(len(rows), np.inf)
        for i in selected:
            relative = matrices @ matrices[i].T
            angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
            closest = np.minimum(closest, angle)
        closest[selected] = -1
        selected.append(int(np.argmax(closest)))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    for ax, i in zip(axes.ravel(), selected):
        row = rows[i]
        ax.imshow(stamps[i])
        ax.set_title(
            f"{row['tile']} / {row['particle_id']} · SNR {float(row['score']):.2f}\n"
            f"ψ {float(row['frealign_psi']):.1f}° · θ {float(row['frealign_theta']):.1f}° · φ {float(row['frealign_phi']):.1f}°",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Different fitted rotations · shaded domain surfaces")
    fig.savefig(out / "orientation_gallery.png", dpi=200)
    plt.close(fig)


def template_check_figure(comparisons, projected, out):
    """Saved templates next to the re-projected ones under each tested convention."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(comparisons), 3, figsize=(8, 2.65 * len(comparisons)), squeeze=False, layout="constrained"
    )
    for i, ((row, saved, transfer), predicted) in enumerate(zip(comparisons, projected)):
        experimental = stack_image(row["particle_image"], Path.cwd())
        for ax, image, title in zip(
            axes[i],
            (experimental, saved, predicted),
            ("Experimental patch", "Saved SMAP template", "Reproduced template"),
        ):
            lo, hi = np.percentile(image, [1, 99])
            ax.imshow(image, cmap="gray", origin="lower", vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(f"{row['tile']} / {row['particle_id']}\nSNR {float(row['score']):.2f}")
        axes[i, 2].set_xlabel(f"NCC {correlation(saved, predicted):.4f}")
    fig.suptitle("Rotation conversion check against saved templates")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"rotation_check.{ext}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tiles_3p8")
    parser.add_argument("--review", default="review_current")
    parser.add_argument("--model", default="pilot/output_named/7OIK_SP.mrc")
    parser.add_argument("--structure", default="7OIK.cif")
    parser.add_argument("--micrograph")
    parser.add_argument("--resolution", type=float, default=8.0, help="Display smoothing FWHM in Angstroms.")
    parser.add_argument(
        "--mass-fraction", type=float, default=0.85, help="Positive SP mass enclosed by display contour."
    )
    args = parser.parse_args()
    if args.resolution <= 0 or not 0 < args.mass_fraction < 1:
        parser.error("Use positive resolution and a mass fraction strictly between 0 and 1.")
    root = Path.cwd()
    run = Path(args.run)
    review = Path(args.review)
    rows = read_csv(review / run.name / "detected_particles.csv")
    settings = json.loads((review / "review_settings.json").read_text())
    cutoff = float(settings["score_cutoff"])
    offsets, total_tiles = reviewed_offsets(run, settings)
    if any(r["tile"] not in {o["tile"] for o in offsets} for r in rows):
        raise ValueError("Accepted table includes a tile outside the reviewed coverage.")
    if not rows or any(float(r["score"]) < cutoff for r in rows):
        raise ValueError("Accepted table is empty or contains below-cutoff entries.")
    required = ("x", "y", "score", "frealign_psi", "frealign_theta", "frealign_phi")
    if not all(np.isfinite(float(r[k])) for r in rows for k in required):
        raise ValueError("Invalid coordinates, SNR or angles.")
    out = review / "overlays" / run.name
    out.mkdir(parents=True, exist_ok=True)
    with mrcfile.open(args.model) as m:
        volume = np.array(m.data, dtype=np.float32, copy=True)
        voxel = np.array([m.voxel_size.x, m.voxel_size.y, m.voxel_size.z], dtype=float)
        if (int(m.header.mapc), int(m.header.mapr), int(m.header.maps)) != (1, 2, 3):
            raise ValueError("Nonstandard MRC axes require explicit handling.")
    p = parameters(run / rows[0]["tile"] / "search.par")
    apix = p["aPerPix"]
    if volume.ndim != 3 or len(set(volume.shape)) != 1 or not np.allclose(voxel, apix, atol=1e-5):
        raise ValueError("Model must be cubic with voxels matching original image pixel size.")
    paths = [Path(args.micrograph)] if args.micrograph else list(root.glob("*_fractions.mrc"))
    if len(paths) != 1:
        raise ValueError("Use --micrograph to select exactly one original image.")
    with mrcfile.open(paths[0]) as m:
        micrograph = np.array(m.data, dtype=np.float32, copy=True)
    if micrograph.ndim != 2:
        raise ValueError("Expected a 2D motion-corrected micrograph.")
    convention, checks, projected, details = verify_convention(volume, rows, root, run, out)
    print(f"Rotation check passed: {convention}, median NCC={details['median_template_NCC']:.4f}", flush=True)
    mesh, level = make_surface(volume, apix, args.resolution, args.mass_fraction)
    color_surface(mesh, volume, Path(args.structure), apix, out)
    stamps, side = render_surfaces(mesh, rows, convention, out)
    boxes = rois_for(rows, offsets)
    make_figures(
        micrograph,
        rows,
        stamps,
        side,
        boxes,
        checks,
        projected,
        convention,
        apix,
        cutoff,
        run.name,
        out,
        offsets,
        total_tiles,
    )
    paired_display = make_raw_overlay_pairs(
        micrograph, rows, stamps, side, boxes, apix, cutoff, run.name, out
    )
    orientation_gallery(rows, stamps, convention, out)
    template_check_figure(checks, projected, out)
    details.update(
        pixel_size_A=apix,
        model_voxel_size_A=voxel.tolist(),
        original_model_shape_zyx=list(volume.shape),
        display_resolution_A=args.resolution,
        surface_mass_fraction=args.mass_fraction,
        surface_level=level,
        accepted_detections=len(rows),
        final_SNR_cutoff=cutoff,
        roi_boxes_xywh=boxes,
        micrograph=str(paths[0]),
        reviewed_tiles=[o["tile"] for o in offsets],
        reviewed_tile_footprints=offsets,
        total_tiles=total_tiles,
        coverage_definition="Union of reviewed tile IMAGE footprints; excludes no SMAP edge margins",
        paired_display=paired_display,
        roi_selection="Three candidate-rich, separated reviewed sections; not a random sample",
    )
    (out / "overlay_settings.json").write_text(json.dumps(details, indent=2) + "\n")
    (out / "README.txt").write_text(
        "Surfaces show estimated fitted poses at original physical scale.\n"
        "The searched SP model is binned and smoothed only to render a readable surface.\n"
        "The isosurface is a visualization choice, not a molecular boundary measurement.\n"
        "Domain colors use Ahel et al. 2020 broad author-numbered residue ranges.\n"
        "Colors are assigned by nearest modeled atom after CIF-to-SP registration.\n"
        "Directional lighting shows surface depth; it does not encode density or SNR.\n"
        "Rotation convention was checked against saved SMAP templates.\n"
        "Biological orientations have not been independently validated.\n"
        "Purple overview shading marks area outside reviewed image footprints.\n"
        "Footprints include tile margins; shading is not an exact searchable-area mask.\n"
        "Paired experimental/overlay ROI views share contrast, filtering and physical scale.\n"
        "ROI backgrounds are Gaussian-smoothed by 1 original pixel for display.\n"
        "ROIs are candidate-rich selected sections, not an unbiased image sample.\n"
        "Accepted CSVs, maps, search outputs and SNR thresholds were not changed.\n"
    )
    print(f"Saved: {(out / 'rnf213_surface_overlay.png').resolve()}")
    print(f"Saved: {(out / 'rnf213_raw_overlay_pairs.png').resolve()}")
    print(f"Saved: {(out / 'rotation_check.png').resolve()}")
    print(f"Saved: {(out / 'orientation_gallery.png').resolve()}")


if __name__ == "__main__":
    main()
