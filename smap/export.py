"""Export SMAP detections to a RELION 3.1+ particle STAR file (also imported by cryoSPARC).

New in the port (not part of v2.1). Intended use: baited reconstruction / local
reconstruction of detected particles with ``relion_reconstruct`` or cryoSPARC.

Geometry (validated in tests/test_export.py against an independent real-space projector):

* SMAP builds the template for rotation R as the central slice T(k) = V(R^T [kx, ky, 0])
  of the potential V[row=y, col=x, page=z] (the .mrc as written/read by RELION: x = column).
  RELION's projector uses the same relation with A = Euler_angles2matrix(rot, tilt, psi),
  so the RELION angles are the ZYZ decomposition of R itself.
* The orientation exported is ``q_opt`` (particles.csv), the rotation whose template is
  stored in templates.mrc and gives score_final. In v2.1 mode (fix_optimize_angle 0) q_opt
  differs from the refined grid rotation by a small Nelder-Mead step (PORT_NOTES item 14).
* particles.par / RNF213 registration CSV: Frealign angles of Q = q_opt * diag(1,-1,-1);
  q_opt = diag(1,-1,-1) Rz(psi) Ry(theta) Rz(phi) diag(1,-1,-1).
* Defocus: rlnDefocusU/V = df1/df2 in Angstrom, rlnDefocusAngle = -ast_rad*180/pi
  (the inverse of SMAP's CTFFind input conversion, identical to particles.par ANGAST).
* Coordinates: x = column, y = row of the original micrograph. ``--coord-origin 0``
  (default) writes zero-based pixels like the RNF213 workflow STAR files; use 1 if the
  downstream program expects one-based pixels. The extracted SMAP patches
  (``rlnImageName j@particles.mrc``) are already centred, so rlnOriginX/YAngst = 0.

usage:
  python -m smap.export smap OUT_DIR [OUT_DIR ...] --micrograph NAME [--offset X0 Y0]
                        [--min-score 8] [--images] -o particles.star
  python -m smap.export rnf213 all_candidates_registration.csv --par search.par
                        --micrograph NAME [--min-score 8] -o particles.star
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from . import io as sio
from .utils import normalize_rm

D_FLIP = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------- RELION Euler convention

def relion_matrix(rot, tilt, psi) -> np.ndarray:
    """RELION Euler_angles2matrix (degrees) -> A (3, 3)."""
    a, b, g = (math.radians(v) for v in (rot, tilt, psi))
    ca, sa, cb, sb, cg, sg = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(g), math.sin(g)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return np.array([[cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
                     [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
                     [sc, ss, cb]])


def relion_angles(A: np.ndarray) -> tuple[float, float, float]:
    """RELION Euler_matrix2angles -> (rot, tilt, psi) in degrees."""
    A = normalize_rm(np.asarray(A, dtype=np.float64))
    abs_sb = math.hypot(A[0, 2], A[1, 2])
    if abs_sb > 16 * np.finfo(float).eps:
        gamma = math.atan2(A[1, 2], -A[0, 2])
        alpha = math.atan2(A[2, 1], A[2, 0])
        if abs(math.sin(gamma)) < np.finfo(float).eps:
            sign_sb = math.copysign(1.0, -A[0, 2] / math.cos(gamma))
        else:
            sign_sb = math.copysign(1.0, A[1, 2]) if math.sin(gamma) > 0 else -math.copysign(1.0, A[1, 2])
        beta = math.atan2(sign_sb * abs_sb, A[2, 2])
    elif A[2, 2] > 0:
        alpha, beta, gamma = 0.0, 0.0, math.atan2(-A[1, 0], A[0, 0])
    else:
        alpha, beta, gamma = 0.0, math.pi, math.atan2(A[1, 0], -A[0, 0])
    return math.degrees(alpha), math.degrees(beta), math.degrees(gamma)


def frealign_to_smap(psi, theta, phi) -> np.ndarray:
    """particles.par angles (Frealign PSI THETA PHI of q_opt*diag(1,-1,-1)) -> q_opt."""
    def rz(x):
        c, s = math.cos(x), math.sin(x)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def ry(x):
        c, s = math.cos(x), math.sin(x)
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    Q = D_FLIP @ rz(math.radians(psi)) @ ry(math.radians(theta)) @ rz(math.radians(phi))
    return Q @ D_FLIP


# ---------------------------------------------------------------- readers

def _par_value(params: dict, key, default=None):
    v = params.get(key, default)
    return v[0] if isinstance(v, list) else v


def optics_from_par(par_file) -> dict:
    params, _ = sio.read_params_file(par_file)
    return dict(pixel=float(_par_value(params, "aPerPix")),
                voltage_kV=float(_par_value(params, "V_acc")) / 1000,
                cs_mm=float(_par_value(params, "Cs")) * 1000)


def rows_from_smap(out_dir, micrograph, offset=(0, 0), images=False) -> list[dict]:
    """One dict per particles.csv row of a SMAP/port output folder."""
    out_dir = Path(out_dir)
    rows = []
    with open(out_dir / "particles.csv") as fh:
        for r in csv.DictReader(fh):
            q = np.array([float(r[f"q_opt_{i}{j}"]) for i in range(3) for j in range(3)]).reshape(3, 3)
            rows.append(dict(
                micrograph=micrograph, x=float(r["x_orig0_est"]) + offset[0],
                y=float(r["y_orig0_est"]) + offset[1], R=q, score=float(r["score_final"]),
                score_refined=float(r["score_refined"]), df1_A=float(r["df1_nm"]) * 10,
                df2_A=float(r["df2_nm"]) * 10, angast_deg=-float(r["ast_rad"]) * 180 / math.pi,
                image=f"{int(r['num_id']):06d}@{(out_dir / 'particles.mrc').as_posix()}" if images else None))
    return rows


def rows_from_rnf213(csv_file, micrograph=None) -> list[dict]:
    """RNF213 all_candidates_registration.csv (rnf213_workflow_v3 'register' step)."""
    rows = []
    with open(csv_file) as fh:
        for r in csv.DictReader(fh):
            R = frealign_to_smap(float(r["frealign_psi"]), float(r["frealign_theta"]),
                                 float(r["frealign_phi"]))
            rows.append(dict(
                micrograph=micrograph or r.get("micrograph") or r["tile"], x=float(r["x"]), y=float(r["y"]),
                R=R, score=float(r["score"]), score_refined=None, df1_A=float(r["defocus_u_A"]),
                df2_A=float(r["defocus_v_A"]), angast_deg=float(r["astigmatism_angle_deg"]), image=None))
    return rows


# ---------------------------------------------------------------- writer

def write_star(rows, path, pixel, voltage_kV, cs_mm, amp_contrast=0.07, coord_origin=0,
               image_size=None) -> int:
    images = any(r.get("image") for r in rows)
    labels = ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnAngleRot", "rlnAngleTilt",
              "rlnAnglePsi", "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnOpticsGroup",
              "rlnAutopickFigureOfMerit", "rlnClassNumber"]
    if images:
        labels += ["rlnImageName", "rlnOriginXAngst", "rlnOriginYAngst"]
    with open(path, "w") as fh:
        fh.write("# SMAP (Python port) detections; angles = RELION ZYZ of SMAP q_opt; "
                 f"coordinates {'zero' if coord_origin == 0 else 'one'}-based pixels; "
                 "figure of merit = final SMAP SNR\n\n")
        fh.write("data_optics\n\nloop_\n_rlnOpticsGroup #1\n_rlnOpticsGroupName #2\n"
                 "_rlnMicrographOriginalPixelSize #3\n_rlnVoltage #4\n_rlnSphericalAberration #5\n"
                 "_rlnAmplitudeContrast #6\n")
        line = f"1 opticsGroup1 {pixel:.6f} {voltage_kV:.2f} {cs_mm:.3f} {amp_contrast:.3f}"
        if images:
            fh.write("_rlnImagePixelSize #7\n_rlnImageSize #8\n_rlnImageDimensionality #9\n")
            line += f" {pixel:.6f} {int(image_size)} 2"
        fh.write(line + "\n\n\ndata_particles\n\nloop_\n")
        for i, lab in enumerate(labels, 1):
            fh.write(f"_{lab} #{i}\n")
        for r in rows:
            rot, tilt, psi = relion_angles(r["R"])
            vals = [r["micrograph"], f"{r['x'] + coord_origin:.4f}", f"{r['y'] + coord_origin:.4f}",
                    f"{rot:.6f}", f"{tilt:.6f}", f"{psi:.6f}", f"{r['df1_A']:.2f}", f"{r['df2_A']:.2f}",
                    f"{r['angast_deg']:.4f}", "1", f"{r['score']:.4f}", "1"]
            if images:
                vals += [r["image"], "0.0", "0.0"]
            fh.write(" ".join(vals) + "\n")
    return len(rows)


def read_star_particles(path) -> list[dict]:
    """Minimal reader for the data_particles loop (used by tests and round trips)."""
    labels, rows, block = [], [], None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("data_"):
            block, labels = s, []
            continue
        if s == "loop_":
            continue
        if block == "data_particles":
            if s.startswith("_"):
                labels.append(s.split()[0][1:])
            else:
                rows.append(dict(zip(labels, s.split())))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    a1 = sub.add_parser("smap", help="SMAP/port output folders (particles.csv)")
    a1.add_argument("out_dirs", nargs="+")
    a1.add_argument("--micrograph", required=True, help="rlnMicrographName to write")
    a1.add_argument("--offset", nargs=2, type=float, default=(0.0, 0.0), metavar=("X0", "Y0"),
                    help="zero-based tile origin in the micrograph (same for all folders)")
    a1.add_argument("--par", default=None, help="search.par for optics (default OUT_DIR/../search.par)")
    a1.add_argument("--images", action="store_true", help="also write rlnImageName j@particles.mrc")
    a2 = sub.add_parser("rnf213", help="RNF213 all_candidates_registration.csv")
    a2.add_argument("csv")
    a2.add_argument("--par", required=True, help="one tile's search.par (optics)")
    a2.add_argument("--micrograph", required=True, help="rlnMicrographName, e.g. the *_fractions.mrc name")
    for a in (a1, a2):
        a.add_argument("-o", "--out", default="particles.star")
        a.add_argument("--min-score", type=float, default=None, help="keep score_final >= this")
        a.add_argument("--amp-contrast", type=float, default=0.07)
        a.add_argument("--coord-origin", type=int, choices=(0, 1), default=0)
    a = ap.parse_args(argv)
    if a.source == "smap":
        rows = []
        for d in a.out_dirs:
            rows += rows_from_smap(d, a.micrograph, tuple(a.offset), a.images)
        par = Path(a.par) if a.par else Path(a.out_dirs[0]).parent / "search.par"
    else:
        rows = rows_from_rnf213(a.csv, a.micrograph)
        par = Path(a.par)
    if a.min_score is not None:
        rows = [r for r in rows if r["score"] >= a.min_score]
    opt = optics_from_par(par)
    size = None
    if a.source == "smap" and a.images and rows:
        size = int(sio.mr(Path(a.out_dirs[0]) / "particles.mrc").shape[0])
    n = write_star(rows, a.out, opt["pixel"], opt["voltage_kV"], opt["cs_mm"], a.amp_contrast,
                   a.coord_origin, size)
    print(f"wrote {n} particles to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
