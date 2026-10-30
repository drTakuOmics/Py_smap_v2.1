"""Atomic model -> scattering potential (smappoi_calculate_SP in smappoi_search_global.m).

Parsers mirror @smap/read_pdb_file.m and @smap/read_cif_file.m (the latter used grep/awk;
the same whitespace-split columns are used here). The per-atom potential loop is
evaluated only inside each atom's 5 A sphere, which is exactly the set of voxels the
MATLAB code updates (``inds=find(D<=atomRad)``), so results are identical while the
cost drops from O(nAtoms * N^3) to O(nAtoms * r^3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._sf_table import SF_TABLE
from .ctf import def_consts
from .io import mw
from .utils import (apply_phase_shifts_3d_real, crop_or_pad, matlab_mode,
                    particle_diameter, resize_for_fft, rrj)

W_POT = 4.871  # "Rullgard's calculation" (mean inner potential offset)


@dataclass
class Atoms:
    xyz: np.ndarray        # (3, n) Angstrom
    atom_list: list        # labels whose first character selects the scattering factor
    b_factor: np.ndarray   # (n,)
    chain_ids: list


def read_pdb_file(fn: str | Path) -> Atoms:
    """@smap/read_pdb_file.m. QUIRK: lines shorter than 78 chars are skipped."""
    xs, ys, zs, names, bf, chains = [], [], [], [], [], []
    text = Path(fn).read_text(errors="replace")
    for raw in text.split("\n"):
        line = raw
        if len(line) < 78:
            line = " " * 78
        head = line[0:4]
        if "ATOM" in head or "TER" in head or "HETA" in head:
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                float(line[54:60])  # occupancy must parse, as in MATLAB
                b = float(line[60:66])
            except ValueError:
                continue
            xs.append(x); ys.append(y); zs.append(z)
            names.append(line[11:16].strip())
            bf.append(b)
            chains.append(line[20:22].strip())
    xyz = np.array([xs, ys, zs], dtype=np.float64)
    return _truncate_nan(Atoms(xyz, names, np.array(bf), chains))


def read_cif_file(fn: str | Path, chains: list[str] | None = None) -> Atoms:
    """@smap/read_cif_file.m: lines starting with 'ATOM'; awk columns
    $2 id, $3 type_symbol, $19 auth_asym_id, $11-$13 xyz, $15 B."""
    xs, ys, zs, el, bf, ch = [], [], [], [], [], []
    with open(fn, "r", errors="replace") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            f = line.split()
            if len(f) < 15:
                continue
            # awk $19 (auth_asym_id); empty when the CIF has fewer columns (e.g. the
            # 18-column 6ek0_LSU.cif). MATLAB falls back to $17 only if awk errors.
            chain = f[18] if len(f) > 18 else (f[16] if len(f) > 16 else "")
            if chains and chain not in chains:
                continue
            el.append(f[2]); ch.append(chain)
            xs.append(float(f[10])); ys.append(float(f[11])); zs.append(float(f[12]))
            bf.append(float(f[14]))
    xyz = np.array([xs, ys, zs], dtype=np.float64)
    return _truncate_nan(Atoms(xyz, el, np.array(bf), ch))


def _truncate_nan(at: Atoms) -> Atoms:
    bad = np.nonzero(np.isnan(at.xyz).any(axis=0))[0]
    if bad.size:
        k = bad[0]
        at = Atoms(at.xyz[:, :k], at.atom_list[:k], at.b_factor[:k], at.chain_ids[:k])
    return at


def parameterize_sf(elem: str):
    g = SF_TABLE.get(elem)
    if g is None:
        print(f"did not find atom type {elem}")
        g = [0.0] * 10
    return np.array(g[:5]), np.array(g[5:])


def electrostatic_potential(atoms: Atoms, a_per_pix: float, b_arb=0.0, device=None,
                            batch: int = 4096) -> np.ndarray:
    """Per-atom potential sum on the SMAP grid (no +wPot). Returns float64 (N,N,N)
    in MATLAB [row=y, col=x, page=z] layout."""
    import torch

    xyz = atoms.xyz - np.nanmean(atoms.xyz, axis=1, keepdims=True)
    xyz = xyz - np.nanmean(xyz, axis=1, keepdims=True)
    bfac = atoms.b_factor.astype(np.float64)
    if b_arb is not None and b_arb != -1:
        bfac = np.full(xyz.shape[1], float(b_arb))

    pad_A = 5.0
    mm = np.stack([xyz.min(axis=1), xyz.max(axis=1)], axis=1)
    edge = max(math.ceil((2 * np.max(np.abs(mm)) + pad_A) / a_per_pix), 211)
    cp = edge // 2 + 1
    # tempVec from rrj(zeros(edge^3))*(edge-1): spacing (edge-1)/(2(cp-1)) pixels
    # QUIRK: for even edge this is (edge-1)/edge, not 1.
    scale = (edge - 1) / (2.0 * (cp - 1))
    step = scale * a_per_pix  # Angstrom per voxel along each axis

    cc = def_consts()
    lead = ((16 * math.pi ** 2.5 * (cc["h"] / (2 * math.pi)) ** 2) / (cc["m_e"] * cc["q_e"])) * 1e20
    atom_rad = 5.0

    dev = torch.device(device) if device else torch.device("cpu")
    V = torch.zeros(edge ** 3, dtype=torch.float64, device=dev)
    h = int(math.ceil(atom_rad / step)) + 1
    off = torch.arange(-h, h + 1, device=dev)
    oy, ox, oz = torch.meshgrid(off, off, off, indexing="ij")
    oy, ox, oz = oy.reshape(-1), ox.reshape(-1), oz.reshape(-1)

    first = [str(a)[0] for a in atoms.atom_list]
    ab = {e: parameterize_sf(e) for e in set(first)}
    A = np.stack([ab[e][0] for e in first])            # (n,5)
    B = np.stack([ab[e][1] for e in first])            # (n,5)
    B = B + bfac[:, None] + 16 * a_per_pix ** 2

    n = xyz.shape[1]
    for s in range(0, n, batch):
        e = min(n, s + batch)
        p = torch.as_tensor(xyz[:, s:e], device=dev)                  # (3,b)
        # nearest grid index (0-based) for each coordinate
        ci = torch.round(p / step).long() + (cp - 1)                  # (3,b)
        iy = ci[1][:, None] + oy[None]                                 # rows <- y
        ix = ci[0][:, None] + ox[None]                                 # cols <- x
        iz = ci[2][:, None] + oz[None]
        gy = (iy - (cp - 1)).double() * step
        gx = (ix - (cp - 1)).double() * step
        gz = (iz - (cp - 1)).double() * step
        D2 = (gx - p[0][:, None]) ** 2 + (gy - p[1][:, None]) ** 2 + (gz - p[2][:, None]) ** 2
        D = torch.sqrt(D2)
        ok = (D <= atom_rad) & (iy >= 0) & (iy < edge) & (ix >= 0) & (ix < edge) & (iz >= 0) & (iz < edge)
        a_t = torch.as_tensor(A[s:e], device=dev)
        b_t = torch.as_tensor(B[s:e], device=dev)
        terms = (a_t / b_t ** 1.5)[:, None, :] * torch.exp(-(4 * math.pi ** 2) * D2[:, :, None] / b_t[:, None, :])
        val = lead * terms.sum(-1)
        lin = (iy * edge + ix) * edge + iz
        V.index_add_(0, lin[ok], val[ok])
    return V.reshape(edge, edge, edge).cpu().numpy()


def calculate_sp(params: dict, out_dir: str | Path | None = None, device=None, log=print):
    """smappoi_calculate_SP: structure file -> scattering potential volume (float32).

    Writes <outputDir>/<name>_EP.mrc and <name>_SP.mrc like v2.1; returns (SP, path).
    """
    fn = Path(params["structureFile"])
    ext = fn.suffix
    if ext in (".pdb", ".PDB", ".pdb1"):
        atoms = read_pdb_file(fn)
    elif ext == ".cif":
        ch = params.get("chains")
        atoms = read_cif_file(fn, ch.split(" ") if ch else None)
    else:
        raise ValueError(f"File format for {fn} not recognized")
    a_per_pix = float(params["aPerPix"])
    log(f"calculating potential for {atoms.xyz.shape[1]} atoms...")
    b_arb = params.get("bArb", 0)
    EPV = electrostatic_potential(atoms, a_per_pix, b_arb=b_arb, device=device)
    EPV = EPV + W_POT

    out_dir = Path(out_dir or params["outputDir"])
    name = (params.get("modelName") or "").strip() or fn.stem
    fn_EP = out_dir / f"{name}_EP.mrc"
    fn_SP = out_dir / f"{name}_SP.mrc"
    mw(EPV, fn_EP, a_per_pix)
    EPV = EPV.astype(np.float32)  # MATLAB re-reads the written (single) files

    SP = finalize_sp(EPV, params, log=log)
    mw(SP, fn_SP, a_per_pix)
    return SP, fn_SP


def finalize_sp(EPV: np.ndarray, params: dict, log=print) -> np.ndarray:
    """Post-processing of smappoi_calculate_SP: pad to the search box, convert to
    phase shift per voxel, and centre on the centre of mass (sub-pixel)."""
    a_per_pix = float(params["aPerPix"])
    # QUIRK (v2.1 behaviour): smap.particleDiameter calls the class method `bindata`
    # without the `smap.` prefix, which fails (confirmed in Octave), so v2.1 always
    # takes the catch branch: edge = 2.5*size -> in practice edge_max (512).
    # Set params['legacy_particle_diameter']=0 to use the intended estimate instead.
    legacy = params.get("legacy_particle_diameter", 1)
    try:
        if legacy:
            raise RuntimeError("legacy v2.1 path")
        pd = particle_diameter(EPV)
        e2 = int(round(2.5 * pd))
        edge = resize_for_fft(np.zeros((e2, e2)), "crop").shape[0]
    except Exception:
        log("problem estimating particle diameter...")
        pd = EPV.shape[0]
        edge = 2.5 * pd
    if edge > params["edge_max"]:
        edge = params["edge_max"]
        log(f"using edge_max for padding ({params['edge_max']}^3)...")
    edge = int(max(edge, max(EPV.shape)))
    log(f"padding to {edge}^3 volume...")
    # single precision from here on, as in MATLAB (SPV_out=single(...))
    SPV = crop_or_pad(EPV, [edge] * 3, W_POT).astype(np.float32)

    cc = def_consts()
    dx = a_per_pix / 1e10
    V = params["V_acc"]
    lam = cc["h"] / math.sqrt(cc["q_e"] * V * cc["m_e"] * (cc["q_e"] / cc["m_e"] * V / cc["c_v"] ** 2 + 2))
    k = 2 * math.pi / lam
    SPV *= np.float32(cc["IC"] * dx / (2 * k))

    bg = matlab_mode(SPV)
    x = SPV - bg
    del SPV
    n = x.shape[0]
    cpv = n // 2 + 1
    dummy = np.arange(-(cpv - 1), cpv - (2 - n % 2) + 1, dtype=np.float64)
    M = x.sum(dtype=np.float64)
    com_x = np.einsum("j,ijk->", dummy, x, dtype=np.float64) / M  # X varies along columns
    com_y = np.einsum("i,ijk->", dummy, x, dtype=np.float64) / M  # Y varies along rows
    com_z = np.einsum("k,ijk->", dummy, x, dtype=np.float64) / M
    x = apply_phase_shifts_3d_real(x, -np.array([com_y, com_x, com_z]))
    return np.real(x + bg).astype(np.float32)
