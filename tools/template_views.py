#!/usr/bin/env python
"""What the search correlates: the templates of one search at a few orientations, before and after
the CTF and the whitening filter, next to a central slice of the potential.

    python tools/template_views.py SEARCH.par --out template_views.png [--plane K]

Uses the search's own setup (smap.search_global.prepare: potential resampled to the search sampling,
CTF of the .par's defocus planes, whitening filter from the micrograph's spectrum) and its central-
slice projector, so the panels show the templates the search uses. Views: along a 5-, 3- and 2-fold
axis when the .par has a symmetryFile, otherwise along the model's x, y and z axes. Rows: projected
potential; with the CTF; with the CTF and the whitening filter (what is cross-correlated with the
whitened micrograph). --plane: defocus plane (default: the middle one). Nothing is written to the
search's folders.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root (smap)


def rotation_order(M: np.ndarray, max_order: int = 6) -> int:
    """Order of a rotation: the smallest n <= max_order with M^n = identity (0 if there is none)."""
    P = np.eye(3)
    for k in range(1, max_order + 1):
        P = P @ M
        if np.abs(P - np.eye(3)).max() < 1e-3:
            return k
    return 0


def symmetry_axes(S) -> dict:
    """{order: unit axis} with one axis per rotation order (identity excluded)."""
    out = {}
    for M in np.asarray(S, dtype=np.float64):
        n = rotation_order(M)
        if n < 2 or n in out:
            continue
        w, v = np.linalg.eig(M)
        a = np.real(v[:, int(np.argmin(np.abs(w - 1)))])
        out[n] = a / np.linalg.norm(a)
    return out


def view_rotation(axis) -> np.ndarray:
    """A rotation whose viewing direction (third row, smap.symmetry convention) is ``axis``."""
    e3 = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    t = np.array([1.0, 0, 0]) if abs(e3[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = t - (t @ e3) * e3
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(e3, e1), e3])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("par", help="search .par (with its micrograph and template potential)")
    ap.add_argument("--out", required=True, help="output .png")
    ap.add_argument("--plane", type=int, default=None, help="defocus plane (0-based; default the middle one)")
    ap.add_argument("--label", default=None)
    a = ap.parse_args(argv)
    import torch

    from smap import io as sio
    from smap.search_global import _templates, normalize_rm, prepare

    tmp = Path(tempfile.mkdtemp(prefix="template_views_"))
    # a coarse angular grid: the rotation list of the search is not needed here
    txt = re.sub(r"^(angle_inc|psi_inc)\s+.*$", r"\1 30", Path(a.par).read_text(), flags=re.M)
    (tmp / "views.par").write_text(txt)
    try:
        st = prepare(tmp / "views.par", write_files=False, output_dir=tmp / "out")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    p = st.params
    N, (N0, N1) = st.Npix, st.Npix_im
    k = st.n_dfs // 2 if a.plane is None else a.plane
    a_s = float(p["aPerPix"])                                       # search sampling after prepare

    if p.get("symmetryFile"):
        axes = symmetry_axes(sio.read_rotations_file(p["symmetryFile"]))
        views = [(f"{n}-fold axis", view_rotation(axes[n])) for n in sorted(axes, reverse=True)][:3]
    else:
        views = [(f"{n} axis", view_rotation(v)) for n, v in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1]))]
    R = normalize_rm(np.stack([v for _, v in views]))
    Vr = torch.as_tensor(np.ascontiguousarray(st.V_F.real))
    Vi = torch.as_tensor(np.ascontiguousarray(st.V_F.imag))
    ar = torch.arange(1, N + 1, dtype=torch.float64)
    rr, cc_ = torch.meshgrid(ar, ar, indexing="ij")
    cp, K = N // 2 + 1, st.oversample
    T = _templates(Vr, Vi, torch.as_tensor(R), (cc_ - cp) * K, (rr - cp) * K, st.V_F.shape[0] // 2 + 1).numpy()

    off0, off1 = (N0 // 2 + 1) - (N // 2 + 1), (N1 // 2 + 1) - (N // 2 + 1)    # as in run_job
    rows = [[], [], []]
    for t in T:
        img = np.zeros((N0, N1))
        img[off0:off0 + N, off1:off1 + N] = t
        F = np.fft.rfftn(img, axes=(1, 0))                                     # (h0, N1)
        ctf_only = np.fft.irfftn(F * st.CTF_half[:, :, k], s=(N1, N0), axes=(1, 0))
        searched = np.fft.irfftn(F * st.fPSD_half * st.CTF_half[:, :, k], s=(N1, N0), axes=(1, 0))
        for r, x in zip(rows, (img, ctf_only, searched)):
            r.append(x[off0:off0 + N, off1:off1 + N])
    V = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(st.V_F))))
    c = V.shape[2] // 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nv = len(views)
    fig, ax = plt.subplots(3, nv + 1, figsize=(3.2 * (nv + 1), 9.6))
    ext = (-N * a_s / 2 / 10, N * a_s / 2 / 10, N * a_s / 2 / 10, -N * a_s / 2 / 10)       # nm
    ax[0, 0].imshow(V[:, :, c], cmap="gray", extent=ext)
    ax[0, 0].set_title("potential, central slice", fontsize=9)
    f = np.arange(N1 // 2) / (N1 * a_s)                                         # 1/A along x
    df_um = float(np.mean(st.defocus[k][:2])) / 1000
    ax[1, 0].plot(f, st.CTF_half[0, :N1 // 2, k], lw=0.6, color="#2a78d6")
    ax[1, 0].set_title(f"CTF, defocus {df_um:.2f} µm", fontsize=9)
    wf = st.fPSD_half[0, :N1 // 2]
    ax[2, 0].plot(f, wf / max(float(np.max(wf)), 1e-12), lw=0.8, color="#eb6834")
    ax[2, 0].set_title("whitening filter (from the micrograph)", fontsize=9)
    for axx in ax[1:, 0]:
        axx.set_xlabel("spatial frequency (1/Å)", fontsize=8)
        axx.axhline(0, color="#999999", lw=0.5)
        axx.tick_params(labelsize=7)
    labels = ("projected potential", "× CTF", "× CTF × whitening (correlated with the image)")
    for j, (name, _) in enumerate(views, start=1):
        for i in range(3):
            x = rows[i][j - 1]
            if i == 0:
                ax[i, j].imshow(x, cmap="gray", extent=ext)
            else:
                lim = float(np.percentile(np.abs(x), 99.7)) or 1.0
                ax[i, j].imshow(x, cmap="gray", vmin=-lim, vmax=lim, extent=ext)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
        ax[0, j].set_title(name, fontsize=9)
    for i, lab in enumerate(labels):
        ax[i, 1].set_ylabel(lab, fontsize=8)
    ax[0, 0].set_xlabel("nm", fontsize=8)
    ax[0, 0].tick_params(labelsize=7)
    fig.suptitle(f"{a.label or Path(a.par).stem}: templates as searched ({a_s:g} Å/px, {N} px box)", fontsize=10)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"-> {out} ({nv} views; defocus plane {k + 1} of {st.n_dfs}, {df_um:.2f} µm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
