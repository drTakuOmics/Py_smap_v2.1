"""Template bias of omitted-region reconstructions (new; not part of v2.1).

Detections found with a template can reproduce the template in their average (template
bias). The test is to search with a template from which a region was removed (omit or
anchor template), reconstruct, and check whether the removed density comes back. Following
the definition of Lucas et al., eLife 2023 (also used by Zhang et al., eLife 2026), written
independently here:

    Omega = sum_M (V_full - V_omit) / sum_M V_full

over the omitted region M, where V_full and V_omit are reconstructions from independent
searches with the full and the omit template. Omega = 0: the omitted density is fully
recovered (no bias); Omega = 1: it vanishes when absent from the template (full bias).

M is where the full template exceeds the omit template: T_full - T_omit > f * mean(top-k
voxels of the difference) (f = 0.2, k = 100 by default).

Additions:

* **Scale matching.** V_full and V_omit come from different particle sets (numbers, defocus,
  filtering), so their overall scale and offset differ. V_omit is first fitted to V_full by
  least squares (V_omit ~ a V_full + b) over a reference region: inside the omit template's
  envelope and away (``--gap`` voxels) from M. Omega is reported raw and scale-matched.
* **Per-region values.** M is split into connected regions (e.g. one per omitted domain),
  each with its own Omega and voxel count.
* **Normalised Omega.** Inside M part of the density belongs to the kept region (the masks
  of neighbouring parts overlap), so even complete bias gives Omega < 1. With
  e_M = sum_M T_omit / sum_M T_full (that kept fraction, from the templates),
  ``omega_norm = omega / (1 - e_M)`` runs from 0 (no bias) to 1 (complete bias).
* **Recovery correlation.** Pearson correlation, inside M, between V_omit and the omitted
  density T_full - T_omit: high when the recovered density has the expected shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import io as sio


def omit_mask(t_full: np.ndarray, t_omit: np.ndarray, frac: float = 0.2, top_k: int = 100) -> np.ndarray:
    """Mask M of the omitted region: T_full - T_omit > frac x mean of the top_k difference voxels."""
    d = np.asarray(t_full, np.float64) - np.asarray(t_omit, np.float64)
    flat = np.sort(d.ravel())[::-1]
    k = min(top_k, flat.size)
    thr = frac * float(flat[:k].mean())
    return d > thr


def envelope(t: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Voxels above frac of the template maximum."""
    t = np.asarray(t, np.float64)
    return t > frac * float(t.max())


def match_scale(v_ref: np.ndarray, v: np.ndarray, region: np.ndarray) -> tuple[float, float]:
    """(a, b) minimising |v - (a v_ref + b)|^2 over region; v is then mapped to (v - b) / a."""
    x, y = v_ref[region].ravel(), v[region].ravel()
    A = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def omega(v_full: np.ndarray, v_omit: np.ndarray, mask: np.ndarray) -> float:
    """Omega = sum_M (V_full - V_omit) / sum_M V_full (NaN when the denominator is 0)."""
    den = float(v_full[mask].sum())
    return float((v_full[mask] - v_omit[mask]).sum() / den) if den != 0 else float("nan")


def kept_fraction(t_full, t_omit, mask) -> float:
    """Share of the full template's density inside mask that belongs to the kept region."""
    return float(np.asarray(t_omit, np.float64)[mask].sum() / np.asarray(t_full, np.float64)[mask].sum())


def template_bias(v_full, v_omit, t_full, t_omit, frac: float = 0.2, top_k: int = 100, gap: int = 3,
                  min_region_voxels: int = 10) -> dict:
    """Omega (raw, scale-matched and normalised) of an omit reconstruction, overall and per
    connected region of M, with the recovery correlation. The four volumes share one grid."""
    v_full, v_omit = np.asarray(v_full, np.float64), np.asarray(v_omit, np.float64)
    if not (v_full.shape == v_omit.shape == np.shape(t_full) == np.shape(t_omit)):
        raise ValueError("maps and templates must share one grid (resample them first)")
    M = omit_mask(t_full, t_omit, frac, top_k)
    ref = envelope(t_omit) & ~ndimage.binary_dilation(M, iterations=gap)
    a, b = match_scale(v_full, v_omit, ref)
    v_omit_s = (v_omit - b) / a if a != 0 else v_omit
    d = np.asarray(t_full, np.float64) - np.asarray(t_omit, np.float64)
    corr = float(np.corrcoef(v_omit_s[M], d[M])[0, 1]) if M.sum() > 2 else float("nan")
    lab, n = ndimage.label(M)
    regions = []
    for i in range(1, n + 1):
        Mi = lab == i
        if Mi.sum() < min_region_voxels:
            continue
        com = [float(c) for c in ndimage.center_of_mass(Mi)]
        om = omega(v_full, v_omit_s, Mi)
        regions.append(dict(region=i, voxels=int(Mi.sum()), center_voxel=com,
                            omega_raw=omega(v_full, v_omit, Mi), omega=om,
                            omega_norm=om / (1 - kept_fraction(t_full, t_omit, Mi))))
    om = omega(v_full, v_omit_s, M)
    e_m = kept_fraction(t_full, t_omit, M)
    return dict(omega_raw=omega(v_full, v_omit, M), omega=om, omega_norm=om / (1 - e_m), kept_fraction=e_m,
                scale_a=a, offset_b=b, mask_voxels=int(M.sum()), reference_voxels=int(ref.sum()),
                recovery_correlation=corr, regions=regions,
                params=dict(frac=frac, top_k=top_k, gap=gap),
                note="omega: 0 = omitted density fully recovered, 1 = absent (full template bias)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Template bias (Omega) of an omit-template reconstruction")
    ap.add_argument("--full-map", required=True, help="reconstruction from the full-template search (.mrc)")
    ap.add_argument("--omit-map", required=True, help="reconstruction from the omit-template search (.mrc)")
    ap.add_argument("--full-template", required=True, help="full template / scattering potential (.mrc)")
    ap.add_argument("--omit-template", required=True, help="omit template on the same grid (.mrc)")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--mask-out", help="write the omitted-region mask (.mrc)")
    a = ap.parse_args(argv)
    maps = [sio.mr(p) for p in (a.full_map, a.omit_map, a.full_template, a.omit_template)]
    rep = template_bias(*maps, frac=a.frac, top_k=a.top_k, gap=a.gap)
    if a.mask_out:
        import mrcfile
        with mrcfile.open(a.full_template, permissive=True) as m:
            voxel = float(m.voxel_size.x) or 1.0
        sio.mw(omit_mask(maps[2], maps[3], a.frac, a.top_k).astype(np.float32), a.mask_out, voxel)
    txt = json.dumps(rep, indent=1)
    if a.json:
        Path(a.json).write_text(txt)
    print(f"Omega = {rep['omega']:.3f} (raw {rep['omega_raw']:.3f}, normalised {rep['omega_norm']:.3f}), "
          f"recovery r = {rep['recovery_correlation']:.2f}, {rep['mask_voxels']} voxels in the omitted region")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
