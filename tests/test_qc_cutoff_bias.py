"""Tests for smap.qc, smap.cutoff and smap.bias on synthetic data."""

import csv
import json

import numpy as np
from scipy import ndimage

from smap import bias, cutoff, qc
from smap import io as sio


def _noise_mip(shape, n_orient=40, sigma=1.5, seed=0):
    """Pixelwise max over correlated Gaussian fields, flat-fielded like search_mip_scaled."""
    rng = np.random.default_rng(seed)
    stack = np.stack([ndimage.gaussian_filter(rng.standard_normal(shape), sigma) for _ in range(n_orient)])
    stack /= stack.std()
    mip, mean, sd = stack.max(0), stack.mean(0), stack.std(0)
    return (mip - mean) / sd, mean, sd


def _plant(z, rc, height, radius=3.0):
    rr, cc = np.ogrid[:z.shape[0], :z.shape[1]]
    for r, c in rc:
        z += height * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * radius ** 2))
    return z


def _write_run(tmp, z, mean, sd, particles):
    tmp.mkdir(parents=True, exist_ok=True)
    margin = 8
    valid = np.zeros(z.shape, bool)
    valid[margin:-margin, margin:-margin] = True
    z = np.where(valid, z, 0.0)
    sio.mw(z, tmp / "search_mip_scaled.mrc", 1.5)
    sio.mw(mean[:, :, None].repeat(2, 2), tmp / "search_mean.mrc", 1.5)
    sio.mw(sd[:, :, None].repeat(2, 2), tmp / "search_SD.mrc", 1.5)
    sio.mw(np.ones(z.shape, np.float32), tmp / "search_mip_dfInds.mrc", 1.5)
    with open(tmp / "particles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["num_id", "score_final", "row_search", "col_search"])
        for k, (r, c) in enumerate(particles, 1):
            w.writerow([k, 9.0, r + 1, c + 1])
    return tmp


def _stack_maps(shape, targets, artifacts, n_orient=60, seed=0):
    """Orientation stack with true targets (one sharp best orientation over a weak low-resolution
    response) and artefacts (a strong response at every orientation plus a few stronger ones,
    which keeps their z high after flat-fielding); returns z, mean, sd."""
    rng = np.random.default_rng(seed)
    stack = np.stack([ndimage.gaussian_filter(rng.standard_normal(shape), 1.5) for _ in range(n_orient)])
    stack /= stack.std()
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    k = np.arange(n_orient)
    for (r, c), amp in targets:
        g = amp * (0.12 + 0.88 * np.exp(-(k - rng.integers(n_orient)) ** 2 / 2.0))
        stack += g[:, None, None] * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 8.0)[None]
    for (r, c), amp in artifacts:
        g = np.full(n_orient, 0.45 * amp)
        g[rng.choice(n_orient, 2, replace=False)] = amp
        stack += g[:, None, None] * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 8.0)[None]
    mip, mean, sd = stack.max(0), stack.mean(0), stack.std(0)
    return (mip - mean) / sd, mean, sd


def test_qc_passes_true_targets_and_flags_broad_artefact(tmp_path):
    rng = np.random.default_rng(5)
    pos = [tuple(p) for p in rng.integers(20, 180, size=(9, 2))]
    targets = [(p, a) for p, a in zip(pos[:8], np.linspace(4.5, 9.0, 8))]
    z, mean, sd = _stack_maps((200, 200), targets, [(pos[8], 9.0)], seed=6)
    assert z[pos[8]] > min(z[p] for p in pos[:8])          # the artefact scores like a detection
    run = _write_run(tmp_path / "run", z, mean, sd, pos)
    res = qc.run_qc(run, k=5.0, particle_radius_px=6)
    flags = [r["flags"] for r in res["rows"]]
    assert all(f == "" for f in flags[:8]), flags
    assert flags[8] != "" and ("cc_sd_high" in flags[8] or "cc_mean_high" in flags[8])
    assert res["summary"]["signal_model"]["used"]
    qc.write_qc(res, run)
    assert json.loads((run / "particles_qc.json").read_text())["n_pass"] == 8


def test_qc_small_population_falls_back(tmp_path):
    z, mean, sd = _stack_maps((120, 120), [((60, 60), 8.0)], [], seed=7)
    run = _write_run(tmp_path / "run", z, mean, sd, [(60, 60)])
    res = qc.run_qc(run)
    assert not res["summary"]["signal_model"]["used"] and res["rows"][0]["flags"] == ""


def test_qc_image_level_thickness(tmp_path):
    z, mean, sd = _noise_mip((120, 120), seed=2)
    run = _write_run(tmp_path / "run", _plant(z, [(60, 60)], 6.0), mean, sd, [(60, 60)])
    res = qc.run_qc(run, image_info={"thickness_nm": 320.0}, thickness_window=(50, 250))
    assert res["rows"][0]["flags"] == "thickness"


def test_anisotropy_separates_ridge_from_round_peak():
    z = np.zeros((80, 80))
    z = _plant(z, [(40, 20)], 6.0)
    z[38:43, 45:75] += 6.0                            # ridge (edge-like)
    zs = ndimage.gaussian_filter(z, 2)
    assert qc.anisotropy(zs, 40, 20) < 2.0
    assert qc.anisotropy(zs, 40, 60) > 4.0


def test_tail_cutoff_is_calibrated_on_noise():
    """On particle-free images, the tail-model cut-off gives about n_false false clusters,
    and is closer to that rate than a GEV fitted to the whole distribution."""
    tail, bulk = [], []
    for s in range(12):
        z, _, _ = _noise_mip((192, 192), seed=10 + s)
        valid = np.ones(z.shape, bool)
        rep = cutoff.tail_cutoff(z, valid, n_false=1.0, radius=6)
        assert rep["gpd_pixel"] >= rep["gpd_cluster"] - 1e-3
        tail.append(cutoff.n_clusters(z, rep["gpd_cluster"], valid))
        bulk.append(cutoff.n_clusters(z, rep["gev_bulk"], valid))
    assert 0.4 <= np.mean(tail) <= 2.5
    assert abs(np.log(max(np.mean(tail), 0.05))) <= abs(np.log(max(np.mean(bulk), 0.05)))


def test_tail_cutoff_stable_when_particles_present():
    z, _, _ = _noise_mip((192, 192), seed=30)
    valid = np.ones(z.shape, bool)
    c0 = cutoff.tail_cutoff(z, valid, radius=6)["gpd_cluster"]
    rng = np.random.default_rng(3)
    peaks = [tuple(p) for p in rng.integers(15, 177, size=(25, 2))]
    c1 = cutoff.tail_cutoff(_plant(z.copy(), peaks, 5.0), valid, radius=6)["gpd_cluster"]
    assert abs(c1 - c0) < 0.4


def test_decoy_cutoff_matches_requested_rate():
    maps = []
    for s in range(6):
        z, _, _ = _noise_mip((128, 128), seed=50 + s)
        maps.append((z, np.ones(z.shape, bool)))
    rep = cutoff.decoy_cutoff(maps, n_false=1.0)
    mean_at = np.mean([cutoff.n_clusters(z, rep["decoy"], v) for z, v in maps])
    assert mean_at <= 1.0


def _blob(shape, center, sigma):
    zz, yy, xx = np.indices(shape)
    return np.exp(-((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) / (2 * sigma ** 2))


def test_omega_recovers_known_bias_with_scale_mismatch():
    shape = (48, 48, 48)
    A = _blob(shape, (24, 24, 18), 5) + _blob(shape, (24, 16, 24), 4)
    B = _blob(shape, (24, 32, 30), 3.5)
    t_full, t_omit = A + B, A
    rng = np.random.default_rng(0)
    beta = 0.6                                        # 60 % of the omitted density recovered
    v_full = 1.0 * (A + B) + 0.01 * rng.standard_normal(shape)
    v_omit = 2.5 * (A + beta * B) + 0.3 + 0.01 * rng.standard_normal(shape)   # other scale + offset
    rep = bias.template_bias(v_full, v_omit, t_full, t_omit)
    assert abs(rep["omega"] - (1 - beta)) < 0.05
    assert abs(rep["omega_raw"] - (1 - beta)) > 0.3   # the raw value is misled by the scale change
    assert rep["recovery_correlation"] > 0.9
    assert len(rep["regions"]) == 1


def test_omega_extremes():
    shape = (40, 40, 40)
    A, B = _blob(shape, (20, 20, 14), 5), _blob(shape, (20, 20, 28), 4)
    full = A + B
    none = bias.template_bias(full, full, full, A)
    assert abs(none["omega"]) < 1e-6 and abs(none["omega_norm"]) < 1e-6
    total = bias.template_bias(full, A, full, A)           # omitted density absent: complete bias
    assert total["omega"] < 1.0                             # neighbouring kept density sits inside M
    assert abs(total["omega_norm"] - 1.0) < 0.02


def test_decoy_cutoff_ignores_merged_background():
    """A MIP z map has a raised background: at low thresholds every pixel joins one region.
    The decoy cut-off must come from the top, not from that merged region."""
    from scipy.ndimage import gaussian_filter
    from smap.cutoff import decoy_cutoff
    rng = np.random.default_rng(5)
    z = gaussian_filter(rng.standard_normal((300, 300)), 2.0)
    z = 4.5 + 0.5 * z / z.std()
    z[150, 150] = 7.0                                   # one isolated decoy peak
    res = decoy_cutoff([(z, np.ones_like(z, bool))], n_false=1.0)
    assert res["decoy"] > 5.5


def test_particle_radius_from_model(tmp_path):
    import json
    from pathlib import Path
    from smap.cutoff import particle_radius_px
    lsu = Path(__file__).resolve().parents[1] / "examples" / "ribosome_in_situ" / "model" / "6ek0_LSU.cif"
    (tmp_path / "search_status.json").write_text(json.dumps(
        {"sampling_A": 1.5, "provenance": {"input_sha256": {"structureFile": {"path": str(lsu)}}}}))
    r, how = particle_radius_px(tmp_path)
    assert 70 < r < 130 and "6ek0_LSU" in how
    r2, how2 = particle_radius_px(tmp_path / "missing")
    assert r2 == 10.0 and "fallback" in how2


def test_analysis_mask_drops_padding_and_low_sd(tmp_path):
    """FFT padding stays inside combine's 'valid' frame; there the orientation SD is ~0 and
    the flat-fielded z explodes. The analysis mask must exclude it (as on the MEF decoy map)."""
    from scipy.ndimage import gaussian_filter
    from smap import io as sio
    from smap.cutoff import decoy_cutoff, load_z
    rng = np.random.default_rng(11)
    n, pad = 220, 30
    img = np.full((n, n), -1.2e-17, np.float32)               # padding as written by prepare
    img[pad:n - pad, pad:n - pad] = rng.standard_normal((n - 2 * pad, n - 2 * pad))
    sd = np.full((n, n), 0.02, np.float32)
    sd[pad:n - pad, pad:n - pad] = 1.0
    zz = gaussian_filter(rng.standard_normal((n, n)), 1.5)
    z = (4.9 + 0.45 * zz / zz.std()).astype(np.float32)
    z[:pad, :] = 5.0 + 3.0 * np.abs(rng.standard_normal((pad, n)))   # speckle: SD ~ 0 in the padding
    z[:3, :] = 0                                            # combine's thin frame margin
    for name, arr in (("searchImage.mrc", img), ("search_SD.mrc", sd), ("search_mip_scaled.mrc", z)):
        sio.mw(arr, tmp_path / name, 1.5)
    raw = sio.mr(tmp_path / "search_mip_scaled.mrc").astype(np.float64)
    c_raw = decoy_cutoff([(raw, raw != 0)])["decoy"]
    assert np.isnan(c_raw) or c_raw > 11                    # unmasked: the padding dominates
    zm, keep, info = load_z(tmp_path, edge_px=5, with_info=True)
    assert info["image_box"] == [pad, n - pad - 1, pad, n - pad - 1]
    assert info["kept_fraction"] < 0.8 and not keep[:pad + 5, :].any() and not keep[:, :pad + 5].any()
    assert decoy_cutoff([(zm, keep)])["decoy"] < 7.5


def test_tail_cutoff_crowded_image_keeps_background():
    """Crowded field (like the MEF LSU image): discs of particle radius around every detection
    cover the image. The fit must shrink the discs, keep >= 5% of the pixels and return a
    finite cut-off instead of failing on an empty background."""
    from scipy.ndimage import gaussian_filter
    from smap.cutoff import tail_cutoff
    rng = np.random.default_rng(21)
    n = 400
    zz = gaussian_filter(rng.standard_normal((n, n)), 1.5)
    z = 4.9 + 0.3 * zz / zz.std()
    yy, xx = np.mgrid[:n, :n]
    centres = [(int(r), int(c)) for r, c in rng.integers(20, n - 20, (60, 2))]
    for r, c in centres:
        z += 10.0 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 15.0 ** 2))
    valid = np.ones_like(z, bool)
    res = tail_cutoff(z, valid, radius=60.0, centres=centres)
    assert res["mask_radius_used_px"] < 60.0
    assert np.isfinite(res["gpd_cluster"]) and 5.0 < res["gpd_cluster"] < 10.0
