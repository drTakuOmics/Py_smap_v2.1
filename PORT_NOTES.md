# `smap/` — Python/PyTorch port of SMAP v2.1 `smappoi_search_global`

Source of truth: `smap_v2.1/src/smap_tools/smappoi_search_global.m`, `@smap/*.m` and
`quaternion.m` (J. P. Rickgauer, MIT License). Translated one-to-one from the MATLAB.
The earlier simplified prototype (`src/smap_tools_python/`, its tests and the compiled
MATLAB/Docker files) was removed in 0.2.0; the entry point is `python -m smap.search_global`
(`smap-search-global` after `pip install -e .`).

## Status

| MATLAB lines | Step | Python | Status |
|---|---|---|---|
| 51–354, 1992–2381 | params, rotations, defocus stack, whitening, PDB/CIF → scattering potential | `search_global.prepare`, `structure.py` | ported, Octave parity |
| 355–627 | CTF stack, Fourier volume, thresholds, main CC loop | `prepare`, `run_job` | ported, Octave parity |
| 628–863 | combine jobs, flat-field normalisation, `search_SH.txt`, `highVals.txt` | `combine` | ported |
| 864–968 | `clusterImByThr` per defocus plane, duplicate suppression, `opt_toDo` files | `refine.select_particles` | ported, Octave parity (clustering) |
| 983–1353 | per-particle grid refinement: defocus, angular bump set, defocus, full-image peak → `opt_done` | `refine.refine_particles` | ported, Octave parity (scoring) |
| 1418–1562, 1811–1990 | `optimize_xcorr` (exit-wave template, Nelder–Mead shift/angle) at original pixel size | `refine.optimize_particles` | ported, Octave parity (template, objective) |
| 1566–1704 | `list_final.txt`, `particles.{mrc,par,txt}`, `templates.mrc` | `refine.write_final` | ported; `particles.mat` replaced by `particles.csv` |

Output files keep v2.1 names and formats, and parse with the RNF213 review readers
(`matrix(...)`, `particles.par` with one header row, stacks indexed `j@particles.mrc`).
`run.log` contains the lines the workflow scripts look for ("starting optimization on",
"nothing to optimize", "Done combining search output").

New outputs: `search_status.json` (completeness, evaluated pose count, trial count, raw
pre-selection threshold, score definitions, calibration summary, decoy settings, MIP summary)
and per-job `scratch/search_XXXX_status.json`.

Full-image maps (on by default, `write_mip 0` disables; they do not change any v2.1 file):
`search_mip.mrc` (max raw CC over orientations at the best defocus plane),
`search_mip_scaled.mrc` (its pixelwise z-score; equals `search_vals.mrc` wherever v2.1 kept a
value, i.e. raw CC > `arbThr`), `search_mip_qInds.mrc`, `search_mip_dfInds.mrc` and
`search_pval.mrc` (−ln p, anisotropic-Gaussian p-value of Zhang et al., IUCrJ 2025, computed
over the kept region; `smap/pvalue.py`, same computation as Leopard-EM, |Δ| < 1e-3 on test
data). `particles.csv` gains `neg_log_p_search`. v2.1 keeps flat-fielded values only above
`arbThr`, so a complete MIP / p-value map was not available before.

Array convention: Python axis k = MATLAB dimension k+1 (images `[row, col]`, volumes
`[row, col, page]`); `io.smap_to_zyx` / `io.zyx_to_smap` convert to/from mrcfile-native
`[z, y, x]`. Rotation stacks are `(n, 3, 3)`, `R[i-1] == R(:,:,i)`. Coordinates written to
text files are 1-based like v2.1.

## Run

```bash
pip install -e .            # numpy scipy mrcfile torch (CUDA build for GPU)
python -m smap.search_global search.par                 # all jobs, then refinement
python -m smap.search_global search.par --batch-size 8  # batched orientations per GPU pass
python -m smap.search_global search.par --no-refine     # stop after combine
```

## Command-line flags

| Flag | Purpose |
|---|---|
| `--path-map OLD=NEW` | rewrite path prefixes in the `.par` (e.g. Docker `/work` → local project folder), repeatable |
| `--output-dir DIR` | write outputs to DIR instead of the `.par` outputDir (keeps an existing v2.1 `output/` intact) |
| `--log FILE` | also write all messages to FILE in UTF-8 (use for `run.log`; PowerShell `>` writes UTF-16) |
| `--batch-size B` | orientations per GPU pass; scheduling only |
| `--device` | `cpu`, `cuda`, `cuda:1`, … |
| `--no-refine` | stop after combine |
| `--max-rotations K` | debug only: the run is flagged incomplete, refinement is skipped, exit code 2, and "Done combining search output" is **not** printed (so RNF213 completion checks ignore it) unless `--allow-incomplete` |

### Running the port on an existing RNF213 tile (without touching the v2.1 output)

```bash
python -m smap.search_global <tile>/search.par --path-map /work=<project root> \
       --output-dir <tile>/output_port --batch-size 8 --log <tile>/run_port.log
python -m smap.compare <tile>/output <tile>/output_port --json <tile>/compare_port.json
```

`smap.compare` pairs particles one-to-one by search-grid position and reports final/refined
score differences, rotation angle, defocus, counts at SNR ≥ 7 and ≥ 8, search_mean/SD maps,
search_SH tails and highVals, against declared tolerances (exit code 1 if any is exceeded).
Use the same tool for CPU vs GPU and batch 1 vs batch 8.

## Provenance and schema

`search_status.json` records completeness, evaluated pose count, trial count, `arbThr`
(raw pre-selection threshold, operator `>`), score kind/stage, coordinate frame, sampling,
the cluster-based calibration summary, and provenance: SHA-256 of image/structure/SP/rotation
inputs, the `.par` text, port version and git commit (`+modified` if the package has local
edits), library versions, GPU name and run flags. `particles_schema.json` documents every
`particles.csv` column; `x_orig0_est, y_orig0_est` are zero-based original-image centres
estimated from the optimisation (the RNF213 workflow still registers saved patches for final
coordinates).

## Options (`.par` keys; v2.1 ignores unknown keys; defaults reproduce v2.1)

| Key | Default | Effect |
|---|---|---|
| `oversample K` | 1 | Zero-pad the potential K-fold before its FFT. Template CC vs exact projection 0.987 → 0.9997 at K = 2 (`experiments/e1`). GPU memory for the volume ×K³ (single precision when K > 1) |
| `dedupe_orientations 1` | 0 | Drop repeated grid orientations (φ = 0/φ_max at every θ: 4,750 of 288,990 at 3.8°). Changes rotation indices |
| `template_bfactor B` | 0 | Multiply the template CTF by exp(−B k²/4) (search, refinement, optimisation). Fit B on detected particles; see `experiments/e3` |
| `fix_even_grid 1` | 0 | Exact voxel spacing for even potential boxes (v2.1 scales coordinates by (N−1)/N) |
| `fix_pdb_short_lines 1` | 0 | Keep PDB lines shorter than 78 characters (v2.1 drops them) |
| `legacy_particle_diameter 0` | 1 | Use the intended particle-diameter box estimate (v2.1's call always fails → `edge_max`) |
| `fix_cluster_thresholds 1` | 0 | Compare pixel distance with `dThr` and angle with `qThr` (v2.1 swaps them; same result when both are 10) |
| `fix_optimize_angle 1` | 0 | `optimize_angle` uses the trial rotation (v2.1 builds the template from the starting rotation, so only the shift is optimised) |
| `refine_local_peak R` | 0 | Final full-image peak in refinement searched within R search pixels of the particle (v2.1: anywhere in the image) |
| `refine_continuous 1` | 0 | Stage C: one Nelder–Mead over rotation vector (3, deg), shift (2, px) and common defocus offset (1, nm) with initial steps inc_degrees/2, 0.5 px, 2·aPerPix nm, instead of v2.1's quaternion (4 params, one redundant) + shift with initial steps ≈0.03° and 0.025 px and no defocus. Refined defocus is written to particles.par/csv (`experiments/e4`) |
| `write_mip 0` | 1 | Do not write the MIP / p-value maps |
| `decoy_template mirror` | — | Search with the mirror image (enantiomer) of the potential: page axis reflected about the rotation centre. For target–decoy FDR (`smap.decoy`) |
| `decoy_image phase_random` | — | Search a phase-randomised copy of the image (same amplitude spectrum, `decoy_seed S`) |

`df_inc` is an existing key: `experiments/e2` shows the 50 nm default loses on average
3–15 % CC (B = 100…0 Å²) for particles between planes; 25 nm reduces this to 1–4 %.

## Target–decoy calibration (`smap.decoy`)

Run each tile twice with identical settings, the second time with `decoy_template mirror`
(writes `<model>_decoy_mirror.mrc` in the output folder) and `--output-dir <tile>/output_decoy`.
FDR(c) ≈ N_decoy(score ≥ c) / N_target(score ≥ c); q = running minimum from the top.
`python -m smap.decoy --pair T D [--pair ...] --cutoff 8` writes `.json/.csv/.png/.pdf`.
The mirror decoy keeps mass, size and spectrum; structures with near-mirror features still
partly match their mirror, so it tends to over-estimate false positives. The phase-randomised
image (`decoy_image phase_random`) removes all structure and tends to under-estimate false
positives caused by real non-target density. Either doubles the search time.

## RELION / cryoSPARC export (`smap.export`)

RELION angles are the ZYZ decomposition (RELION `Euler_matrix2angles`) of `q_opt`. Checked by
projecting the potential in real space with RELION's definition P(x,y) = Σ_z V(Aᵀ[x,y,z])
(independent of SMAP code): NCC 0.989–0.994 with the SMAP templates vs 0.74–0.83 for the
inverse rotation or the mirror volume (`tests/test_export.py`). Not yet run through RELION
itself. Defocus: U/V = df1/df2 × 10 Å, angle = −ast·180/π (= particles.par ANGAST).
Coordinates are zero-based like the RNF213 STAR files (`--coord-origin 1` otherwise).
`--images` adds `rlnImageName j@particles.mrc` (centred SMAP patches, origin 0) so
`relion_reconstruct --ctf` can run on the detected patches directly. SMAP's search does not
model amplitude contrast (`F_abs 0`); the STAR uses `--amp-contrast` (default 0.07).

## Tests (`pytest tests`)

* `test_parity_octave.py`, `test_parity_refine.py` — compared with the **original MATLAB code
  run in Octave 8.4** (`tests/refs/*.mat`, generated by `tests/octave/*.octave`).
* `test_end_to_end.py` — synthetic particle at known orientation/position through search,
  clustering, refinement and optimisation (v2.1 defaults and all options on); output files are
  parsed with the RNF213 reader semantics.
* `test_options.py` — oversampling accuracy, de-duplication, even-grid spacing, envelope,
  strict survival counts, cluster-based calibration, layout round trip, incomplete-run flag.
* `test_compare_cli.py` — full CLI runs with Docker-style `/work` paths remapped: batch 1 vs
  batch 8 agree within the `smap.compare` tolerances; the comparison detects a 0.5 change in a
  final score; `run.log` contains the RNF213 completion lines; incomplete runs are not marked done.
* `test_decoy.py` — mirror decoy keeps the rotation centre, phase randomisation keeps |F|,
  FDR/q bookkeeping, target vs mirror-decoy runs through the CLI.
* `test_export.py` — RELION Euler round trip, Frealign inverse, real-space projection check,
  STAR writer.
* `test_pvalue_mip.py` — MIP maps equal v2.1's `search_vals.mrc` where v2.1 kept values; the
  planted particle is the p-value maximum; background tail.

Not covered by Octave: `mask_a_volume` (Octave's `bwdist` is 2-D only), `read_cif_file`'s awk
call, `imresize` (Octave's differs from MATLAB; ported from MATLAB's documented algorithm
and checked against known MATLAB outputs), `rotationVectorToMatrix` and the quaternion class
(ported from source), and `fminsearch` (ported from MATLAB's algorithm; Octave's differs).
Nothing here has been compared with a MATLAB v2.1 run on experimental data yet.

## v2.1 behaviours found during the port (kept unless an option says otherwise)

1. `particleDiameter` calls `bindata` without `smap.` → always fails → SP box = `edge_max`.
2. `psdFilterFlag 0`: `resize_F(ones,1)` returns zeros → NaN output. **Port deviates** (ones filter).
3. Even potential box: coordinates scaled by (N−1)/N (`fix_even_grid`).
4. PDB lines < 78 characters dropped (`fix_pdb_short_lines`).
5. Orientation grid repeats (`dedupe_orientations`); a peak can be reported under either index.
6. `search_XXXX_SD.dat` keeps only the last defocus plane's SD.
7. `radialmeanj`: spectrum corners fall into bin 1.
8. `resize_F` downsampling removes the volume mode without adding it back.
9. 18-column CIFs: awk `$19` chain column is empty (port falls back to column 17 for chain filtering).
10. `sum1`/`sum2` are accumulated in double precision (no precision issue).
11. Duplicate suppression compares pixel distance with `qThr` and angle with `dThr`.
12. Refinement defocus scans step `2·aPerPix` then `aPerPix` in nm units (3.0 and 1.5 nm at 1.5 Å/px).
13. The refinement's final value is the maximum over the **whole** image for the refined template,
    so a particle's reported position can jump to a different particle (`refine_local_peak`).
14. `optimize_angle` ignores the four rotation parameters it is given; the Nelder–Mead still moves
    them, so `q_opt` = small arbitrary rotation × starting rotation (`fix_optimize_angle`).
15. `RotMat2e` divides by w and is inaccurate near 180° rotations (affects `pairwiseQD`); the port
    uses a stable conversion for distances and the original one for written quaternions.
16. Flat-fielded scores are only complete above `x_complete` (raw CC retained only above `arbThr`);
    `search_status.json` reports it. Search-tail counts are correlated evaluations, not particles;
    the cluster-based N_eff estimate there is informational and needs particle-free controls.
17. Stage-B refinement scores a patch the size of the template box (circular correlation). When
    the CTF delocalisation (≈ λ·Δf/d at resolution d) approaches the box size, the patch score
    is aliased: in a synthetic test with a 34-px box at 1.5 Å/px and 1.5 µm defocus the stage-B
    score was 0.4–1.0 for particles with search score 7–8, and the defocus/angle scans moved
    away from the truth; with a 78-px box at 0.8 µm it tracked the search score (5.7 → 9.7).
    RNF213 boxes are large, so the effect should be small there; not changed.
