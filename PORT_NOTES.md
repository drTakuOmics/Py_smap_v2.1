# `smap/` — Python/PyTorch port of SMAP v2.1 `smappoi_search_global`

Source of truth: `smap_v2.1/src/smap_tools/smappoi_search_global.m` and `@smap/*.m`
(J. P. Rickgauer, MIT License). Translated one-to-one from the MATLAB; no algorithmic changes.

## Status

| MATLAB lines | Step | Python | Status |
|---|---|---|---|
| 51–354, 1992–2381 | params, rotations, defocus stack, image whitening, PDB/CIF → scattering potential | `search_global.prepare`, `structure.py` | ported, tested |
| 355–485 | CTF stack, Fourier volume, thresholds | `prepare`, `ctf.py` | ported, tested |
| 486–627 | main loop: template → FFT → ×CTF → SD-normalise → CC; moments; above-threshold list | `search_global.run_job` | ported, tested |
| 628–863 | combine jobs, pixelwise normalisation, `search_SH.txt`, `highVals.txt` | `search_global.combine` | ported, tested |
| 864–1990 | `clusterImByThr`, `pairwiseQD`, particle refinement/optimisation | — | **not ported yet** |

Array convention: Python axis k = MATLAB dimension k+1 (images `[row, col]`, volumes `[row, col, page]`),
rotations stored `(n, 3, 3)` with `R[i-1] == R(:,:,i)`. Output files use v2.1 names and formats.

## Run

```bash
pip install -e .            # needs numpy scipy mrcfile torch (CUDA build for GPU)
python -m smap.search_global sample_search.par                 # all nCores jobs in one process
python -m smap.search_global sample_search.par 2               # job 2 only (like smap_run.sh)
python -m smap.search_global sample_search.par --batch-size 8  # batched orientations per GPU pass
```

`--batch-size 1` reproduces MATLAB's one-orientation-at-a-time order; larger values change only
scheduling. Precision follows MATLAB: projections in float64, FFT pipeline float32, moments float64.

## Tests (`pytest tests_port`)

* `test_parity_octave.py` — compares against the **original MATLAB code executed in Octave 8.4**
  (`tests_port/refs/refs.mat`, made by `tests_port/octave/make_reference.m`): cropOrPad, resizeForFFT,
  rrj, nm, resize_F, radialmeanj, psdFilter, mask_central_cross, bindata, applyPhaseShifts, ctf,
  calculate_search_grid, assignJobs, readParamsFile, the main CC loop (lines 290–560, 5 rotations ×
  2 defoci, batch 1 and 3), per-atom potential and SP post-processing.
  Octave shims (`tests_port/octave/shims`) replace `nanmean`, `nansum`, `gather`, `split`.
* `test_end_to_end.py` — synthetic particle inserted at a known orientation/position; the full
  `prepare → run_job (2 jobs) → combine` path must return it as the top `highVals.txt` entry.

Not covered by Octave: `mask_a_volume` (Octave's `bwdist` is 2-D only) and the `read_cif_file`
awk call (Octave's `textscan` collapses empty fields differently from MATLAB).

## v2.1 behaviours found during the port (kept unless noted)

1. `particleDiameter` calls `bindata` without the `smap.` prefix → always fails (confirmed in Octave),
   so the SP box is always `edge_max` (512) or the model size if larger. Kept as default;
   `legacy_particle_diameter 0` in the .par enables the intended estimate.
2. `psdFilterFlag 0` is broken: `resize_F(ones,1,'newSize')` subtracts the mode and returns zeros
   → all templates 0, SD = 0, NaN output (confirmed in Octave). **Port deviates**: uses a ones filter.
3. Scattering-potential grid spacing is `(N-1)/N · aPerPix` when the box edge N is even
   (from `rrj(...)*(N-1)`): the model is shrunk by 1/N (e.g. 0.3 % at N = 296). Odd N is exact
   (6ek0_LSU gives N = 327). Kept; worth fixing upstream.
4. `read_pdb_file` skips every line shorter than 78 characters → PDB files without element
   columns lose atoms silently. Kept.
5. Orientation grid includes both φ = 0° and φ = 360° at every θ (duplicates, ~1.6 % of 288,990
   orientations at 3.8°). Kept.
6. `search_XXXX_SD.dat` stores only the SD of the last defocus plane (overwritten in the df loop).
   Normalisation itself is correct. Kept.
7. `radialmeanj`: pixels beyond the last radial bin (spectrum corners) fall into bin 1 (`histc` → 0 →
   clamped to 1). Kept.
8. `resize_F` downsampling removes the volume mode and does not add it back. Kept.
9. 6ek0_LSU.cif has 18 columns, so the awk `$19` chain column is empty; chain selection would
   return nothing. Port falls back to column 17 for chain filtering.
10. `sum1`/`sum2` are accumulated in **double** precision in v2.1 (gpuArray double), not single.
