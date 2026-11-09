# Py_smap_v2.1

[![High-resolution template matching: every projection against every pixel](docs/figures/hrtm_banner.gif)](docs/hrtm_banner.html)

<sub>The atomic model is projected at every orientation (template bank); each projection is compared with every pixel of the image. Peaks of the z-score map are particles, and the best-scoring projection gives the pose. Interactive version: [docs/hrtm_banner.html](docs/hrtm_banner.html) (download and open in a browser). Drawn with [Colorfigures](https://april-zhu.com/colorfigures) (MIT, April Zhu).</sub>

**Open-source Python/PyTorch port of SMAP v2.1 — high-resolution template matching (HRTM) for
cryo-EM images.**

[![python-tests](https://github.com/drTakuOmics/Py_smap_v2.1/actions/workflows/python-tests.yml/badge.svg)](https://github.com/drTakuOmics/Py_smap_v2.1/actions/workflows/python-tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-2878B5)
![license](https://img.shields.io/badge/license-MIT-555555)

SMAP finds individual copies of a known macromolecule in a single 2D cryo-EM image. It compares
the image with noise-free projections of an atomic model over all orientations and several
defocus planes, and it validates every hit against the statistics of the whole search, without
2D classification or averaging. This repository translates SMAP v2.1 (J. Peter Rickgauer,
HHMI Janelia) from MATLAB to Python, with the original author's permission. It keeps the same
algorithm, `.par` files and output files.

![How a search works](docs/figures/workflow.svg)

## Why a Python port

- **Open from end to end.** v2.1 needs MATLAB or the MATLAB Runtime inside Docker. The port
  installs with `pip`; it needs no licence, no container and no `sudo`.
- **The same science, checked.** The code was translated one function at a time and compared
  with the original MATLAB code run in Octave. On real data (RNF213, below) the port finds the
  same 13 candidates as the MATLAB v2.1 run, with final scores within 0.014.
- **Runs on the GPUs you have.** PyTorch on any CUDA GPU, or on CPU, with orientations batched
  on the GPU. One 1024 × 1024 tile searched over 289,000 orientations (3.8°) took 244 s on an
  RTX 3090.
- **Readable and extensible.** Tests run in CI. Every run writes a provenance file (input
  hashes, code version, GPU). New analyses are opt-in (below); with none of them set, the
  port reproduces v2.1, including its known quirks
  ([PORT_NOTES.md](PORT_NOTES.md)).

**Added in the port** (none of these change a v2.1 output file):

- full-image z-score, MIP and p-value maps (Zhang et al., IUCrJ 2025);
- 3D peak landscapes ("correlograms");
- whole-micrograph figures: fitted poses drawn on the micrograph, z-score map, landscape with the cutoff;
- target–decoy FDR estimates;
- RELION/cryoSPARC STAR export;
- a tool that compares two runs;
- optional refinement improvements.

## Two kinds of data

| | In situ: cells | Purified protein: single-particle cryo-EM |
|---|---|---|
| Example | Legacy SMAP v2.1 sample: mouse embryonic fibroblast; 60S ribosomal subunit (LSU, PDB 6EK0); decoy: rearranged *E. coli* ribosome (PDB 5J5B) | **RNF213 pilot:** purified RNF213 micrographs (EMPIAR-10711); model PDB 7OIK (RNF213:UBE2L3) and partial (domain-omitted) templates |
| Folder | [`examples/ribosome_in_situ`](examples/ribosome_in_situ) | [`examples/rnf213_purified`](examples/rnf213_purified) |
| Question | Locate and orient one complex in a crowded cell | Detect ~600 kDa particles and test the port, refinement options and validation controls on a purified sample |

HRTM was developed for in situ data. In a purified sample there are many particles per image
and the ice is thin, so the RNF213 pilot is a controlled test bed before applying the same
workflow to cells.

## The `.par` file in brief

A search is described by a plain-text `.par` file with one `key value` per line; `#` starts a
comment. The essential keys:

| Key | Meaning |
|---|---|
| `function search_global` | the program to run (global 2D search) |
| `imageFile` | pre-processed (motion-corrected, summed) micrograph `.mrc` |
| `structureFile` / `modelFile` | atomic model (`.cif`/`.pdb`, converted to a scattering potential) or a potential `.mrc` |
| `outputDir` | where results go |
| `aPerPix`, `aPerPix_search` | pixel size of the image, and of the search (the image is resampled) |
| `defocus` | `df1 df2 angle` in CTFFind convention (Å, Å, degrees) |
| `V_acc`, `Cs` | acceleration voltage (V) and spherical aberration (m) |
| `angle_inc`, `psi_inc` | orientation sampling (degrees); 3.8° gives 289,000 orientations |
| `T_sample`, `df_inc` | sample thickness and defocus-plane spacing (nm) |
| `arbThr`, `optThr` | values stored for flat-fielding, and the score needed to refine a candidate |
| `nCores` | number of jobs (one per GPU) |

The upstream README ([docs/SMAP_v2.1_README.md](docs/SMAP_v2.1_README.md)) documents every key
and output file. The port's opt-in keys are listed in [PORT_NOTES.md](PORT_NOTES.md) (section "Options").

## Install

```bash
conda create -n pysmap -c conda-forge --override-channels python=3.11 -y
conda activate pysmap
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your NVIDIA driver (or /whl/cpu)
pip install -e ".[test]"
python tools/gpu_smoke.py        # same synthetic search on CUDA and CPU; prints "GPU SMOKE TEST PASSED"
pytest                           # CPU, a few minutes; no MATLAB or Octave needed
```

## Run

```bash
# legacy in situ example (the .par uses the upstream Docker paths /opt/smap/...)
python -m smap.search_global examples/ribosome_in_situ/sample_search.par \
       --path-map /opt/smap=examples/ribosome_in_situ --output-dir results/ribosome \
       --batch-size 8 --log results/ribosome_run.log

# compare two runs (MATLAB v2.1 vs port, CPU vs GPU, option A vs B)
python -m smap.compare results/run_a results/run_b --json compare.json
```

The decoy for the ribosome example is in the same folder (`model/5j5b_monster.pdb`). Run it the
same way with `modelFile` set to that file, then compare score distributions with
`python -m smap.decoy`.

## Example 1: RNF213 pilot (purified, single-particle micrographs)

Micrographs (4092 × 5760 px, 1.072 Å/px) were cut into 1024² tiles. Each tile was searched
with the full RNF213:UBE2L3 template (7OIK) and, for validation, with a partial template in
which the E3 module was omitted. The workflow scripts, commands and the tile runner for the
port are in [examples/rnf213_purified](examples/rnf213_purified).

**1. The port reproduces SMAP v2.1 on real data.** On tile 0003 both find the same 13
candidates in the same order, 11 of them with final SNR ≥ 8. Final scores differ by at most
0.014 (blue). With the optional refinement (`refine_continuous 1`, `refine_local_peak 20`,
ochre) every score rises, by +0.31 on average. Some of that rise is expected even for noise,
because more parameters are fitted, so the cutoff for this mode has to be set with a decoy.

![v2.1 vs port](docs/figures/rnf213_tile0003_parity.png)

On the 21-micrograph anchor batch (below), 1,027 tiles of 19 micrographs were searched with both:
pysmap paired 1,763 of 1,793 v2.1 candidates (98.3%), with final-SNR differences of mean −0.0006
and SD 0.075. On the same RTX 3090 and the same tiles, pysmap (`--batch-size 8`) used about
3.3× less GPU time than SMAP v2.1 (72 vs 236 GPU hours, wall time including start-up)
([details](examples/rnf213_purified/README.md#3-port-check-pysmap-vs-smap-v21-on-the-same-tiles)).

**2. One micrograph, three angular steps.** The pilot micrograph was searched with the full 7OIK
template at 5°, 3.8° and 1.8° (SMAP v2.1): 137, 162 and 179 detections on the 41 tiles finished
in all three runs. At 3.8°, every accepted detection drawn as the 7OIK surface at its fitted pose,
at physical scale (close-ups, orientations and the anchor batch:
[examples/rnf213_purified](examples/rnf213_purified/README.md#workflow-recap)):

![pilot micrograph with fitted 7OIK surfaces](examples/rnf213_purified/results/figures/pilot_surface_overview_3p8.png)

**3. The full z-score map is now available.** v2.1 stored scores only above `arbThr`. The port
writes the whole map, so every candidate can be shown as a 3D peak landscape (the per-pixel
z-score at the search stage, before refinement):

![tile 0003 correlogram gallery](docs/figures/rnf213_tile0003_gallery.png)

| One candidate: peak, top view and profiles | Whole tile with the cutoff-8 contour |
|---|---|
| ![candidate 8](docs/figures/rnf213_tile0003_candidate08.png) | ![tile overview](docs/figures/rnf213_tile0003_overview.png) |

The tiles of a micrograph can be stitched into one landscape (`tools/rnf213/rnf213_landscape.py`;
here the pilot micrograph, where only tile 0003 was searched with the port, the others grey):

![micrograph landscape](docs/figures/rnf213_micrograph_landscape_tile0003.png)

**4. Validation controls.**

- A mirror-image (enantiomer) template is **not** a decoy for 2D searches. Every projection of
  the mirrored model equals a projection of the original at another orientation, so handedness
  is invisible in a single projection. The repository's tests confirm this numerically, and
  the mirror search of tile 0003 found the same particles again (final SNR 8.7–11.7).
  Enantiomer controls stay valid for 3D searches in reconstructions or tomograms.
- A decoy is a **different structure of comparable mass and resolution** that cannot be in the
  sample, as in Rickgauer et al. 2024, where an *E. coli* ribosome was the decoy for mammalian
  LSU searches. `tools/make_decoy_model.py` trims a candidate decoy (e.g. a dynein motor
  domain) to the searched template's atom count and shape. It also reports truncated side
  chains, which would weaken the decoy.
- Identity is tested with **omitted-domain (anchor) searches**: particles found with the
  E3-omitted template are reconstructed and checked for the omitted density. The 21-image
  anchor batch and the port check on it are in the example.

## Example 2: 60S ribosomes in a cell (legacy SMAP sample, whole micrograph)

`sample_search.par` searches an 864-px crop of the legacy micrograph. Here the whole micrograph
(3456 × 3456 px, 1.032 Å/px, 357 nm across) was searched with the same settings
(1.5 Å search sampling, 3.8°: 288,990 orientations, 5 defocus planes over 200 nm): once with the
60S subunit (6EK0) and once with the decoy, a rearranged *E. coli* ribosome of similar mass
(5J5B), which is not in the sample.

75 candidates passed `optThr 7` and were refined. 69 have a final score ≥ 7.25 (66 ≥ 8; the
highest is 17.3). The search took 48 min on one RTX 3090 (10.0 ms per orientation, batch size 8).

![LSU detections on the whole micrograph](examples/ribosome_in_situ/results/figures/full_LSU_summary.png)

<sub>Left: each detection with final score ≥ 7.25 drawn as the 6EK0 surface at its fitted pose
and position. Middle: search z-score over the whole micrograph (12 Å max-binning), cyan contour
at 7.25, detections circled (grey: refined, below the cutoff). Right: the same map as a
landscape with the cutoff plane; labels mark the 12 highest detections. Close-ups, micrograph
next to the fitted surfaces:
[full_LSU_poses_pairs.png](examples/ribosome_in_situ/results/figures/full_LSU_poses_pairs.png).</sub>

**Where to put the cutoff.** The four map-based estimates and the target–decoy cutoff at
q ≤ 0.05 agree within 0.25 (7.10–7.34):

| Method | Cutoff |
|---|---|
| v2.1 Gaussian threshold (one false positive per image) | 7.34 |
| Tail fit of the z-score map (generalised Pareto) | 7.23 |
| Extreme-value fit of the z-score map (GEV) | 7.15 |
| Decoy (5J5B) z-score map | 7.10 |
| Target–decoy on refined candidates, q ≤ 0.05 | 7.25 |
| Target–decoy, q ≤ 0.01 | 7.90 |

The decoy search gave 8 refined candidates (final scores 6.81–7.86). Three of them are above
7.25, against 69 LSU detections (estimated FDR 4%); none is above 7.9.

```bash
P="--path-map /opt/smap=examples/ribosome_in_situ --batch-size 8"
python -m smap.search_global examples/ribosome_in_situ/full_search.par $P --output-dir results/full_LSU
python -m smap.search_global examples/ribosome_in_situ/full_search_decoy.par $P --output-dir results/full_decoy
python -m smap.cutoff results/full_LSU --decoy results/full_decoy --plot
python -m smap.decoy --pair results/full_LSU results/full_decoy --out results/decoy_calibration
python tools/search_figures.py results/full_LSU --cutoff 7.25 --label "LSU (6EK0)" --out results/figures_LSU
```

## Other workflows and troubleshooting

- [`examples/decolace_tric/`](examples/decolace_tric): TRiC in cell lamellae (DeCo-LACE, EMPIAR-11063), with
  the controls needed to read the detections: deposited ribosomes as positive control, phase-randomised
  tiles for the noise rate after refinement, a decoy protein, image-artifact zones, and the open state.
- [`examples/aav2_capsid/`](examples/aav2_capsid): AAV2 capsid on purified virus-like particles
  (EMPIAR-10202) with the icosahedral grid reduction (`symmetryFile`): symmetry operators from a
  deposited full assembly, CTF and positive control from the published RELION particle list, and the
  detected capsids coloured by radius.
- [`docs/FAQ.md`](docs/FAQ.md): common problems (jobs, GPU memory, `.par` pitfalls, CTF and pixel size,
  thresholds and noise, controls) with causes, fixes and commands.

## Repository layout

| Path | Contents |
|---|---|
| `smap/` | the port: `search_global` (setup, GPU search, combine, CLI), `refine` (clustering, refinement, optimisation, output files), `structure` (atomic model → scattering potential), `ctf`, `grid`, `rot`, `io`, `utils`, `stats`; new: `compare`, `decoy`, `export`, `pvalue`, `provenance`, `symmetry` |
| `examples/ribosome_in_situ/` | upstream SMAP v2.1 sample: image, LSU target, ribosome decoy, rotations, `sample_search.par`; whole-micrograph searches (`full_search.par`, `full_search_decoy.par`) and their figures |
| `examples/rnf213_purified/` | RNF213 pilot: workflow recap, workflow scripts, results (figures, tables) |
| `examples/aav2_capsid/` | AAV2 capsid: icosahedral search, `.par` files from a RELION STAR file, comparison with the published particles, figures |
| `examples/decolace_tric/` | TRiC in DeCo-LACE lamellae: tile selection, searches, controls, re-refinement, review and one-page summary; TRiC templates (closed, open) |
| `tools/` | `gpu_smoke.py`, `make_decoy_model.py`, `gen_sf_table.py`, `search_figures.py` (poses, z-score map and landscape of one search), `ctf_check.py` (CTF against the Thon rings), `cryosparc_ctf_to_pars.py` (per-micrograph `.par` from cryoSPARC), `star_to_pars.py` (per-micrograph `.par` and particle lists from a RELION STAR file), `match_positions.py` (detections against a reference particle list), `radial_scan.py` (positions, pixel size, defocus and contrast sign of near-spherical particles, without an orientation search); `tools/rnf213/`: tile runner, correlograms, micrograph landscape |
| `tests/` | Octave-parity tests (references generated from the original MATLAB code), synthetic end-to-end, CLI and tool tests |
| `experiments/` | small numerical studies behind the optional improvements |
| `docs/` | upstream SMAP v2.1 README, figures, interactive banner (`hrtm_banner.html`), [`FAQ.md`](docs/FAQ.md) |
| [`PORT_NOTES.md`](PORT_NOTES.md) | technical notes: status, flags, options, outputs, tests, 17 v2.1 behaviours found during the port |

## Citing

If you use this software, please cite the original method and software:

- Rickgauer J.P., Choi H., Moore A.S., Denk W., Lippincott-Schwartz J. Structural dynamics of
  human ribosomes in situ reconstructed by exhaustive high-resolution template matching.
  *Mol. Cell* 84, 4912–4928 (2024). [doi:10.1016/j.molcel.2024.11.003](https://doi.org/10.1016/j.molcel.2024.11.003).
  SMAP v2.1 version of record: [doi:10.5281/zenodo.13994274](https://doi.org/10.5281/zenodo.13994274).
- Rickgauer J.P., Grigorieff N., Denk W. Single-protein detection in crowded molecular
  environments in cryo-EM images. *eLife* 6, e25648 (2017). [doi:10.7554/eLife.25648](https://doi.org/10.7554/eLife.25648).

and, when you use the p-value maps: Zhang K. et al., *IUCrJ* 12 (2025)
([eh5020](https://journals.iucr.org/m/issues/2025/02/00/eh5020/)).

RNF213 example data: EMPIAR-10711; model PDB 7OIK (Ahel J. et al.).

## License

MIT. The original SMAP v2.1 copyright notice is kept in [LICENSE](LICENSE).
