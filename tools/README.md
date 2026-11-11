# Tools

Command-line scripts around a search, grouped by the stage at which they are used. Run them from the
repository folder (`python tools/<script>.py --help` lists every option). None of them changes a
search output; figures and exports go to the folder given with `--out` or `--out-dir`.

The search itself and the analyses that belong to the package are modules: see
[Package commands](#package-commands) below.

## 1. Check the installation

| Script | Purpose |
|---|---|
| `gpu_smoke.py` | the same synthetic search on the GPU and the CPU, compared with `smap.compare` (1–2 min) |

## 2. Prepare a search

| Script | Purpose |
|---|---|
| `star_to_pars.py` | one `.par` per micrograph from a RELION particle STAR file (CTF per micrograph); the particles of each micrograph as a positive control |
| `cryosparc_ctf_to_pars.py` | one `.par` per micrograph from a cryoSPARC CTF job |
| `ctf_check.py` | the defocus of a `.par` against the Thon rings of its micrograph (angle convention, df1/df2 order, offset) |
| `radial_scan.py` | near-spherical particles without an orientation search: positions, pixel size, defocus and contrast sign |
| `template_views.py` | the templates of a search as correlated: projection, × CTF, × CTF × whitening |
| `make_fragment.py` | spherical fragments of a model as partial templates |
| `make_decoy_model.py` | a decoy model trimmed to the mass and shape of the searched template |

Symmetry operators for `symmetryFile` come from the package: `python -m smap.symmetry`.

## 3. Review a search

| Script | Purpose |
|---|---|
| `search_figures.py` | one search: detections drawn at their poses on the micrograph, z-score map, z-score landscape |
| `overlay_figures.py` | several searches of the same micrograph in one figure, one colour per search (for example two states of a complex) |
| `match_positions.py` | detections against a reference particle list: found, missed and extra at several cutoffs |

## 4. From detections to a map

| Script | Purpose |
|---|---|
| `export_detections.py` | the detections of many searches as one RELION STAR file with normalised particle stacks; duplicates removed, crowded micrographs left out, orientations compared with a reference |
| `check_reconstruction.py` | a reconstruction from that STAR file with RELION's conventions, compared with the template, before cryoSPARC or RELION |
| `expand_symmetry.py` | symmetry expansion of the STAR file with the operators of the search (needed after a search with `symmetryFile`) |

Order: `export_detections.py` → `check_reconstruction.py` → (`expand_symmetry.py`) → import.
Worked example: [`examples/aav2_capsid`](../examples/aav2_capsid); problems: [`docs/FAQ.md`](../docs/FAQ.md), section 11.

## 5. Project-specific and maintenance

| Script | Purpose |
|---|---|
| `rnf213/rnf213_port_tiles.py` | run the port on tile trees prepared for SMAP v2.1 (RNF213 project layout) and compare the two |
| `rnf213/rnf213_correlogram.py` | z-score peak landscapes of one tile |
| `rnf213/rnf213_landscape.py` | the tiles of a micrograph stitched into one landscape |
| `gen_sf_table.py` | regenerate `smap/_sf_table.py` from the original MATLAB scattering-factor table |

## Package commands

| Command | Purpose |
|---|---|
| `python -m smap.search_global SEARCH.par` | the search: setup, global search, refinement, output files |
| `python -m smap.compare A B` | two output folders against declared tolerances (v2.1 vs port, CPU vs GPU) |
| `python -m smap.cutoff RESULT` | detection cutoffs from the z-score map of a search |
| `python -m smap.decoy --pair TARGET DECOY` | target–decoy false-discovery estimate on refined candidates |
| `python -m smap.symmetry MODEL.cif --out OPS.txt` | rotations of the model's point group for `symmetryFile` |
| `python -m smap.export smap RESULT ...` | detections of single searches as a RELION STAR file (coordinates and angles) |
| `python -m smap.qc`, `python -m smap.bias` | per-particle quality measures; template bias of omitted-region reconstructions |
