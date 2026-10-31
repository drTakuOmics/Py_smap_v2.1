# RNF213 pilot: HRTM on purified single-particle micrographs

**What this is.** A pilot of 2D high-resolution template matching on cryo-EM micrographs of
*purified* RNF213 (EMPIAR-10711, 4092 × 5760 px, 1.072 Å/px), not on cells. The full template is
the RNF213:UBE2L3 model (PDB 7OIK); partial templates (domain-omitted "anchors", spherical
fragments) test whether detections recover density that the template did not contain. Reference
picks (`refined_particles.star`) from the single-particle analysis are used only to count
positional matches. Results are in [results/](results) and in the main [README](../../README.md).

**Data and scripts** live in the RNF213 project on the GPU workstation
(`~/Documents/RNF213-2DTM/rnf213-2dtm`, the folder Docker mounted as `/work` for v2.1). The
original workflow scripts (`rnf213_workflow_v3.py`, `rnf213_review.py`, `rnf213_batch_review.py`,
…) read the port's output unchanged. The port's helpers for this project are in
[`tools/rnf213/`](../../tools/rnf213). Keep this repository next to the project
(`~/Documents/RNF213-2DTM/Py_smap_v2.1`); v2.1 tile trees are never modified.

## Install (once)

```bash
cd ~/Documents/RNF213-2DTM
git clone https://github.com/drTakuOmics/Py_smap_v2.1.git      # or: git -C Py_smap_v2.1 pull
conda create -n pysmap -c conda-forge --override-channels python=3.11 -y && conda activate pysmap
pip install torch --index-url https://download.pytorch.org/whl/cu128   # driver 570 / CUDA 12.8
pip install -e "Py_smap_v2.1[test]"
python Py_smap_v2.1/tools/gpu_smoke.py        # CUDA vs CPU on a synthetic tile; must print PASSED
```

## Results so far

| Run | Tile 0003 (3.8°, full template) |
|---|---|
| SMAP v2.1 (MATLAB, Docker) | 13 refined candidates, 11 with final SNR ≥ 8 |
| Port, v2.1 mode | same 13 candidates, max \|Δ final SNR\| 0.014, 244 s on an RTX 3090 |
| Port, `refine_continuous 1` + `refine_local_peak 20` | same 13, all higher (mean +0.31), 11 ≥ 8, 371 s |
| Port, mirror template | 11 candidates at 8.7–11.7: the same particles (a mirror is not a 2D decoy) |

Scores per candidate: [results/tile0003_final_scores.csv](results/tile0003_final_scores.csv).
In progress: the 21-image E3-omitted anchor batch (`batch_noE3_30`, 1134 tiles) with the port.

## One tile, v2.1 mode and improved mode

```bash
cd ~/Documents/RNF213-2DTM/rnf213-2dtm
T=../Py_smap_v2.1/tools/rnf213/rnf213_port_tiles.py
python $T prepare --source tiles_3p8 --dest tiles_3p8_port
python $T prepare --source tiles_3p8 --dest tiles_3p8_plus --set refine_continuous=1 --set refine_local_peak=20
python $T run --tiles tiles_3p8_port --only tile_0003 --device cuda
python $T run --tiles tiles_3p8_plus --only tile_0003 --device cuda
python $T compare --a tiles_3p8 --b tiles_3p8_port      # v2.1 vs port (should agree)
python $T compare --a tiles_3p8 --b tiles_3p8_plus      # v2.1 vs improved
```

`prepare` rewrites only `imageFile`/`outputDir` (same `/work/...` style), hard-links the tile
images, checks that every referenced input exists and rejects unknown `--set` keys.
`run` calls `python -m smap.search_global <tile>/search.par --path-map /work=<project>
--log <tile>/run.log`, skips completed tiles with the same rule as
`rnf213_batch_workflow.py`, archives unfinished attempts under `<tile>/attempts/` and
writes `run_state.json` (elapsed seconds). `compare` runs `smap.compare` per tile and writes
`<b>/review/compare_vs_<a>/per_tile.csv`.

## Whole tree and the existing review scripts (unchanged)

```bash
python $T run --tiles tiles_3p8_port --device cuda
python rnf213_workflow_v3.py --tiles tiles_3p8_port --reference refined_particles.star --out tiles_3p8_port/review/merged
python rnf213_review.py --runs tiles_3p8 tiles_3p8_port --out review_current   # replaces review_current
```

`rnf213_review.py` only writes `review_current` and replaces a previous generated one; copy
the existing `review_current` first if you want to keep it.

Batch layout: `--source batch_noE3_3/searches --dest batch_noE3_3_port/searches`; jobs.csv is
re-pointed, so `rnf213_batch_review.py --batch batch_noE3_3_port` reads the port outputs.
Each tile keeps its own search.par, so a partial-template (anchor) batch is searched with the
same anchor model. `run --images image_0001 image_0002` restricts a batch run to some images.

## Z-score correlograms (2DTM / HRTM style)

```bash
python ../Py_smap_v2.1/tools/rnf213/rnf213_correlogram.py --tile tiles_3p8_port/tile_0003 \
       --reference refined_particles.star --out figures/correlogram_tile0003
# full vs partial template on the same region (shared z scale):
python ../Py_smap_v2.1/tools/rnf213/rnf213_correlogram.py --tile tiles_3p8_port/tile_0003 \
       --compare <partial-template port tile> --compare-label "E3/CTD-omitted anchor" --out figures/corr_full_vs_anchor
```

Overview (tile top view), one figure per candidate (3-D landscape, top view with the cutoff
contour, x/y profiles), a gallery of all candidates, `candidates.csv`; PNG/PDF/SVG + JSON.

Whole micrograph in one 3-D space (all port-searched sections stitched, candidates merged
across overlaps within 20 px and labelled "id: final SNR"):

```bash
python ../Py_smap_v2.1/tools/rnf213/rnf213_landscape.py --tree tiles_3p8_port --out figures/landscape_3p8
```

## Target–decoy and STAR export

The decoy is a different protein of similar mass and size (not a mirror image; see
PORT_NOTES). Build it matched to the template that was searched (atom count = SMAP mass,
principal-axis proportions) and check side-chain completeness in the report:

```bash
python ../Py_smap_v2.1/tools/make_decoy_model.py --decoy decoy/8DYU.cif \
       --match <anchor or full-template CIF that was searched> --out decoy/8DYU_matched
```

Then point the decoy tree at the written CIF:

```bash
python $T prepare --source tiles_3p8 --dest tiles_3p8_decoy \
       --set structureFile=/work/decoy/8DYU_matched.cif --set modelFile=/work/decoy/8DYU_matched.cif \
       --set decoy_label=8DYU_matched
python $T run --tiles tiles_3p8_decoy --only tile_0003 --device cuda
python -m smap.decoy --pair tiles_3p8_port/tile_0003/output tiles_3p8_decoy/tile_0003/output --cutoff 8 --out decoy_tile0003
python -m smap.export rnf213 tiles_3p8_port/review/merged/all_candidates_registration.csv \
       --par tiles_3p8_port/tile_0003/search.par --micrograph "$(ls *_fractions.mrc)" --min-score 8 \
       -o tiles_3p8_port/review/particles_relion.star
```

## Checked before delivery (synthetic project, CPU)

Two overlapping 1024² tiles of a synthetic 1.072 Å/px micrograph with a pilot-style
`search.par` (`aPerPix_search 1.5`, CTFFind-format defocus, `T_sample 50`, `margin_pix 96`,
`/work` paths): `prepare` → `run` (v2.1 mode and `refine_continuous`/`refine_local_peak`)
→ `compare` → unchanged `rnf213_workflow_v3.py` (registration NCC ≥ 0.99; registered
zero-based centres equal the planted centres to 0.1 px) → unchanged `rnf213_review.py`.
The CUDA path was not run here; `gpu_smoke.py` is the first check on the workstation.
