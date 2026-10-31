# Running the port on the RNF213 project (Linux GPU workstation)

Project root (the folder Docker mounts as `/work`): `~/Documents/RNF213-2DTM/rnf213-2dtm`.
Keep this repository outside the project, e.g. `~/Documents/RNF213-2DTM/Py_smap_v2.1`.
No Docker, no sudo. The v2.1 tile trees are never modified.

## Install (once)

```bash
cd ~/Documents/RNF213-2DTM
git clone https://github.com/drTakuOmics/Py_smap_v2.1.git      # or: git -C Py_smap_v2.1 pull
source <the venv you use for rnf213_workflow_v3.py>/bin/activate
pip install -e "Py_smap_v2.1[test]"
python Py_smap_v2.1/tools/gpu_smoke.py        # CUDA vs CPU on a synthetic tile; must print PASSED
```

If `gpu_smoke.py` reports that CUDA is unavailable, the installed PyTorch wheel does not
match the NVIDIA driver; check `nvidia-smi` (driver / CUDA version) and install the matching
wheel from pytorch.org.

## One tile, v2.1 mode and improved mode

```bash
cd ~/Documents/RNF213-2DTM/rnf213-2dtm
T=../Py_smap_v2.1/tools/rnf213_port_tiles.py
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

## Z-score correlograms (Leopard-EM / HRTM style)

```bash
python ../Py_smap_v2.1/tools/rnf213_correlogram.py --tile tiles_3p8_port/tile_0003 \
       --reference refined_particles.star --out figures/correlogram_tile0003
# full vs partial template on the same region (shared z scale):
python ../Py_smap_v2.1/tools/rnf213_correlogram.py --tile tiles_3p8_port/tile_0003 \
       --compare <partial-template port tile> --compare-label "E3/CTD-omitted anchor" --out figures/corr_full_vs_anchor
```

Overview (tile top view), one figure per candidate (3-D landscape, top view with the cutoff
contour, x/y profiles), a gallery of all candidates, `candidates.csv`; PNG/PDF/SVG + JSON.

## Target–decoy and STAR export

The decoy is a different protein of similar mass and size (not a mirror image; see
PORT_NOTES). Put its model in the project and point the decoy tree at it:

```bash
python $T prepare --source tiles_3p8 --dest tiles_3p8_decoy \
       --set structureFile=/work/decoy/DECOY.cif --set modelFile=/work/decoy/DECOY.cif --set decoy_label=DECOY_NAME
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
