# TRiC in cell lamellae (DeCo-LACE, EMPIAR-11063)

Scripts and settings for searching the chaperonin TRiC (CCT, ~0.95 MDa) in the DeCo-LACE tiles of
mouse ER-HoxB8 neutrophil-like cells (Elferich et al., eLife 2022, EMPIAR-11063), with the controls
needed to interpret the detections. The deposited cisTEM project of each lamella provides per-tile CTF
values and the authors' 2DTM peaks of the large ribosomal subunit, which serve as a positive control.

Run everything from the repository folder; data and results go to a working folder:

```
export WORK=/path/to/decolace
cd /path/to/Py_smap_v2.1
```

Problems and fixes: [`docs/FAQ.md`](../../docs/FAQ.md).

## Data and templates

| Item | Source |
|---|---|
| Tiles, masks, CTF, deposited peaks | EMPIAR-11063 (aligned micrographs at 1.5 Å/px; one cisTEM database per lamella) |
| TRiC, closed | `models/7NVL_mouse.cif.gz` (see [`models/README.md`](models/README.md)) |
| TRiC, open | `models/open_hybrid_mouse.cif.gz` |
| Ribosome control | mouse large subunit, PDB 6SWA without Ebp1 (`make_lsu_mouse.py`); human 6EK0 in `examples/ribosome_in_situ/model/` |
| Decoy | lumazine synthase, PDB 1HQK assembly (60-mer, ~1.0 MDa; absent from mammals) |

Search settings (`tric_decolace.par`): 1.5 Å/px, 3.8° angular step (288,990 orientations), 5 defocus
planes over 200 nm around each tile's ctffind defocus, refinement of candidates with search score ≥ 7.
Per-tile pixel size, voltage, Cs and defocus come from the database.

## Steps

**1. Select tiles and download** (tiles ranked by the number of deposited ribosomes; the dark area
outside the beam is filled with noise, as in the DeCo-LACE paper):

```
python examples/decolace_tric/decolace_prep.py select --dest $WORK --n 50
python examples/decolace_tric/decolace_prep.py fetch --dest $WORK
```

**2. Search** (resumable; one search per tile; `status.py` shows the state of every search):

```
BATCH=4 nohup bash examples/decolace_tric/run_queue.sh $WORK/jobs_LSU_control.txt $WORK/jobs_TRiC.txt > $WORK/queue.log 2>&1 &
python examples/decolace_tric/status.py
```

**3. Controls** (each writes job lists for `run_queue.sh`):

| Script | Control |
|---|---|
| `make_controls.py --phaserand 00463 00453` | phase-randomised copies of tiles: noise rate after refinement |
| `make_controls.py --filled 00520` | image-artifact zone filled with noise: are edge detections artifacts? |
| `make_decoy_jobs.py --model 1HQK-assembly1.cif` | decoy protein on the same tiles and settings |
| `make_lsu_mouse.py --model 6swa.cif` | ribosome control with the mouse template |
| `make_open_jobs.py` | open-state template on the same tiles |

```
wget -O $WORK/models/1HQK-assembly1.cif.gz https://files.rcsb.org/download/1HQK-assembly1.cif.gz && gunzip -f $WORK/models/1HQK-assembly1.cif.gz
python examples/decolace_tric/make_decoy_jobs.py --dest $WORK --model $WORK/models/1HQK-assembly1.cif
```

**4. Refine more candidates without searching again** (optional; same threshold for everything compared):

```
python -u examples/decolace_tric/refine_more.py --dest $WORK --target TRiC --tiles 00453 00463 00334 --opt-thr 6.5 --device cuda
```

**5. Review and report**:

```
python examples/decolace_tric/decolace_review.py --dest $WORK
python examples/decolace_tric/tric_report.py --dest $WORK --strict 7.8
python examples/decolace_tric/lsu_gap.py --dest $WORK --results results_LSU_mouse
python examples/decolace_tric/tric_summary.py --dest $WORK
```

`decolace_review.py` matches the ribosome control with the deposited peaks, masks image-artifact
zones, and tests TRiC detections for overlap with ribosomes and pose coherence. `tric_report.py`
counts detections per tile for TRiC, open state, decoy and noise at two cutoffs: the per-tile
one-false-positive cutoff of the search, and a cutoff set from the phase-randomised tiles
(`--strict`). `tric_summary.py` collects everything, including the picked templates of the closed and
open states on the same tiles, in `$WORK/report/index.html`. For re-refined runs use
`--tag opt6.5` (and `--lsu results_LSU_mouse_opt6.5` for the review).

## Files

| File | Purpose |
|---|---|
| `decolace_prep.py` | tile selection, CTF from the database, template potentials, downloads |
| `tric_decolace.par` | common search settings |
| `run_queue.sh`, `status.py` | resumable queue; state of every search |
| `make_controls.py`, `make_decoy_jobs.py`, `make_lsu_mouse.py`, `make_open_jobs.py` | control and comparison searches |
| `refine_more.py` | refinement of more candidates from saved search maps |
| `decolace_review.py`, `tric_report.py`, `lsu_gap.py`, `tric_summary.py` | review, figures, one-page summary |
| `models/` | TRiC templates (closed, open) |
