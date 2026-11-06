# Troubleshooting and FAQ

Common problems when running 2D template matching with pysmap on cellular lamellae
(DeCo-LACE tiles, EMPIAR-11063) and on purified particles (MS2 virions, EMPIAR-10075).
Each entry gives the symptom, the cause, the fix and, where useful, the command.

Commands are run from the repository folder; the helper scripts are in
[`examples/decolace_tric/`](../examples/decolace_tric) and [`examples/ms2_capsid/`](../examples/ms2_capsid).
Data and results live in working folders outside the repository; set them once per shell:

```
export PYSMAP=/path/to/Py_smap_v2.1      # repository clone
export WORK=/path/to/decolace            # DeCo-LACE working folder (tiles, .par files, results, report)
export MS2_WORK=/path/to/ms2_work        # MS2 working folder
export MS2_DATA=/path/to/ms2_empiar10075 # MS2 micrographs and models
conda activate pysmap
cd $PYSMAP
```

A single page with all DeCo-LACE results: `python examples/decolace_tric/tric_summary.py --dest $WORK`,
then open `$WORK/report/index.html`.

---

## 1. Running long jobs

### 1.1 No tmux or screen on the machine
Start jobs in the background with `nohup` and follow the log file.

```
nohup bash examples/decolace_tric/run_queue.sh $WORK/jobs_TRiC.txt > $WORK/run.log 2>&1 &
tail -f $WORK/run.log
```

### 1.2 The log file stays empty while the job is running
- **Cause:** Python buffers its output when writing to a file instead of a terminal.
- **Fix:** run Python with `-u`. The helper scripts flush every line themselves.

### 1.3 Checking what is running

```
python examples/decolace_tric/status.py
pgrep -af "run_queue|search_global|refine_more"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
```

### 1.4 `kill PID` does not stop the job
- **Cause:** `cd ... && nohup python ... &` starts a sub-shell; the PID printed after `&` is the shell's,
  not Python's.
- **Fix:** stop the process by name. "Terminated" or "Exit 143" afterwards confirms it.

```
pkill -f refine_more.py
```

### 1.5 A `tail -f` window shows progress that does not change
The window is following an old log file. Check the current state with `status.py` (1.3).

### 1.6 Changing the order of queued jobs
A queue is a waiting shell such as `while ! grep -q "ALL DONE" log; do sleep 300; done; ...`.
While it is still waiting, stop it by a name in its command line and start the new order. To start
something after a given process ends, wait on its PID:

```
nohup bash -c 'while kill -0 PID 2>/dev/null; do sleep 300; done; NEXT_COMMAND' > LOG 2>&1 &
```

### 1.7 `No module named smap`
Run from the repository folder, in the `pysmap` environment (see the top of this page).

---

## 2. Input data

### 2.1 HTTP 404 when downloading DeCo-LACE tiles
- **Cause:** the cisTEM database names the latest alignment (`*_2.mrc`); EMPIAR may hold only `*_0.mrc`.
- **Fix:** `decolace_prep.py fetch` tries `_0`, `_1` and `_2`, skips missing files and lists them in
  `fetch_failed.txt`.

### 2.2 A queue waits indefinitely for an image
`run_queue.sh` waits for a missing image only while `decolace_prep.py fetch` is running, so a
failed download no longer blocks the queue.

### 2.3 Pixel sizes differ slightly between tiles
Tiles carry their own pixel size (for example 1.4999 vs 1.5 Å). Each tile's `.par` uses its own
`aPerPix`; selection accepts differences up to `--pixel-tol` (default 0.001 Å).

### 2.4 Disk space
A search given a `.cif` writes the template potential (`*_SP.mrc`, `*_EP.mrc`, about 0.5 GB each) into
its result folder. Compute the potential once and set `structureFile` to the `*_SP.mrc`.

---

## 3. Parameter (`.par`) files

### 3.1 A setting silently takes a wrong value
- **Cause:** the SMAP v2.1 reader matches a key anywhere in a line, so a path that contains a key name
  is misread; lines containing `#` are ignored.
- **Fix:** read every generated `.par` back and compare (`check_par` in the helper scripts).

### 3.2 Defocus format
With `defocus_format CTFFind`, `defocus df1 df2 angle` is given in Å, Å and degrees, as written by
ctffind, cisTEM and cryoSPARC. pysmap converts to nm and radians internally.

### 3.3 Number of defocus planes
Planes = `T_sample / df_inc + 1` (200 / 50 → 5). GPU memory grows with the number of planes (5.2).

---

## 4. CTF and pixel size

### 4.1 Per-tile and per-particle defocus
- **Per tile:** each tile has its own `.par` with its own df1, df2 and astigmatism angle. For
  DeCo-LACE these are the ctffind values stored in the deposited cisTEM database; the
  `TEMPLATE_MATCH_LIST` table records that the deposited search used the same values.
- **Per particle:** the tile defocus is the centre of the search planes. Refinement then scans
  ±50 nm for every candidate (`df1_nm`, `df2_nm` in `particles.csv`).

### 4.2 Checking a CTF against the image
`tools/ctf_check.py` compares the Thon rings with the CTF as given, with the astigmatism angle
flipped and with df1/df2 swapped, and scans a mean-defocus offset. The angle can only be tested on
images with astigmatism above about 500 Å; weak rings give low correlations.

```
for p in $WORK/pars_TRiC/*.par; do python tools/ctf_check.py $p | sed 's/^[^:]*: //'; done
```

### 4.3 Cs-corrected microscopes
Use the actual Cs (EMPIAR-10075: 0.02 mm) both for CTF estimation and in the search.

### 4.4 A large particle is never detected
- **Symptom:** maximum score about 7 on every micrograph although particles are visible.
- **Cause:** pixel size error. For a 270 Å capsid, 4% (1.16 Å instead of 1.114 Å) moves the rim by
  about 5.5 Å, more than the resolution of the search.
- **Diagnosis:** `ms2_ctf_scan.py` correlates the image with a rotationally averaged capsid template,
  without an orientation search, while scanning pixel size, defocus and contrast sign.
- **Fix:** rebuild the `.par` files at the calibrated pixel size. Defocus scales with
  (new / old pixel)², which is exact for Cs ≈ 0; the CTF does not need to be re-estimated.

```
python examples/ms2_capsid/ms2_ctf_scan.py $MS2_WORK/results_a3.8/*/ --fine --out $MS2_WORK/diag
python examples/ms2_capsid/ms2_make_cal_pars.py --jobs $MS2_WORK/pars_a3.8/jobs.txt --calib $MS2_WORK/diag/ms2_pixel_calibration.csv --model $MS2_DATA/models/2MS2-assembly1.cif --work $MS2_WORK
```

### 4.5 Inverted contrast
`ms2_ctf_scan.py` scores both signs. With normal contrast the template "as built" wins when the CTF
is applied, and the "flipped" template wins without CTF (the particle is a dark disc).

---

## 5. GPU memory

### 5.1 Out of memory on large images
3456 × 3456 tiles fail at batch size 8 on a 24 GB card; use `BATCH=4`. `run_queue.sh` retries once
at half the batch size.

### 5.2 Out of memory with many defocus planes
- **Cause:** memory grows with the number of planes; 17 or more planes on 3072² images exceed 24 GB
  even at batch size 2.
- **Fix:** at most 11 planes, 50 nm apart (refinement covers ±50 nm around each particle), and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### 5.3 Refinement on the CPU
Possible but slow with a 512³ template box (about 2 minutes per candidate). Run it on the GPU when
no search is using it.

---

## 6. Scores, thresholds and false positives

### 6.1 Search score and final score
- **Search score** (`search_mip_scaled.mrc`): the flat-fielded z-score of the global grid.
- **Final score** (`score_final` in `particles.csv`): after local refinement of orientation, defocus
  and position. Detection cutoffs apply to the final score.

### 6.2 Real particles score 6–7 in the search and are never refined
- **Cause:** only candidates with search score ≥ `optThr` (default 7) are refined. On a coarse grid
  (3.8°) a particle whose orientation falls between grid points scores lower.
- **Fix without a new search:** `refine_more.py` repeats clustering, refinement and optimisation from
  the saved maps (all values above `arbThr` 6 are kept) with a lower `optThr`.
- **Cost:** at 6.5 about 5× more candidates than at 7, and correspondingly more noise reaching
  refinement (6.3).
- **Example** (EMPIAR-11063 tile 463, mouse large-subunit template, cutoff 8.0): 49 → 62 of the 84
  deposited ribosomes, with the same 7–8 detections outside the deposited list.

```
python examples/decolace_tric/refine_more.py --dest $WORK --target TRiC --tiles all --count-only
python -u examples/decolace_tric/refine_more.py --dest $WORK --target TRiC --tiles 00453 00463 --opt-thr 6.5 --device cuda
```

### 6.3 More false positives than the cutoff predicts
- **Cause:** the default cutoff (one false positive per image, about 7.4 here) is derived from
  search-stage statistics. Refinement then searches locally and raises noise peaks as well; tail fits
  of the search map do not include this.
- **Fix:** run the same search on phase-randomised copies of real images (same power spectrum, no
  particles) and set the final cutoff where these give at most about one hit per image. Report counts
  at both cutoffs.
- **Example** (TRiC on EMPIAR-11063): 3–4 noise hits per tile at 7.42, at most 1 at 7.8 with
  `optThr` 7. The cutoff has to be measured again after changing `optThr`.

```
python examples/decolace_tric/make_controls.py --dest $WORK --phaserand 00463 00453
python examples/decolace_tric/tric_report.py --dest $WORK --strict 7.8
```

---

## 7. False positives from image artifacts

### 7.1 Many detections along a line or edge
- **Cause:** strong low-resolution features (protective platinum layer, lamella or ice edges)
  correlate with the template along their whole length; barrel-shaped templates align with straight
  edges.
- **Fix:** mask artifact zones and count detections there separately. To confirm, fill the zone with
  noise and search again (EMPIAR-11063 tile 520: 200 TRiC detections → 1).

```
python examples/decolace_tric/make_controls.py --dest $WORK --filled 00520
```

---

## 8. Controls

| Control | Measures | Note |
|---|---|---|
| Phase-randomised image, same template and settings | false positives from noise after refinement | sets the final cutoff (6.3) |
| Decoy protein, same images and settings | detections produced by any object of similar mass | must be absent from the organism |
| Known particle vs a deposited list | sensitivity of the pipeline | template from the same species |
| Artifact zone filled with noise | whether edge detections are artifacts | 7.1 |

- **A mirror-image template is not a decoy.** Projections do not carry handedness: the mirrored volume
  produces the same projections at a reflected orientation and finds the same particles. pysmap
  rejects `decoy_template mirror`.
- **Decoy for a ~1 MDa target:** lumazine synthase (PDB 1HQK, 60-mer, about 1.0 MDa), absent from
  mammals. Apoferritin is unsuitable for mammalian cells (endogenous ferritin, and about half the mass).
- **Symmetric decoys under-count noise:** a 60-fold symmetric template has about 60× fewer distinct
  projections, hence fewer chances to match noise. Use phase-randomised images for the noise rate.
- **Same species:** for mouse cells use a mouse template (large ribosomal subunit: 6SWA without Ebp1,
  as in the deposited search), not the human 6EK0.
- **Same settings for everything compared:** target, decoy, noise images and alternative states share
  images, angular grid, `optThr` and cutoffs.

```
python examples/decolace_tric/make_lsu_mouse.py --dest $WORK --model $WORK/models/6swa.cif
python examples/decolace_tric/lsu_gap.py --dest $WORK --results results_LSU_mouse_opt6.5
```

---

## 9. Angular sampling

| Grid | Orientations | Relative cost |
|---|---|---|
| 3.8° | 288,990 | 1× |
| 2.5° out-of-plane / 1.5° in-plane (cisTEM default) | ~1.7 M | ~6× |
| 1.88° (typical SMAP full search) | ~2.4 M | ~8× |

Cost scales with (1 / step)³. A coarse grid followed by refinement of more candidates (6.2) recovers
much of the sensitivity of a finer grid at a fraction of the cost. When comparing with a deposited
cisTEM search, take its settings from the database (`TEMPLATE_MATCH_LIST`), which records what was run.

---

## 10. Symmetric templates

- `symmetryFile` (operators from the mmCIF `pdbx_struct_oper_list`) restricts the grid to one
  asymmetric unit; for icosahedral symmetry this is 60× fewer orientations.
- Large particles need an accurate pixel size (4.4) and a finer angular step (2.5° for MS2).

```
python -m smap.symmetry MODEL.cif --out ops.txt --assembly MODEL-assembly1.cif
```

---

## 11. Folder layout

| Content | Location |
|---|---|
| All results in one page | `$WORK/report/index.html` (`tric_summary.py`) |
| State of every search | `python examples/decolace_tric/status.py` |
| TRiC tables and figure | `$WORK/report/tric_report*.png`, `.csv` |
| Per-tile settings | `$WORK/pars_<search>/<tile>.par` |
| Search results | `$WORK/results_<search>/<tile>/` (`_opt6.5`: re-refined) |
| MS2 | `$MS2_WORK/` (`diag/`, `pars_cal/`, `results_cal/`) |
| Helper scripts | `examples/decolace_tric/`, `examples/ms2_capsid/` |
