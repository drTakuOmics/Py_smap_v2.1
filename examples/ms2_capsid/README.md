# MS2 capsid (EMPIAR-10075): icosahedral search and pixel-size calibration

Search for the bacteriophage MS2 capsid (180 coat proteins, ~2.5 MDa, ~27 nm) in motion-corrected
micrographs of EMPIAR-10075, using the point-group reduction of the angular grid (`symmetryFile`) and
a calibration of pixel size and defocus from the particles themselves.

```
export MS2_DATA=/path/to/ms2_empiar10075   # micrographs/ and models/
export MS2_WORK=/path/to/ms2_work          # .par files, results, diagnostics
cd /path/to/Py_smap_v2.1
```

Data: 4096² sums, 300 kV, Cs 0.02 mm (Cs-corrected), nominal 1.16 Å/px. No CTF is deposited; CTF is
estimated with cryoSPARC Patch CTF (pixel 1.16 Å, Cs 0.02 mm).

## 1. Template and symmetry operators

```
cd $MS2_DATA/models && wget -q https://files.rcsb.org/download/2MS2.cif https://files.rcsb.org/download/2MS2-assembly1.cif.gz && gunzip -f 2MS2-assembly1.cif.gz && cd -
python -m smap.symmetry $MS2_DATA/models/2MS2.cif --out $MS2_DATA/models/2MS2_I.txt --assembly $MS2_DATA/models/2MS2-assembly1.cif
```

Expected: 60 rotations (I), assembly check median distance ≈ 0 Å.

## 2. Per-micrograph `.par` files from cryoSPARC

```
J=/path/to/cryosparc_project/J5     # Patch CTF job folder
python tools/cryosparc_ctf_to_pars.py $J/*exposures_ctf_estimated.cs $J/*passthrough_exposures.cs --template examples/ms2_capsid/ms2_capsid.par --micrograph-dir $MS2_DATA/micrographs --out-dir $MS2_WORK/pars_a3.8 --results-dir $MS2_WORK/results_a3.8 --max-fit-A 6 --limit 10
for p in $(head -3 $MS2_WORK/pars_a3.8/jobs.txt); do python tools/ctf_check.py $p; done
```

## 3. First search and calibration

The deposited pixel size is about 4% larger than the scale of the 2MS2 model (calibrated: 1.114 Å).
For a 27 nm capsid this is enough to prevent any detection, so calibrate on the first search before
interpreting it. `ms2_ctf_scan.py` correlates each micrograph with a rotationally averaged capsid
template (no orientation search) while scanning pixel size, defocus and contrast sign.

```
bash examples/ms2_capsid/run_jobs.sh $MS2_WORK/pars_a3.8/jobs.txt --batch-size 4 --path-map /ms2=$MS2_DATA
python examples/ms2_capsid/ms2_diag.py $MS2_WORK/results_a3.8/*/ --out $MS2_WORK/diag
python examples/ms2_capsid/ms2_ctf_scan.py $MS2_WORK/results_a3.8/*/ --fine --out $MS2_WORK/diag
```

A probe search is enough for the scan (`python -m smap.search_global PAR --max-rotations 2000
--no-refine --allow-incomplete`): it needs `searchImage.mrc`, the template potential and
`search_status.json`.

## 4. Search at the calibrated pixel size

Defocus is rescaled by (new / old pixel)² (exact for Cs ≈ 0) and centred between the cryoSPARC value and
the value preferred by the particles; at most 11 planes, 50 nm apart (GPU memory grows with the number of
planes); 2.5° angular step.

```
python examples/ms2_capsid/ms2_make_cal_pars.py --jobs $MS2_WORK/pars_a3.8/jobs.txt --calib $MS2_WORK/diag/ms2_pixel_calibration.csv --model $MS2_DATA/models/2MS2-assembly1.cif --work $MS2_WORK
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash examples/ms2_capsid/run_jobs.sh $MS2_WORK/pars_cal/jobs.txt --batch-size 4
```

## Files

| File | Purpose |
|---|---|
| `ms2_capsid.par` | common search settings (template, symmetry operators, sampling) |
| `run_jobs.sh` | sequential, resumable runner |
| `ms2_diag.py` | orientation-mean, raw MIP and z maps of a finished search |
| `ms2_ctf_scan.py` | pixel-size, defocus and contrast-sign scan with a rotationally averaged template |
| `ms2_make_cal_pars.py` | `.par` files at the calibrated pixel size and defocus |
