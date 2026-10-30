# Tests

| File | What it checks |
|---|---|
| `test_parity_octave.py` | setup helpers, CTF, grid, `.par` reader, the main CC loop, scattering potential vs the original MATLAB code run in Octave (`refs/refs.mat`) |
| `test_parity_refine.py` | clustering, patch cropping, phase shifts, refinement scoring, exit-wave template and optimisation objective vs Octave (`refs/refs_refine.mat`) |
| `test_end_to_end.py` | synthetic particle recovered by search → refinement → optimisation; outputs parse like the RNF213 readers |
| `test_options.py` | each opt-in improvement, survival counts, calibration, layout adapters, incomplete-run flag |
| `test_compare_cli.py` | CLI with Docker-style paths, batch 1 vs 8 equivalence via `smap.compare`, `run.log`, incomplete runs |

`synthetic.py` holds the shared synthetic volume/image generator.

## Regenerating the Octave references

Needs Octave >= 8 with the `image` package and a clone of the upstream
https://github.com/jpr-smap/smap_v2.1. The `.octave` scripts define the few missing MATLAB
functions themselves (no `.m` files are kept in this repository).

```bash
cd /path/to/smap_v2.1/src/smap_tools
octave --no-gui -q --eval "addpath(pwd); out_dir='/path/to/Py_smap_v2.1/tests/refs'; \
  par_file='/path/to/smap_v2.1/sample_search.par'; cif_file='/path/to/smap_v2.1/model/6ek0_LSU.cif'; \
  source('/path/to/Py_smap_v2.1/tests/octave/make_reference.octave'); \
  source('/path/to/Py_smap_v2.1/tests/octave/make_reference_refine.octave')"
```
