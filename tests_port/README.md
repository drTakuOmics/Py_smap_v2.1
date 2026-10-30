Regenerate `refs/refs.mat` from the original MATLAB sources (Octave >= 8 with the `image` package):

```bash
cd /path/to/smap_v2.1/src/smap_tools
octave --no-gui -q --eval "addpath(pwd); addpath('/path/to/Py_smap_v2.1/tests_port/octave/shims'); \
  out_dir='/path/to/Py_smap_v2.1/tests_port/refs'; par_file='/path/to/smap_v2.1/sample_search.par'; \
  cif_file='/path/to/smap_v2.1/model/6ek0_LSU.cif'; run('/path/to/Py_smap_v2.1/tests_port/octave/make_reference.m')"
```
