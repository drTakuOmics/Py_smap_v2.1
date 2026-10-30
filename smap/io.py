"""File I/O: parameter files, MRC volumes, rotation files, binary .dat outputs.

Array convention used throughout this package: Python axis k == MATLAB dimension k+1,
so an image read here has the same [row, col] indexing as ``smap.mr`` in MATLAB and a
volume has MATLAB's [row, col, page] layout (NOT mrcfile's native [z, y, x]).
Rotation stacks are stored as (n, 3, 3), i.e. MATLAB ``R(:,:,i)`` == ``R[i-1]``.
"""

from __future__ import annotations

import re
from pathlib import Path

import mrcfile
import numpy as np

# --------------------------------------------------------------------------- params

# Defaults copied from @smap/readParamsFile.m ('search_global'); [] -> None.
SEARCH_GLOBAL_DEFAULTS = dict(
    outputDir=None, listFormat="dat", imageFile=None, modelFile=None, maskFile=None,
    aPerPix=1.0, defocus=[70.0, 70.0, 0.0], MTF=[0, 0.935, 0, 0, 0.64], arbThr=5.0,
    highThr=None, nCores=1, units=None, rotationsFile=None, F_abs=0, psdFilterFlag=1,
    sectors=None, Cs=0.0027, Cc=0.0027, V_acc=300000.0, deltaE=0.7, a_i=50e-6,
    binRange=20, nBins=1024, aPerPix_search=None, T_sample=100, df_inc=50, qThr=10,
    dThr=10, optimizeFlag=1, optThr=7, maskCrossFlag=1, range_degrees=2.0,
    inc_degrees=0.5, debugFlag=0, PDBFile=None, MW_target=None, bArb=0, chains=None,
    modelName=None, edge_max=512, defocus_format="CTFFind", angle_inc=None,
    psi_inc=None, structureFile=None, mask_params=[0, 3], keep_scratch_flag=0,
    margin_pix=0,
    # ---- Python-port options (ignored by MATLAB v2.1; defaults reproduce v2.1) ----
    oversample=1, dedupe_orientations=0, template_bfactor=0, fix_even_grid=0,
    fix_pdb_short_lines=0, legacy_particle_diameter=1, fix_cluster_thresholds=0,
    fix_optimize_angle=0, refine_local_peak=0,
)
PATH_FIELDS = ("outputDir", "imageFile", "modelFile", "maskFile", "rotationsFile",
               "PDBFile", "structureFile")
_SEARCH_GLOBAL_STRING_FIELDS = {
    "outputDir", "imageFile", "modelFile", "maskFile", "rotationsFile", "listFormat",
    "PDBFile", "chains", "modelName", "defocus_format", "structureFile",
}

_NUM_TOKEN = re.compile(r"[,;\s]+")


def _str2num(text: str):
    """Subset of MATLAB ``str2num`` sufficient for .par values.

    Returns a float, a list of floats, or None when MATLAB would return [].
    """
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    toks = [t for t in _NUM_TOKEN.split(s) if t]
    if not toks:
        return None
    vals = []
    for t in toks:
        tl = t.lower()
        if tl == "true":
            vals.append(1.0)
        elif tl == "false":
            vals.append(0.0)
        elif tl == "pi":
            vals.append(float(np.pi))
        else:
            try:
                vals.append(float(t))
            except ValueError:
                return None
    return vals[0] if len(vals) == 1 else vals


def _strtrim(s: str) -> str:
    return s.strip(" \t\r\n\v\f\x00")


def read_params_file(fn: str | Path, fn_type: str | None = None):
    """Port of @smap/readParamsFile.m.

    Kept MATLAB quirks: any line containing '#' is ignored entirely; a field matches
    when its name appears *anywhere* in a line (regexp split), and the value is the
    text between the first and second occurrence of the field name.
    """
    lines_in = []
    with open(fn, newline="") as fh:
        for raw in fh:
            line = raw[:-1] if raw.endswith("\n") else raw  # fgetl drops only '\n'
            if "#" not in line and line != "":
                lines_in.append(line)
    if fn_type is None:
        fxn = next(l for l in lines_in if "function" in l)
        fn_type = fxn.split("function ", 1)[1]
        fn_type = _strtrim(fn_type)

    if fn_type != "search_global":
        raise NotImplementedError(f"read_params_file: '{fn_type}' not ported yet")

    params = {k: (list(v) if isinstance(v, list) else v) for k, v in SEARCH_GLOBAL_DEFAULTS.items()}
    fields = list(SEARCH_GLOBAL_DEFAULTS.keys())
    for line in lines_in:
        for f in fields:
            parts = re.split(f, line)
            if len(parts) > 1:
                zz = _strtrim(parts[1])
                if f in _SEARCH_GLOBAL_STRING_FIELDS:
                    params[f] = zz
                else:
                    num = _str2num(zz)
                    if num is not None:
                        params[f] = num
    return params, fn_type


def zp(num, n: int, pad_char: str = "0") -> str:
    s = num if isinstance(num, str) else str(int(num)) if float(num).is_integer() else str(num)
    return s.rjust(n, pad_char)


# --------------------------------------------------------------------------- MRC

def mr(filename: str | Path) -> np.ndarray:
    """Port of smap.mr: returns float32 with MATLAB [row, col(, page)] layout."""
    with mrcfile.open(filename, permissive=True) as m:
        data = np.asarray(m.data, dtype=np.float32)
    if data.ndim == 3:
        if data.shape[0] == 1:
            return np.ascontiguousarray(data[0])
        return np.ascontiguousarray(np.transpose(data, (1, 2, 0)))
    return np.ascontiguousarray(data)


def mw(arr: np.ndarray, filename: str | Path, rez: float) -> None:
    """Port of smap.mw: writes float32, undoing the layout change of :func:`mr`."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        a = np.transpose(a, (2, 0, 1))
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(filename, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(a))
        m.voxel_size = float(rez)


# --------------------------------------------------------------------------- rotations

def read_rotations_file(fn: str | Path) -> np.ndarray:
    """Port of smap.readRotationsFile -> (n, 3, 3); each 3 lines are the rows of R."""
    vals = np.array(Path(fn).read_text().split(), dtype=np.float64)
    vals = vals.reshape(-1, 4)[:, 1:]
    return vals.reshape(-1, 3, 3).copy()


def write_rotations_file(R: np.ndarray, fn: str | Path) -> None:
    """Port of smap.writeRotationsFile (single precision, CRLF line endings)."""
    R = np.asarray(R, dtype=np.float32)
    with open(fn, "w", newline="") as fh:
        for i, Ri in enumerate(R, start=1):
            for row in Ri:
                fh.write("%7i\t%5.4f\t%5.4f\t%5.4f\r\n" % (i, row[0], row[1], row[2]))


# --------------------------------------------------------------------------- .dat

def write_single(fn: str | Path, values) -> None:
    np.asarray(values, dtype="<f4").ravel().tofile(fn)


def write_double(fn: str | Path, values) -> None:
    np.asarray(values, dtype="<f8").ravel().tofile(fn)


def read_single(fn: str | Path) -> np.ndarray:
    return np.fromfile(fn, dtype="<f4")


def read_double(fn: str | Path) -> np.ndarray:
    return np.fromfile(fn, dtype="<f8")


# --------------------------------------------------------------------------- layout adapters

def smap_to_zyx(vol: np.ndarray) -> np.ndarray:
    """SMAP/MATLAB layout [row=y, col=x, page=z] -> mrcfile/NumPy-native [z, y, x]."""
    v = np.asarray(vol)
    return v if v.ndim == 2 else np.transpose(v, (2, 0, 1))


def zyx_to_smap(vol: np.ndarray) -> np.ndarray:
    """mrcfile/NumPy-native [z, y, x] -> SMAP/MATLAB layout [row=y, col=x, page=z]."""
    v = np.asarray(vol)
    return v if v.ndim == 2 else np.transpose(v, (1, 2, 0))
