import numpy as np
from pathlib import Path


def write_dat(array, filename):
    """Write array to a binary ``.dat`` file with accompanying ``.hdr``.

    The data are flattened in C order and stored as 32-bit floats.  A text
    header file with the same stem and ``.hdr`` extension stores the array
    shape separated by tabs, mirroring MATLAB's ``dw`` helper.
    """
    arr = np.asarray(array, dtype=np.float32)
    path = Path(filename)
    with path.open("wb") as fh:
        arr.ravel().tofile(fh)
    hdr = path.with_suffix(".hdr")
    hdr.write_text("\t".join(str(s) for s in arr.shape) + "\n")


def read_dat_file(filename):
    """Read triples of ``double`` values from a ``.dat`` file.

    Returns three arrays ``(ai, al, av)`` corresponding to every third value in
    the file, replicating MATLAB's ``readDatFile`` utility.
    """
    data = np.fromfile(filename, dtype=np.float64)
    ai = data[0::3]
    al = data[1::3]
    av = data[2::3]
    return ai, al, av
