import numpy as np

from .mr import mr
from .crop_pad import extendj, cutj
from .reg2vols import reg2vols


def register_multiple_fragments(fn_array, edge_size=None, max_shift=200):
    """Register multiple volume fragments against a reference.

    Parameters
    ----------
    fn_array : sequence of str
        Filenames of volumes in MRC format. The first entry is treated as the
        reference volume and is *not* included in the returned sum.
    edge_size : int, optional
        Target cubic edge size for all volumes.  Volumes are padded or cropped
        as needed.  Defaults to the reference volume's size.
    max_shift : int, optional
        Maximum allowed registration shift in pixels.  Passed through to
        :func:`reg2vols`.

    Returns
    -------
    outref : ndarray
        Sum of the registered fragments (excluding the reference volume).
    shifts : ndarray, shape (N-1, 3)
        Estimated shifts ``(dz, dy, dx)`` applied to each fragment to align it
        with the reference.
    """

    if len(fn_array) < 2:
        raise ValueError("fn_array must contain at least a reference and one fragment")

    ref_vol, _ = mr(fn_array[0])
    if edge_size is None:
        edge_size = ref_vol.shape[0]

    # Ensure cubic volumes of the requested size
    if ref_vol.shape[0] < edge_size:
        pad_val = float(np.median(ref_vol))
        ref_vol = extendj(ref_vol, (edge_size,) * 3, pad_value=pad_val)
    elif ref_vol.shape[0] > edge_size:
        ref_vol = cutj(ref_vol, (edge_size,) * 3)

    fragments = []
    shifts = []
    for fn in fn_array[1:]:
        frag, _ = mr(fn)
        if frag.shape[0] < edge_size:
            pad_val = float(np.median(frag))
            frag = extendj(frag, (edge_size,) * 3, pad_value=pad_val)
        elif frag.shape[0] > edge_size:
            frag = cutj(frag, (edge_size,) * 3)

        reg, shift, _ = reg2vols(frag, ref_vol, max_shift)
        fragments.append(reg)
        shifts.append(shift)

    if not fragments:
        raise ValueError("No fragments provided for registration")

    outref = np.sum(fragments, axis=0)
    # Match background level to reference volume
    bg_ref = float(np.median(ref_vol))
    bg_frag = float(np.median(outref))
    outref = outref - bg_frag + bg_ref
    return outref, np.asarray(shifts)

