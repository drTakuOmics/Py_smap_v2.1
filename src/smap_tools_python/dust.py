import numpy as np
from scipy import ndimage


def dust(volume, criteria):
    """Remove small high-valued specks ("dust") from a volume.

    Parameters
    ----------
    volume : ndarray
        Input 3-D volume.
    criteria : sequence of two numbers
        ``(threshold, min_size)`` where ``threshold`` is the number of
        standard deviations above the mean to keep and ``min_size`` is the
        minimum bounding-box edge length for a region to be retained.

    Returns
    -------
    ndarray
        Volume with small specks suppressed.
    """
    volume = np.asarray(volume, dtype=float)
    thr, dust_size = criteria
    norm = (volume - np.nanmean(volume)) / np.nanstd(volume)
    mask = norm > thr
    labeled, num = ndimage.label(mask)
    if num == 0:
        return volume
    objects = ndimage.find_objects(labeled)
    out = volume.copy()
    replace_val = volume.mean() - 2 * volume.std()
    for i, sl in enumerate(objects, start=1):
        region = (labeled[sl] == i)
        bbox = [s.stop - s.start for s in sl]
        if max(bbox) < dust_size:
            sub = out[sl]
            sub[region] = replace_val
            out[sl] = sub
    return out
