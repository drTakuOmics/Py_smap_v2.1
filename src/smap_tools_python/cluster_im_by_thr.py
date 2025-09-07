import numpy as np
from scipy import ndimage


def cluster_im_by_thr(image, index_image, threshold, rotations):
    """Cluster connected pixels above a threshold and retrieve rotations.

    Parameters
    ----------
    image : ndarray
        2-D array of scores.
    index_image : ndarray
        Integer array mapping each pixel to an index in ``rotations``.
    threshold : float
        Minimum score to include a pixel in a cluster.
    rotations : sequence
        Sequence of rotation objects. Elements are selected via
        ``index_image``.

    Returns
    -------
    ss : list of dict
        Cluster information with ``Area``, ``PixelList``, ``MaxVal`` and ``xy``.
    q_best : list
        Rotations corresponding to each cluster's maximum.
    order : ndarray
        Indices that sort clusters by descending ``MaxVal``.
    xy : ndarray
        ``(N, 2)`` array of peak coordinates for each cluster.
    """
    bw = image > threshold
    labeled, num = ndimage.label(bw)
    if num == 0:
        return [], [], np.array([], dtype=int), np.empty((0, 2), dtype=int)

    objects = ndimage.find_objects(labeled)
    clusters = []
    q_best = []
    xy = []
    peak_vals = []

    for lbl, sl in enumerate(objects, start=1):
        region_mask = labeled[sl] == lbl
        values = image[sl][region_mask]
        if values.size == 0:
            continue
        coords = np.argwhere(region_mask)
        coords[:, 0] += sl[0].start
        coords[:, 1] += sl[1].start
        max_idx = np.argmax(values)
        max_val = values[max_idx]
        max_coord = coords[max_idx]
        clusters.append(
            {
                "Area": int(values.size),
                "PixelList": coords,
                "MaxVal": float(max_val),
                "xy": tuple(max_coord),
            }
        )
        idx_rot = int(index_image[tuple(max_coord)])
        q_best.append(rotations[idx_rot])
        xy.append(max_coord)
        peak_vals.append(max_val)

    order = np.argsort(peak_vals)[::-1]
    clusters = [clusters[i] for i in order]
    q_best = [q_best[i] for i in order]
    xy = np.array([xy[i] for i in order], dtype=int)
    return clusters, q_best, order, xy
