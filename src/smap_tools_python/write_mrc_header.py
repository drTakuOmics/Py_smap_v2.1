import numpy as np


def write_mrc_header(map_array, voxel_size, filename, n_images=None):
    """Write an MRC header and return an open file handle.

    Parameters
    ----------
    map_array : array-like
        Sample data whose shape and statistics populate the header.
    voxel_size : float
        Voxel size in ångström.
    filename : str
        Output file path.
    n_images : int, optional
        Number of sections expected in the file. Defaults to the third
        dimension of ``map_array``.

    Returns
    -------
    file object
        Handle positioned after the 1024-byte header ready for sequential
        writes.
    """

    arr = np.asarray(map_array, dtype=np.float32)
    sizes = list(arr.shape)
    if len(sizes) < 3:
        sizes += [1] * (3 - len(sizes))
    if n_images is not None:
        sizes[2] = int(n_images)

    hdr = np.zeros(256, dtype=np.int32)
    hdr[0:3] = sizes  # dimensions
    hdr[3] = 2  # mode 2 = float32
    hdr[7:10] = sizes  # number of intervals

    def _flt(val):
        return np.asarray(val, dtype=np.float32).view(np.int32)

    hdr[10:13] = _flt(np.array(sizes, dtype=np.float32) * float(voxel_size))
    hdr[13:16] = _flt([90.0, 90.0, 90.0])
    hdr[16:19] = [1, 2, 3]
    stats = [arr.min(), arr.max(), arr.mean()]
    hdr[19:22] = _flt(stats)
    hdr[22] = 0
    hdr[52] = int.from_bytes(b"MAP ", "little")
    hdr[53] = int.from_bytes(bytes([68, 65, 0, 0]), "little")
    hdr[54] = _flt(arr.std())

    handle = open(filename, "wb")
    hdr.tofile(handle)
    return handle


__all__ = ["write_mrc_header"]
