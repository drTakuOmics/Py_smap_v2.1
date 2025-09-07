import numpy as np
from typing import Sequence

from .mr import mr
from .crop_pad import cutj
from .fft import ftj
from .rrj import rrj
from .radial import radialmeanIm


def estimate_detector(file_paths: Sequence[str], frame_index: int = 0) -> np.ndarray:
    """Estimate detector whitening filter from micrograph cross spectra.

    Parameters
    ----------
    file_paths : sequence of str
        Paths to MRC files containing individual movie frames or images.
    frame_index : int, optional
        Zero-based index of the slice to read from each stack.  Defaults to 0.

    Returns
    -------
    numpy.ndarray
        Radially averaged inverse cross-spectrum suitable for whitening.
    """
    file_paths = list(file_paths)
    if not file_paths:
        raise ValueError("file_paths must be non-empty")

    # Read first frame to establish edge size
    first, _ = mr(file_paths[0], start_slice=frame_index + 1, num_slices=1)
    sample = first[:, :, 0]
    edge_size = int(min(sample.shape))
    center = edge_size // 2

    # Load and square-crop all frames
    frames = []
    for fn in file_paths:
        data, _ = mr(fn, start_slice=frame_index + 1, num_slices=1)
        frames.append(cutj(data[:, :, 0], (edge_size, edge_size)).astype(float))
    frames = np.asarray(frames)
    n_frames = frames.shape[0]

    # Accumulate cross-spectral density across frame pairs
    im_F_sum = np.zeros((edge_size, edge_size), dtype=np.float64)
    for i in range(n_frames):
        a_F = ftj(frames[i])
        a_F[center, center] = 0
        for j in range(i + 1, n_frames):
            b_F = ftj(frames[j])
            b_F[center, center] = 0
            im_F_sum += np.sqrt(np.abs(a_F * np.conj(b_F)))

    # Build masks mimicking MATLAB implementation
    mask_floor = rrj(np.ones((edge_size, edge_size))) * edge_size
    ring = np.abs(mask_floor - (edge_size / 2.0)) < 25
    mask_floor = np.full((edge_size, edge_size), np.nan)
    mask_floor[ring] = 1.0

    mask_cross = np.ones((edge_size, edge_size))
    mask_cross[center, :] = np.nan
    mask_cross[:, center] = np.nan

    mask_center = rrj(np.ones((edge_size, edge_size))) * edge_size
    mask_center[mask_center <= 3.5] = np.nan
    mask_center[mask_center > 3.5] = 1.0
    mask = mask_center * mask_cross

    csd = im_F_sum
    csd_masked = csd * mask
    csd_floor = np.nanmean(csd_masked * mask_floor)
    csd_masked = np.where(np.isnan(csd_masked), csd_floor, csd_masked)
    csd_masked /= csd_floor

    profile = radialmeanIm(csd_masked)
    inv_profile = np.reciprocal(profile, where=profile != 0)
    inv_profile[~np.isfinite(inv_profile)] = 1.0
    return inv_profile
