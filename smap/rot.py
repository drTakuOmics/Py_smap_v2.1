"""Rotation helpers: quaternion conversions (smap_tools/quaternion.m), pairwiseQD,
smap2frealign, and the refinement 'bump' set (rotationVectorToMatrix)."""

from __future__ import annotations

import math

import numpy as np

from .utils import matlab_colon, normalize_rm


def rotm_to_quat(R: np.ndarray) -> np.ndarray:
    """quaternion.rotationmatrix -> RotMat2e: [w x y z] with w >= 0. R: (3,3) or (n,3,3)."""
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    Rb = R[None] if single else R
    out = np.zeros((Rb.shape[0], 4))
    for i, M in enumerate(Rb):
        if np.all(M == 0):
            continue
        w = 0.5 * math.sqrt(max(0.0, M[0, 0] + M[1, 1] + M[2, 2] + 1))
        if w == 0:
            sg = lambda v: 1.0 if v >= 0 else -1.0  # noqa: E731  (sgn: +1 for 0)
            out[i] = [0.0,
                      math.sqrt(max(0, -0.5 * (M[1, 1] + M[2, 2]))) * sg(-M[1, 2]),
                      math.sqrt(max(0, -0.5 * (M[0, 0] + M[2, 2]))) * sg(-M[0, 2]),
                      math.sqrt(max(0, -0.5 * (M[0, 0] + M[1, 1]))) * sg(-M[0, 1])]
        else:
            out[i] = [w, 0.25 * (M[2, 1] - M[1, 2]) / w, 0.25 * (M[0, 2] - M[2, 0]) / w,
                      0.25 * (M[1, 0] - M[0, 1]) / w]
    return out[0] if single else out


def rotm_to_quat_robust(R: np.ndarray) -> np.ndarray:
    """Numerically stable R -> q (Shepperd): picks the largest of w,x,y,z to divide by.
    Used for distances/deduplication. (RotMat2e divides by w and loses accuracy for
    rotations near 180 degrees; v2.1 shares that weakness.)"""
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    Rb = R[None] if single else R
    out = np.zeros((Rb.shape[0], 4))
    for i, M in enumerate(Rb):
        tr = M[0, 0] + M[1, 1] + M[2, 2]
        c = np.array([tr, M[0, 0], M[1, 1], M[2, 2]])
        k = int(np.argmax(c))
        if k == 0:
            w = 0.5 * math.sqrt(1 + tr)
            q = [w, (M[2, 1] - M[1, 2]) / (4 * w), (M[0, 2] - M[2, 0]) / (4 * w), (M[1, 0] - M[0, 1]) / (4 * w)]
        elif k == 1:
            x = 0.5 * math.sqrt(max(0.0, 1 + M[0, 0] - M[1, 1] - M[2, 2]))
            q = [(M[2, 1] - M[1, 2]) / (4 * x), x, (M[0, 1] + M[1, 0]) / (4 * x), (M[0, 2] + M[2, 0]) / (4 * x)]
        elif k == 2:
            y = 0.5 * math.sqrt(max(0.0, 1 - M[0, 0] + M[1, 1] - M[2, 2]))
            q = [(M[0, 2] - M[2, 0]) / (4 * y), (M[0, 1] + M[1, 0]) / (4 * y), y, (M[1, 2] + M[2, 1]) / (4 * y)]
        else:
            z = 0.5 * math.sqrt(max(0.0, 1 - M[0, 0] - M[1, 1] + M[2, 2]))
            q = [(M[1, 0] - M[0, 1]) / (4 * z), (M[0, 2] + M[2, 0]) / (4 * z), (M[1, 2] + M[2, 1]) / (4 * z), z]
        q = np.array(q)
        out[i] = q / np.linalg.norm(q)
    return out[0] if single else out


def quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, [w x y z] convention (quaternion.mtimes)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=-1)


def euler_zyz(q: np.ndarray) -> np.ndarray:
    """quaternion.EulerAngles('zyz') -> (..., 3) radians, exactly as in quaternion.m."""
    e1, e2, e3, e4 = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    a1 = np.arctan2(e2*e1 + e3*e4, e3*e1 - e2*e4)
    a2 = np.arccos(np.clip(e1**2 - e2**2 - e3**2 + e4**2, -1, 1))
    a3 = np.arctan2(e3*e4 - e2*e1, e2*e4 + e3*e1)
    return np.stack([a1, a2, a3], axis=-1)


def smap2frealign(R: np.ndarray) -> np.ndarray:
    """@smap/smap2frealign.m -> (n,3) degrees [PSI THETA PHI] (fliplr of zyz angles)."""
    R = np.asarray(R, dtype=np.float64)
    q = rotm_to_quat(R[None] if R.ndim == 2 else R)
    qr = np.array([math.cos(math.pi / 2), math.sin(math.pi / 2), 0.0, 0.0])  # eulerangles('xyz',[pi 0 0])
    q = quat_mult(qr[None], q)
    ang = euler_zyz(q) * 180 / math.pi
    return ang[:, ::-1]


def pairwise_qd(R1: np.ndarray, R2: np.ndarray | None = None) -> np.ndarray:
    """@smap/pairwiseQD.m (two-argument form): rotation angle between pairs, degrees.

    d = 2*acos(|q1.q2|)*180/pi. Independent of the R<->q handedness convention.
    """
    q1 = rotm_to_quat_robust(normalize_rm(R1))
    q2 = q1 if R2 is None else rotm_to_quat_robust(normalize_rm(R2))
    dots = np.clip(np.abs(q1 @ q2.T), 0.0, 1.0)  # acos(>1) is complex in MATLAB; real() -> 0
    return 2 * 180 / math.pi * np.arccos(dots)


def rodrigues(v: np.ndarray) -> np.ndarray:
    """Standard axis-angle rotation matrix for rotation vector v (radians)."""
    th = float(np.linalg.norm(v))
    if th == 0:
        return np.eye(3)
    k = np.asarray(v) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def bump_rotations(range_degrees: float, inc_degrees: float) -> np.ndarray:
    """Local refinement set (smappoi_search_global.m lines 1052-1064) -> (n,3,3).

    MATLAB's rotationVectorToMatrix (Computer Vision Toolbox) returns the transpose of
    the Rodrigues matrix; the grid is symmetric (v and -v), so this only changes order.
    We reproduce the transpose so indices match.
    """
    base = matlab_colon(-range_degrees, inc_degrees, range_degrees)
    pts = []
    for z in base:            # meshgrid(base,base,base)(:) -> y fastest, then x, then z
        for x in base:
            for y in base:
                pts.append((x, y, z))
    pts = np.array(pts)
    pts = pts[np.sqrt(np.sum(pts**2, axis=1)) <= range_degrees]
    return np.stack([rodrigues(p * math.pi / 180).T for p in pts])
