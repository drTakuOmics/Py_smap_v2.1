"""Array helpers ported one-to-one from SMAP v2.1 @smap/*.m.

Each function names its MATLAB source. Indices in docstrings are 1-based when they
refer to MATLAB code. Known MATLAB quirks are reproduced on purpose and marked
``# QUIRK`` so they can be revisited once parity with v2.1 is confirmed.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.fft as sfft
from scipy import ndimage

EPS = np.finfo(np.float64).eps


# --------------------------------------------------------------------------- basics

def matlab_colon(a: float, step: float, b: float) -> np.ndarray:
    """MATLAB ``a:step:b`` (element count uses a small tolerance like MATLAB)."""
    q = (b - a) / step
    n = int(math.floor(q + 1e-10 * max(1.0, abs(q))))
    if n < 0:
        return np.zeros(0)
    return a + step * np.arange(n + 1, dtype=np.float64)


def matlab_linspace(a: float, b: float, n: float) -> np.ndarray:
    """MATLAB linspace: non-integer n is floored."""
    return np.linspace(a, b, int(math.floor(n)))


def matlab_mode(x: np.ndarray):
    """MATLAB ``mode(x(:))``: most frequent value, smallest one on ties."""
    v = np.asarray(x).ravel()
    v = v[~np.isnan(v)] if np.issubdtype(v.dtype, np.floating) else v
    u, c = np.unique(v, return_counts=True)
    return u[np.argmax(c)]


def zp_int(i: int, n: int = 4) -> str:
    return str(int(i)).rjust(n, "0")


def nm(x: np.ndarray) -> np.ndarray:
    """@smap/nm.m: zero mean, unit SD (n-1), NaN-aware."""
    x = np.asarray(x)
    v = x.ravel().astype(np.float64)
    nans = np.isnan(v)
    n = np.sum(~nans)
    m = np.sum(np.where(nans, 0.0, v)) / n
    denom = max(n - 1, 1)
    s = math.sqrt(np.sum(np.abs(v[~nans] - m) ** 2) / denom)
    return ((x - m) / s).astype(np.result_type(x.dtype, np.float32))


def normalize_rm(R: np.ndarray) -> np.ndarray:
    """@smap/normalizeRM.m: nearest orthogonal matrix (U*V') for (n,3,3) or (3,3)."""
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    Rb = R[None] if single else R
    U, _, Vh = np.linalg.svd(Rb)
    out = U @ Vh
    return out[0] if single else out


def assign_jobs(n_indices: int, n_servers: int, server_id: int) -> np.ndarray:
    """@smap/assignJobs.m -> 1-based contiguous index block for ``server_id``."""
    base = math.ceil(n_indices / n_servers)
    ends = np.cumsum(np.full(n_servers, base))
    starts = ends - base + 1
    ends[-1] = min(ends[-1], n_indices)
    s, e = starts[server_id - 1], ends[server_id - 1]
    return np.arange(s, e + 1)


# --------------------------------------------------------------------------- Fourier

def ftj(x: np.ndarray) -> np.ndarray:
    """@smap/ftj.m: centred, unitary-scaled forward FFT (keeps single precision)."""
    out = np.fft.fftshift(sfft.fftn(np.fft.ifftshift(x), workers=-1))
    out /= np.asarray(math.sqrt(x.size), dtype=out.real.dtype)
    return out


def iftj(x: np.ndarray) -> np.ndarray:
    """@smap/iftj.m: centred inverse FFT, real part, NaN -> 0."""
    if np.isnan(x).any():
        x = np.where(np.isnan(x), 0, x)
    y = sfft.ifftn(np.fft.ifftshift(x), workers=-1, overwrite_x=True)
    out = np.fft.fftshift(y.real)
    del y
    out *= np.asarray(math.sqrt(x.size), dtype=out.dtype)
    return out


def rrj(inref) -> np.ndarray:
    """@smap/rrj.m: normalised radius map (0.5 at the edge midpoint)."""
    inref = np.asarray(inref)
    dims_orig = inref.shape
    dim_max = max(inref.shape)
    d3 = inref.shape[2] if inref.ndim > 2 else 1
    cp = dim_max // 2 + 1
    x = np.arange(1, dim_max + 1) - cp
    X = np.broadcast_to(x[None, :, None], (dim_max, dim_max, d3)).astype(np.float64)
    if d3 > 1:
        Y = np.transpose(X, (1, 0, 2))
        cp_z = d3 // 2 + 1
        Z = np.broadcast_to((cp_z - np.arange(1, d3 + 1))[None, None, :], X.shape)
        R = np.sqrt(X**2 + Y**2 + Z**2)
        out = R / (2.0 * R[0, cp - 1, cp - 1])
    else:
        X2 = X[:, :, 0]
        Y2 = X2.T
        R = np.sqrt(X2**2 + Y2**2)
        out = R / (2.0 * R[cp - 1, 0])
    if out.shape != tuple(dims_orig):
        out = crop_or_pad(out, dims_orig)
    return out


def get_ks(im_or_n, a_per_pix: float):
    """@smap/getKs.m -> (k_2d [1/A], centre pixel (1-based))."""
    if np.isscalar(im_or_n):
        im = np.zeros((int(im_or_n), int(im_or_n)), dtype=np.float32)
    else:
        im = np.asarray(im_or_n)
    n = im.shape[0]
    cp = n // 2 + 1
    return rrj(im) / a_per_pix, cp


# --------------------------------------------------------------------------- crop/pad

def crop_or_pad(inref: np.ndarray, new_dims, pad_val: float = 0.0) -> np.ndarray:
    """@smap/cropOrPad.m (centre-preserving crop or pad; no targetXY option).

    Pad output is single precision like MATLAB (``single(ones(..)*padVal)``).
    """
    a = np.asarray(inref)
    nd = a.ndim
    old = np.array(a.shape[:nd])
    new = np.array([int(round(v)) for v in list(new_dims)[:nd]])
    if np.all(old == new):
        return a
    dimL = new > old
    if np.min(new - old) < 0:  # crop (dims that grow keep their full extent)
        cp_old = old // 2 + 1
        sl = []
        for i in range(nd):
            if not dimL[i]:
                e0 = math.ceil(cp_old[i] - new[i] / 2)
                e1 = math.ceil(cp_old[i] + new[i] / 2 - 1)
                sl.append(slice(e0 - 1, e1))
            else:
                sl.append(slice(0, old[i]))
        return a[tuple(sl)]
    # pad
    out_dtype = np.complex64 if np.iscomplexobj(a) else np.float32
    out = np.full(tuple(new), pad_val, dtype=out_dtype)
    cp_old = old // 2 + 1
    cp_new = new // 2 + 1
    off = cp_new - cp_old  # 0-based offset of the old block
    sl = tuple(slice(off[i], off[i] + old[i]) for i in range(nd))
    out[sl] = a
    return out


def _is_3_smooth(n: int) -> bool:
    if n < 1:
        return False
    for p in (2, 3):
        while n % p == 0:
            n //= p
    return n == 1


def resize_for_fft(inref: np.ndarray, mode: str = "crop", pad_val: float = 0.0) -> np.ndarray:
    """@smap/resizeForFFT.m: nearest 2^a*3^b size per dim, then the max over dims
    (=> square/cubic output). Mean is removed before and restored after."""
    inc = 1 if mode == "pad" else -1
    dims_new = []
    for d in inref.shape:
        while not _is_3_smooth(d):
            d += inc
        dims_new.append(d)
    dims_new = [max(dims_new)] * inref.ndim
    mean_val = np.nanmean(inref)
    x = inref - mean_val
    if mode == "pad":
        out = crop_or_pad(x, dims_new, pad_val)
    else:
        out = crop_or_pad(x, dims_new)
    return out + mean_val


def resize_f(inref: np.ndarray, sf: float, method: str = "newSize") -> np.ndarray:
    """@smap/resize_F.m (Fourier crop/pad resampling). Only 'newSize' is ported.

    QUIRK: for sf<1 the volume mode (background) is removed and not added back.
    """
    if method != "newSize":
        raise NotImplementedError("resize_f: only method='newSize' is ported")
    a = np.asarray(inref)
    if a.dtype not in (np.float32, np.float64):
        a = a.astype(np.float64)
    os_ = np.array(a.shape)
    final = np.floor(os_ * sf).astype(int)
    bg = matlab_mode(a)
    if sf > 1:
        F = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a)))
        if a.ndim == 2:
            Fp = crop_or_pad(F, [final[0], final[-1]], 0)
            out = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(Fp))))
            return out * sf**2
        fs = np.floor(os_[:2] * sf).astype(int)
        Fp = crop_or_pad(F, [fs[0], fs[1], fs[0]], 0)
        out = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(Fp))))
        return out * (1.0 / sf) ** 3
    F = sfft.fftn(np.fft.ifftshift(a - a.dtype.type(bg)), workers=-1, overwrite_x=True)
    F = np.fft.fftshift(F)
    if a.ndim == 2:
        Fc = crop_or_pad(F, [final[0], final[-1]]).astype(np.complex128)
        out = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(Fc))))
        return out * sf**2
    Fc = crop_or_pad(F, final).astype(np.complex128)  # MATLAB: double(cropOrPad(..))
    del F
    out = np.real(np.fft.fftshift(sfft.ifftn(np.fft.ifftshift(Fc), workers=-1)))
    return out * sf**3


# --------------------------------------------------------------------------- binning

def histc_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """MATLAB ``[~,bin]=histc(x,edges)``: 1-based bin, 0 when outside [e1, e_end]."""
    edges = np.asarray(edges, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    k = np.searchsorted(edges, x, side="right")
    k[(x > edges[-1]) | (x < edges[0]) | np.isnan(x)] = 0
    return k


def bindata_local(y, x, xrg):
    """Local ``bindata`` inside @smap/radialmeanj.m (empty bins kept as NaN)."""
    whichedge = histc_bins(x, xrg)
    bins = np.clip(whichedge, 1, len(xrg) - 1)  # QUIRK: out-of-range -> bin 1
    nb = bins.max()
    ns = np.bincount(bins - 1, minlength=nb).astype(np.float64)
    ysum = np.bincount(bins - 1, weights=np.asarray(y, dtype=np.float64), minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        ym = ysum / ns
    return ym, ym[bins - 1]


def bindata_smap(y, x, xrg):
    """@smap/bindata.m (1-D): only non-empty bins are returned."""
    ym, yb = bindata_local(y, x, xrg)
    nz = ~np.isnan(ym)
    return ym[nz], yb


def radialmean_2d(imref: np.ndarray, opt: str | None = None):
    """@smap/radialmeanj.m, 2-D branch -> (radial profile, 2-D radial-mean map)."""
    im = np.asarray(imref, dtype=np.float64)
    npix = np.array(im.shape)
    cp = npix // 2 + 1
    n_long = int(npix.max())
    r_coord = rrj(np.ones(im.shape)) * n_long
    odd = im.shape[0] % 2
    if odd:
        if opt == "circ":
            r_bins = np.arange(0, max(cp) + 1) / (n_long - 1)
        else:
            r_bins = matlab_linspace(0, math.sqrt(2) / 2, n_long * math.sqrt(2) / 2 + 1)
    else:
        if opt == "circ":
            r_bins = matlab_linspace(0, 0.5, n_long / 2 + 1)
        else:
            r_bins = matlab_linspace(0, math.sqrt(2) / 2, (n_long + 1) * math.sqrt(2) / 2 + 1)[:-1]
    r_bins = r_bins * n_long
    cp_val = im[cp[0] - 1, cp[1] - 1]
    ym, yb = bindata_local(im.ravel(), r_coord.ravel(), r_bins)
    ym = ym.copy()
    ym[0] = cp_val
    return ym, yb.reshape(im.shape)


def particle_diameter(volref: np.ndarray, thresh: float = 0.005) -> int:
    """@smap/particleDiameter.m (uses the class-level bindata)."""
    v = np.asarray(volref, dtype=np.float64)
    n = max(v.shape)
    r_coord = rrj(v) * n
    if v.shape[0] % 2:
        r_bins = matlab_linspace(0, math.sqrt(2) / 2, v.shape[0] * math.sqrt(2) / 2 + 1)
    else:
        r_bins = matlab_linspace(0, math.sqrt(2) / 2, (v.shape[0] + 1) * math.sqrt(2) / 2 + 1)[:-1]
    r_bins = r_bins * n
    binned, _ = bindata_smap(v.ravel(), r_coord.ravel(), r_bins)
    binned = binned - np.median(binned)
    binned = binned / np.max(binned)
    idx = np.nonzero(binned > thresh)[0]
    return int((idx[-1] + 1) * 2)


# --------------------------------------------------------------------------- filters

def mask_central_cross(imref: np.ndarray) -> np.ndarray:
    """@smap/mask_central_cross.m: zero the central row/column of the spectrum."""
    n = imref.shape[0]
    cp = n // 2 + 1
    real_in = not np.iscomplexobj(imref)
    F = ftj(imref) if real_in else imref.copy()
    dc = np.abs(F[cp - 1, cp - 1])
    F[cp - 1, :] = 0
    F[:, cp - 1] = 0
    F[cp - 1, cp - 1] = dc
    return iftj(F) if real_in else F


def psd_filter(nf_im: np.ndarray, method: str = "sqrt"):
    """@smap/psdFilter.m without sectors -> (filter [centred], whitened image).

    (The unused getPSD call of the MATLAB code is skipped: its result is discarded.)
    """
    x = np.asarray(nf_im, dtype=np.float64)
    x = x - x.mean()
    n_orig = x.shape
    x = crop_or_pad(x, [max(n_orig)] * 2, 0).astype(np.float64)
    npix = np.array(x.shape)
    cp = npix // 2 + 1
    f_amp = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x)))) / math.sqrt(x.size)
    _, f_amp_r = radialmean_2d(f_amp)
    f_amp_r[cp[0] - 1, cp[1] - 1] = 1.0
    f_inv = 1.0 / f_amp_r
    f_inv[cp[0] - 1, cp[1] - 1] = 0.0
    im_out = iftj(ftj(x) * f_inv)
    return crop_or_pad(f_inv, n_orig), crop_or_pad(im_out, n_orig)


def mask_a_volume(mapref: np.ndarray, mask_params) -> np.ndarray:
    """@smap/mask_a_volume.m, 'mask' mode (soft cosine edge around the density)."""
    m = np.asarray(mapref)
    bg = matlab_mode(m)
    mm = m - bg
    thr = np.std(np.abs(mm), ddof=1)
    bw = np.abs(mm) > thr
    D = ndimage.distance_transform_edt(~bw) if bw.any() else np.full(m.shape, np.inf)
    near, cos_w = float(mask_params[0]), float(mask_params[1])
    far = near + cos_w
    D11 = D.astype(np.float64).copy()
    D11[D <= near] = 0
    D11[D >= far] = np.pi
    between = (D > near) & (D < far)
    D11[between] = (D[between] - near) / cos_w * np.pi
    mask = np.cos(D11) / 2 + 0.5
    return mm * mask + bg


def apply_phase_shifts_3d_real(inref: np.ndarray, shifts) -> np.ndarray:
    """@smap/applyPhaseShifts.m, real-volume branch (integer roll + Fourier subpixel)."""
    shifts = np.asarray(shifts, dtype=np.float64)
    s_int = np.trunc(shifts).astype(int)
    s_sub = shifts - np.trunc(shifts)
    v = np.roll(np.asarray(inref), shift=tuple(s_int), axis=(0, 1, 2))
    if np.max(np.abs(s_sub)) > 0:
        xs, ys, zs = v.shape
        g = lambda n: (np.arange(n) - n / 2) / n  # noqa: E731
        # meshgrid(gx,gy,gz): xd varies along dim2, yd along dim1, zd along dim3.
        # dphs = yd*(-s1) + xd*(-s2) + zd*(-s3) is separable -> multiply 1-D ramps
        # in place instead of building full-size phase arrays (memory for 512^3).
        F = ftj(v)
        del v
        ct = F.dtype
        F *= np.exp(1j * 2 * np.pi * g(ys) * (-s_sub[0])).astype(ct)[:, None, None]
        F *= np.exp(1j * 2 * np.pi * g(xs) * (-s_sub[1])).astype(ct)[None, :, None]
        F *= np.exp(1j * 2 * np.pi * g(zs) * (-s_sub[2])).astype(ct)[None, None, :]
        v = iftj(F)
    return v
