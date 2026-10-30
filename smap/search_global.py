"""Port of smappoi_search_global.m (setup, main cross-correlation loop, combine step).

Structure follows the MATLAB file:

* :func:`prepare`     lines 51-485   params, scattering potential, image whitening,
                                     CTF stack, Fourier volume, thresholds
* :func:`run_job`     lines 486-627  per-orientation template generation + CC,
                                     moments, above-threshold list, scratch outputs
* :func:`combine`     lines 628-863  merge jobs, pixelwise (flat-field) normalisation,
                                     survival histogram, highVals.txt
* clustering + particle optimisation (lines 864-1990) are NOT ported yet.

The arithmetic follows MATLAB's precision: projections/templates in float64, the FFT
pipeline in float32, moment accumulation (sum1/sum2) in float64.
``batch_size`` > 1 processes several orientations per GPU pass; it changes scheduling
only, not the per-orientation arithmetic.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.fft as sfft
from scipy.special import erfc, erfcinv

from . import __version__
from . import io as sio
from .ctf import ctf, template_envelope
from .grid import calculate_search_grid
from .structure import calculate_sp
from .utils import (assign_jobs, crop_or_pad, mask_a_volume, mask_central_cross,
                    matlab_colon, matlab_mode, nm, normalize_rm, psd_filter, resize_f,
                    resize_for_fft, zp_int)


# ============================================================================ setup

@dataclass
class SearchSetup:
    params: dict
    out_dir: Path
    scratch_dir: Path
    R_full: np.ndarray                 # (nRot, 3, 3)
    dims_imref: tuple                  # image size before FFT padding
    Npix_im: tuple                     # padded (square) image size
    imref: np.ndarray                  # whitened, padded search image (float32)
    imref_F_half: np.ndarray           # rows 1:cp_im(1) of fftn(ifftshift(imref))/N
    fPSD_half: np.ndarray              # float32, same half layout
    CTF_half: np.ndarray               # float64 imag(CTF), (h0, N1, nDf)
    V_F: np.ndarray                    # complex128 centred 3-D FT of the potential
    Npix: int                          # template edge
    defocus: np.ndarray                # (nDf, 3)
    high_thr: float
    n_samples: float
    log_lines: list = field(default_factory=list)
    fPSD_full: np.ndarray | None = None  # full (corner-origin) whitening filter, float32
    oversample: int = 1                  # OPTION: Fourier oversampling of the potential

    @property
    def n_dfs(self) -> int:
        return self.defocus.shape[0]


def _log(setup_or_list, msg: str):
    print(msg)
    if isinstance(setup_or_list, list):
        setup_or_list.append(msg)


def apply_path_map(params: dict, path_map) -> dict:
    """Rewrite path prefixes in string parameters, e.g. Docker '/work' -> a local folder,
    so an existing v2.1 ``search.par`` can be run unchanged by the port."""
    for old, new in (path_map or []):
        for k in sio.PATH_FIELDS:
            v = params.get(k)
            if isinstance(v, str) and v.startswith(old):
                params[k] = new + v[len(old):]
    return params


def prepare(params_file: str | Path, device=None, write_files: bool = True,
            path_map=None, output_dir=None) -> SearchSetup:
    """Everything before the main loop (MATLAB lines 51-485), shared by all jobs."""
    params, fn_type = sio.read_params_file(params_file)
    if fn_type != "search_global":
        raise ValueError(f"Expected function 'search_global', got '{fn_type}'")
    params = apply_path_map(params, path_map)
    if output_dir:  # write somewhere else than the .par says (keeps v2.1 outputs intact)
        params["outputDir"] = str(output_dir)
    params["_params_file"] = str(Path(params_file).resolve())
    params["_params_text"] = Path(params_file).read_text(errors="replace")
    log: list = []

    # backwards compatibility (structureFile <- PDBFile/modelFile)
    if not params.get("structureFile"):
        if params.get("PDBFile"):
            params["structureFile"] = params["PDBFile"]
        elif params.get("modelFile"):
            params["structureFile"] = params["modelFile"]

    out_dir = Path(params["outputDir"])
    scratch = out_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    sf = params.get("structureFile")
    if sf:
        ext = Path(sf).suffix
        if ext in (".cif", ".pdb", ".PDB", ".pdb1"):
            _, fn_SP = calculate_sp(params, out_dir, device=device, log=lambda m: _log(log, m))
            params["modelFile"] = str(fn_SP)
        elif ext == ".mrc":
            params["modelFile"] = sf
        else:
            raise ValueError(f"Structure filetype not recognized for {sf}")

    # rotations
    if params.get("angle_inc") is not None:
        if params.get("psi_inc") is None:
            params["psi_inc"] = params["angle_inc"]
        _log(log, f"calculating search grid for angle_inc={params['angle_inc']:6.3f}, "
                  f"psi_inc={params['psi_inc']:6.3f}...")
        R, _ = calculate_search_grid("C1", params["angle_inc"], params["psi_inc"],
                                     dedupe=bool(params.get("dedupe_orientations", 0)))
        params["range_degrees"] = (params["angle_inc"] + params["psi_inc"]) / 2
        if write_files:
            sio.write_rotations_file(R, out_dir / "rotations.txt")
    else:
        R = sio.read_rotations_file(params["rotationsFile"])
    _log(log, f"{R.shape[0]} orientations in search grid")

    # defocus stack
    df = np.atleast_1d(np.asarray(params["defocus"], dtype=np.float64)).copy()
    if params.get("defocus_format") == "CTFFind":
        df[0:2] = df[0:2] / 10
        df[2] = df[2] * (-math.pi / 180)
    T, inc = params["T_sample"], params["df_inc"]
    if T > 0:
        t = matlab_colon(-T / 2, inc, T / 2)
        t = t - t.mean()
        df = np.column_stack([df[0] + t, df[1] + t, np.full(t.size, df[2])])
    df = np.atleast_2d(df)
    params["defocus"] = df

    if params.get("aPerPix_search") is None:
        params["aPerPix_search"] = params["aPerPix"]
    a_orig = float(params["aPerPix"])

    # ---- image (MATLAB lines 186-270)
    im = sio.mr(params["imageFile"]).astype(np.float64)
    params["aPerPix_orig"] = a_orig
    params["image_shape_orig"] = im.shape
    if a_orig != params["aPerPix_search"]:
        im = resize_f(im, a_orig / params["aPerPix_search"], "newSize")
        params["aPerPix"] = params["aPerPix_search"]
    dims_imref = im.shape
    im = resize_for_fft(im, "pad", 0).astype(np.float64)
    Npix_im = im.shape
    N0, N1 = Npix_im
    if params["psdFilterFlag"]:
        im = im - im.mean()
        fPSD, im = psd_filter(im, "sqrt")
        im = nm(im)
        fPSD = np.fft.ifftshift(fPSD).astype(np.float32)
    else:
        # DEVIATION: in v2.1 this branch calls resize_F(ones,1,'newSize'), which
        # subtracts the mode (=1) and returns all zeros -> every template becomes 0
        # and SD=0 (NaN output). Confirmed in Octave. We use the intended ones-filter.
        # QUIRK kept: MATLAB ends with fftshift, so the zeroed element is the centre
        # of a *centred* array, which after the half-crop is not the DC term.
        fPSD = np.ones(Npix_im, dtype=np.float32)
        fPSD = np.fft.ifftshift(fPSD)
        fPSD[0, 0] = 0
        fPSD = np.fft.fftshift(fPSD)
    if params["maskCrossFlag"]:
        im = mask_central_cross(im)
    im = crop_or_pad(im, dims_imref)
    im = nm(im)
    im = resize_for_fft(im, "pad", 0)
    im32 = np.asarray(im, dtype=np.float32)
    imref_F = sfft.fftn(np.fft.ifftshift(im32)) / np.float32(math.sqrt(N0 * N1))
    h0 = N0 // 2 + 1
    if write_files:
        sio.mw(im32, out_dir / "searchImage.mrc", params["aPerPix"])
    _log(log, f"using image with {dims_imref[0]} x {dims_imref[1]} image pixels, "
              f"{N0} x {N1} total pixels")

    # ---- scattering potential (MATLAB lines 273-302)
    SPV = sio.mr(params["modelFile"])  # single precision, as in MATLAB
    if params["aPerPix_search"] != a_orig:
        SPV = resize_f(SPV, a_orig / params["aPerPix_search"], "newSize")
        SPV = resize_for_fft(SPV, "crop")
    if params.get("mask_params") is not None:
        SPV = mask_a_volume(SPV.astype(np.float32), params["mask_params"])
    SPV = np.asarray(SPV, dtype=np.float32)
    Npix = int(min(SPV.shape))
    K = int(params.get("oversample", 1) or 1)
    if K > 1:  # OPTION: zero-pad the (background-subtracted) potential K-fold before the FFT
        bg = matlab_mode(SPV)
        SPV = crop_or_pad(SPV - bg, [K * Npix] * 3, 0).astype(np.float32)
        _log(log, f"oversampling potential {K}x -> {SPV.shape[0]}^3")
    F_abs = params.get("F_abs") or 0
    V = (SPV + 1j * F_abs * SPV).astype(np.complex64)
    V_F = np.fft.fftshift(sfft.fftn(np.fft.ifftshift(V))).astype(np.complex128)
    cpv = V_F.shape[0] // 2 + 1
    V_F[cpv - 1, cpv - 1, cpv - 1] = 0

    # ---- CTF (MATLAB lines 359-367): uses current params (aPerPix_search)
    C = ctf(df, [N0, N1], params)
    C = C * template_envelope(N0, params)[:, :, None]   # OPTION (1 when template_bfactor=0)
    C = np.stack([np.fft.ifftshift(C[:, :, i]) for i in range(C.shape[2])], axis=2)
    CTF_half = np.imag(C)[:h0, :, :]

    n_rot_full = R.shape[0]
    n_dfs = df.shape[0]
    if params.get("highThr") is None:
        high_thr = math.sqrt(2) * erfcinv(2.0 / (np.prod(dims_imref) * n_rot_full * n_dfs))
    else:
        high_thr = float(params["highThr"])
    n_samples = n_rot_full * float(np.prod(dims_imref)) * n_dfs

    return SearchSetup(
        params=params, out_dir=out_dir, scratch_dir=scratch, R_full=R,
        dims_imref=tuple(dims_imref), Npix_im=(N0, N1), imref=im32,
        imref_F_half=np.ascontiguousarray(imref_F[:h0, :].astype(np.complex64)),
        fPSD_half=np.ascontiguousarray(fPSD[:h0, :]), CTF_half=CTF_half, V_F=V_F,
        Npix=Npix, defocus=df, high_thr=high_thr, n_samples=n_samples, log_lines=log,
        fPSD_full=np.asarray(fPSD, dtype=np.float32), oversample=K,
    )


# ============================================================================ main loop

def _trilinear(Vr, Vi, u1, u2, u3):
    """MATLAB interpn(...,'linear',0) on a 1-based grid 1..N along each axis."""
    import torch
    N = Vr.shape[0]
    valid = (u1 >= 1) & (u1 <= N) & (u2 >= 1) & (u2 <= N) & (u3 >= 1) & (u3 <= N)
    i1 = torch.clamp(torch.floor(u1), 1, N - 1)
    i2 = torch.clamp(torch.floor(u2), 1, N - 1)
    i3 = torch.clamp(torch.floor(u3), 1, N - 1)
    f1, f2, f3 = u1 - i1, u2 - i2, u3 - i3
    a1, a2, a3 = (i1 - 1).long(), (i2 - 1).long(), (i3 - 1).long()
    vr, vi = Vr.reshape(-1), Vi.reshape(-1)
    out_r = torch.zeros_like(u1)
    out_i = torch.zeros_like(u1)
    for d1 in (0, 1):
        w1 = f1 if d1 else 1 - f1
        for d2 in (0, 1):
            w2 = f2 if d2 else 1 - f2
            for d3 in (0, 1):
                w3 = f3 if d3 else 1 - f3
                idx = ((a1 + d1) * N + (a2 + d2)) * N + (a3 + d3)
                idx = torch.where(valid, idx, torch.zeros_like(idx))
                w = w1 * w2 * w3
                out_r = out_r + w * vr[idx]
                out_i = out_i + w * vi[idx]
    z = torch.zeros_like(out_r)
    return torch.where(valid, out_r, z), torch.where(valid, out_i, z)


def _median_lastdims(x):
    """MATLAB median over all elements of each batch item (mean of middle two)."""
    import torch
    flat = x.reshape(x.shape[0], -1)
    s, _ = torch.sort(flat, dim=1)
    n = s.shape[1]
    if n % 2:
        return s[:, n // 2]
    return 0.5 * (s[:, n // 2 - 1] + s[:, n // 2])


def _templates(Vr, Vi, Rb, gx, gy, cp):
    """Central-slice templates for rotations Rb (B,3,3): MATLAB lines 505-516.

    Returns real-space templates (median removed), fftshifted (centred), float64.
    """
    import torch
    # xyz_r = (R' * [x y 0]')' + cp  ->  p_k = R[0,k]*x + R[1,k]*y + cp
    px = Rb[:, 0, 0, None, None] * gx + Rb[:, 1, 0, None, None] * gy + cp
    py = Rb[:, 0, 1, None, None] * gx + Rb[:, 1, 1, None, None] * gy + cp
    pz = Rb[:, 0, 2, None, None] * gx + Rb[:, 1, 2, None, None] * gy + cp
    tr, ti = _trilinear(Vr, Vi, py, px, pz)        # interpn(V, Y, X, Z)
    projF = torch.complex(tr, ti)
    templ = torch.fft.ifft2(torch.fft.ifftshift(projF, dim=(1, 2))).real
    templ = templ - _median_lastdims(templ)[:, None, None]
    return torch.fft.fftshift(templ, dim=(1, 2))


def make_templates(st: "SearchSetup", rot_indices, device="cpu") -> np.ndarray:
    """Real-space templates (no CTF) for 1-based rotation indices -> (n, N, N)."""
    import torch
    dev = torch.device(device)
    N = st.Npix
    cp = N // 2 + 1
    Vr = torch.as_tensor(np.ascontiguousarray(st.V_F.real), device=dev)
    Vi = torch.as_tensor(np.ascontiguousarray(st.V_F.imag), device=dev)
    ar = torch.arange(1, N + 1, device=dev, dtype=torch.float64)
    rr, cc_ = torch.meshgrid(ar, ar, indexing="ij")
    R = normalize_rm(st.R_full[np.asarray(rot_indices) - 1])
    K = st.oversample
    cpV = st.V_F.shape[0] // 2 + 1
    return _templates(Vr, Vi, torch.as_tensor(R, device=dev), (cc_ - cp) * K, (rr - cp) * K,
                      cpV).cpu().numpy()


def run_job(st: SearchSetup, jobref: int, device=None, batch_size: int = 1,
            write_files: bool = True, max_rotations: int | None = None):
    """Main cross-correlation loop for job ``jobref`` (1-based) of params['nCores'].

    Returns dict with sum1, sum2 (float64, (N0,N1,nDf)), SD (per rotation, last df),
    arb (n,3) rows [value, linear col-major 1-based location, rotation index].
    """
    import torch

    p = st.params
    if device is None:
        if torch.cuda.is_available():
            device = f"cuda:{(jobref - 1) % torch.cuda.device_count()}"
        else:
            device = "cpu"
    dev = torch.device(device)

    n_cores = int(p["nCores"])
    R_inds = assign_jobs(st.R_full.shape[0], n_cores, jobref)  # 1-based
    if max_rotations is not None:
        R_inds = R_inds[:max_rotations]
    R = normalize_rm(st.R_full[R_inds - 1].astype(np.float64))
    file_mult = int(R_inds[0]) - 1
    n_rot = R.shape[0]

    N = st.Npix
    N0, N1 = st.Npix_im
    h0 = N0 // 2 + 1
    nD = st.n_dfs
    cp = N // 2 + 1

    K = st.oversample
    vdt = np.float64 if K == 1 else np.float32   # oversampled volumes kept in single precision
    Vr = torch.as_tensor(np.ascontiguousarray(st.V_F.real.astype(vdt)), device=dev).double()
    Vi = torch.as_tensor(np.ascontiguousarray(st.V_F.imag.astype(vdt)), device=dev).double()
    rr, cc_ = torch.meshgrid(torch.arange(1, N + 1, device=dev, dtype=torch.float64),
                             torch.arange(1, N + 1, device=dev, dtype=torch.float64), indexing="ij")
    gx = (cc_ - cp) * K   # x varies along columns (scaled to the oversampled grid)
    gy = (rr - cp) * K    # y varies along rows
    cpV = st.V_F.shape[0] // 2 + 1

    imF = torch.as_tensor(st.imref_F_half, device=dev)                 # (h0,N1) c64
    fPSD = torch.as_tensor(st.fPSD_half, device=dev)                   # (h0,N1) f32
    CTF = torch.as_tensor(np.moveaxis(st.CTF_half, 2, 0), device=dev)  # (nD,h0,N1) f64
    norm_factor = float(N0 * N1)
    scale_temp = float(2.0 * math.sqrt(h0 * N1))

    # template placement (cropOrPad pad offsets) and fftshift/ifftshift
    off0 = (N0 // 2 + 1) - (N // 2 + 1)
    off1 = (N1 // 2 + 1) - (N // 2 + 1)

    sum1 = torch.zeros((nD, N1, N0), dtype=torch.float64, device=dev)  # stored transposed
    sum2 = torch.zeros_like(sum1)
    SD_all = np.zeros((n_rot, nD), dtype=np.float32)
    arb_chunks = []
    arb_thr = float(p["arbThr"])

    t0 = time.time()
    R_t = torch.as_tensor(R, device=dev)
    for b0 in range(0, n_rot, batch_size):
        b1 = min(n_rot, b0 + batch_size)
        templ = _templates(Vr, Vi, R_t[b0:b1], gx, gy, cpV)           # (B,N,N) f64, centred
        B = b1 - b0
        timg = torch.zeros((B, N0, N1), dtype=torch.float32, device=dev)
        timg[:, off0:off0 + N, off1:off1 + N] = templ.to(torch.float32)

        temp = torch.fft.rfftn(timg, dim=(2, 1))                       # (B,h0,N1) c64
        temp = temp / scale_temp
        temp = temp * fPSD
        tF = temp[:, None] * CTF.to(torch.float32)[None]               # (B,nD,h0,N1)
        v = (tF.real ** 2 + tF.imag ** 2).sum(dim=(2, 3))              # float32
        SD = torch.sqrt(2.0 * v / norm_factor)
        tF = tF / SD[:, :, None, None]
        ccF = imF[None, None] * torch.conj(tF)
        cc = torch.fft.irfftn(ccF, s=(N1, N0), dim=(3, 2), norm="forward")
        cc = cc / math.sqrt(norm_factor)                               # (B,nD,N0,N1) f32
        ccd = cc.double().transpose(2, 3)                              # (B,nD,N1,N0)
        sum1 += ccd.sum(0)
        sum2 += (ccd ** 2).sum(0)
        SD_all[b0:b1] = SD.cpu().numpy()

        hit = ccd > arb_thr
        if bool(hit.any()):
            ib, ij, ic, ir = torch.nonzero(hit, as_tuple=True)        # col-major order
            vals = ccd[ib, ij, ic, ir]
            loc = ir + ic * N0 + ij * N0 * N1 + 1
            qind = ib + b0 + 1 + file_mult
            arb_chunks.append(torch.stack([vals, loc.double(), qind.double()], 1).cpu().numpy())

        done = b1
        if done % max(100, batch_size) < batch_size or done == n_rot:
            dt = time.time() - t0
            print(f"job {jobref}: {done}/{n_rot} orientations, "
                  f"{1000 * dt / done:.1f} ms/orientation")

    arb = np.concatenate(arb_chunks, 0) if arb_chunks else np.zeros((0, 3))
    s1 = sum1.permute(2, 1, 0).cpu().numpy()   # back to (N0,N1,nD)
    s2 = sum2.permute(2, 1, 0).cpu().numpy()
    SD_last = SD_all[:, -1]                    # QUIRK: MATLAB keeps only the last df's SD

    if write_files:
        tag = zp_int(jobref)
        sio.mw(s1, st.scratch_dir / f"search_{tag}_sum.mrc", st.params["aPerPix"])
        sio.mw(s2, st.scratch_dir / f"search_{tag}_ssum.mrc", st.params["aPerPix"])
        sio.write_single(st.scratch_dir / f"search_{tag}_SD.dat", SD_last)
        sio.write_double(st.scratch_dir / f"search_{tag}_arb.dat", arb)
        (st.scratch_dir / f"search_{tag}_status.json").write_text(json.dumps(
            dict(job=jobref, evaluated_orientations=int(n_rot),
                 assigned_orientations=int(assign_jobs(st.R_full.shape[0], n_cores, jobref).size))))
    return dict(sum1=s1, sum2=s2, SD=SD_last, SD_all=SD_all, arb=arb,
                elapsed=time.time() - t0, n_rot=n_rot)


# ============================================================================ combine

def _hist_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """MATLAB hist(x, centers)."""
    mids = (centers[:-1] + centers[1:]) / 2
    k = np.searchsorted(mids, x, side="right")
    return np.bincount(k, minlength=centers.size).astype(np.float64)


def keep_region(st: SearchSetup, margin) -> np.ndarray:
    """Pixels kept by combine's edge/margin rule, (N0, N1) bool."""
    N0, N1 = st.Npix_im
    cp_im = np.array([N0 // 2 + 1, N1 // 2 + 1])
    r = np.abs(np.arange(1, N0 + 1) - cp_im[0])[:, None]
    c = np.abs(np.arange(1, N1 + 1) - cp_im[1])[None, :]
    return ~((r > cp_im[0] - margin[0]) | (c > cp_im[1] - margin[1]))


def combine(st: SearchSetup, n_jobs: int | None = None):
    """MATLAB lines 628-863: merge job outputs, normalise, write SH and highVals."""
    p = st.params
    n_jobs = n_jobs or int(p["nCores"])
    N0, N1 = st.Npix_im
    nD = st.n_dfs
    n_rot_full = st.R_full.shape[0]
    out = st.out_dir

    n_eval = 0
    for j in range(1, n_jobs + 1):
        f = st.scratch_dir / f"search_{zp_int(j)}_status.json"
        n_eval += json.loads(f.read_text())["evaluated_orientations"] if f.exists() else 0
    complete = n_eval == n_rot_full
    if not complete:
        print(f"WARNING: incomplete search ({n_eval} of {n_rot_full} orientations evaluated); "
              "moments and trial counts use the evaluated number; not a v2.1-equivalent run")
        n_rot_full = n_eval
    n_samples = n_rot_full * float(np.prod(st.dims_imref)) * nD
    sum_g = np.zeros((N0, N1, nD))
    ssum_g = np.zeros((N0, N1, nD))
    SDl, av_list = [], []
    with open(out / "search_listAboveThreshold.dat", "wb") as fa:
        for j in range(1, n_jobs + 1):
            tag = zp_int(j)
            SDl.append(sio.read_single(st.scratch_dir / f"search_{tag}_SD.dat"))
            a = sio.read_double(st.scratch_dir / f"search_{tag}_arb.dat").reshape(-1, 3)
            av = np.column_stack([a[:, 2], a[:, 1], a[:, 0]])     # [ind loc val]
            av.astype("<f8").tofile(fa)
            av_list.append(av)
            s = sio.mr(st.scratch_dir / f"search_{tag}_sum.mrc").astype(np.float64)
            ss = sio.mr(st.scratch_dir / f"search_{tag}_ssum.mrc").astype(np.float64)
            sum_g += s.reshape(N0, N1, nD)
            ssum_g += ss.reshape(N0, N1, nD)

    mean_im = sum_g / n_rot_full
    sd_im = np.sqrt(ssum_g / n_rot_full - mean_im ** 2)
    av_all = np.concatenate(av_list, 0) if av_list else np.zeros((0, 3))
    ai, al, av = av_all[:, 0], av_all[:, 1].astype(np.int64), av_all[:, 2]
    r = (al - 1) % N0 + 1
    c = ((al - 1) // N0) % N1 + 1
    z = (al - 1) // (N0 * N1) + 1
    locs = np.column_stack([r, c, z])

    cp_t = st.Npix // 2 + 1
    cp_im = np.array([N0 // 2 + 1, N1 // 2 + 1])
    margin = np.floor((np.array(st.Npix_im) - np.array(st.dims_imref)) / 2) + 1 + math.floor(cp_t / 2) + 1
    if p["margin_pix"] > 0:
        margin = (math.floor(p["margin_pix"]) / 2) * np.ones(2)
    xa = np.abs(locs[:, :2] - cp_im[None])
    keep = ~((xa[:, 0] > cp_im[0] - margin[0]) | (xa[:, 1] > cp_im[1] - margin[1]))
    ai, av, locs = ai[keep], av[keep], locs[keep]

    li = (locs[:, 0] - 1, locs[:, 1] - 1, locs[:, 2] - 1)
    av_norm = (av - mean_im[li]) / sd_im[li]
    order = np.argsort(av_norm, kind="stable")
    av_norm, ai, locs = av_norm[order], ai[order], locs[order]

    ff = np.zeros((N0, N1, nD))
    ff_inds = np.zeros((N0, N1, nD))
    if av_norm.size:
        lin = (locs[:, 0] - 1) + (locs[:, 1] - 1) * N0 + (locs[:, 2] - 1) * N0 * N1
        _, first_rev = np.unique(lin[::-1], return_index=True)
        last = av_norm.size - 1 - first_rev           # last write wins, as in the loop
        l2 = locs[last] - 1
        ff[l2[:, 0], l2[:, 1], l2[:, 2]] = av_norm[last]
        ff_inds[l2[:, 0], l2[:, 1], l2[:, 2]] = ai[last]
    sio.mw(ff, out / "search_vals.mrc", p["aPerPix"])
    sio.mw(ff_inds, out / "search_qInds.mrc", p["aPerPix"])

    if av_norm.size:
        xs = np.linspace(av_norm.min(), av_norm.max(), 10000)
        nt = _hist_centers(av_norm, xs)
        y_meas = nt.sum() - np.cumsum(nt)
        y_model = erfc(xs / math.sqrt(2)) / 2 * n_samples
        with open(out / "search_SH.txt", "w") as fh:
            for a_, b_, c_ in zip(xs, y_model, y_meas):
                fh.write("%8.4f\t%12.3f\t%12i\n" % (a_, b_, int(round(c_))))

    thr = math.sqrt(2) * erfcinv(2.0 / (n_rot_full * (N0 * N1) * nD))
    hi = np.nonzero(av_norm > thr)[0]
    print(f"found {hi.size} values above threshold ({thr:.3f})")
    if hi.size:
        with open(out / "highVals.txt", "w") as fh:
            for k in hi:
                fh.write("%i\t%8.3f\t%i\t%i\t%i\n" % (ai[k], av_norm[k], *locs[k]))

    sio.mw(mean_im, out / "search_mean.mrc", p["aPerPix"])
    sio.mw(sd_im, out / "search_SD.mrc", p["aPerPix"])
    sio.write_single(out / "search_SDs.dat", np.concatenate(SDl) if SDl else [])
    from .provenance import provenance
    from .stats import calibrate_threshold
    cal = calibrate_threshold(ff, mean_im, sd_im, float(p["arbThr"]), keep_region(st, margin),
                              n_samples)
    status = dict(is_complete=complete, evaluated_pose_count=int(n_rot_full),
                  trial_count=n_samples, raw_preselection_threshold=float(p["arbThr"]),
                  threshold_operator=">", score_kind="pixelwise-normalised CC (flat-fielded)",
                  score_stage="global search (pre-refinement)", coordinate_frame=
                  "search grid, 1-based [row, col], padded image", sampling_A=float(p["aPerPix"]),
                  gaussian_threshold=thr, calibration=cal, smap_port_version=__version__,
                  provenance=provenance(p))
    (out / "search_status.json").write_text(json.dumps(status, indent=1))
    return dict(is_complete=complete, av_norm=av_norm, ai=ai, locs=locs, thr=thr, ff=ff, ff_inds=ff_inds,
                mean=mean_im, sd=sd_im)


# ============================================================================ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SMAP global 2DTM search (Python port of v2.1)")
    ap.add_argument("params", help=".par file (function search_global)")
    ap.add_argument("jobref", nargs="?", type=int, default=None,
                    help="job number (1..nCores). Omit to run all jobs in this process.")
    ap.add_argument("--device", default=None, help="cpu, cuda, cuda:1 ... (default: auto)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="orientations per GPU pass (1 = MATLAB order, larger = faster)")
    ap.add_argument("--max-rotations", type=int, default=None,
                    help="debug: limit orientations per job")
    ap.add_argument("--no-refine", action="store_true",
                    help="stop after combine (skip clustering/refinement/optimisation)")
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite path prefixes in the .par file, e.g. /work=D:/rnf213 (repeatable)")
    ap.add_argument("--output-dir", default=None,
                    help="override outputDir from the .par (e.g. to keep an existing v2.1 output)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="refine even when not all orientations were evaluated (debug only)")
    ap.add_argument("--log", default=None,
                    help="also write all output to this UTF-8 file (e.g. run.log; avoids "
                         "PowerShell '>' writing UTF-16, which the RNF213 scripts cannot read)")
    a = ap.parse_args(argv)
    if a.log:
        import sys
        fh = open(a.log, "a", encoding="utf-8")

        class _Tee:
            def __init__(self, *s): self.s = s
            def write(self, x):
                for t in self.s:
                    t.write(x)
                    t.flush()
            def flush(self):
                for t in self.s:
                    t.flush()
        old = sys.stdout
        sys.stdout = _Tee(old, fh)
        try:
            return _main(a, ap)
        finally:
            sys.stdout = old
            fh.close()
    return _main(a, ap)


def _main(a, ap) -> int:
    pm = []
    for item in a.path_map:
        if "=" not in item:
            ap.error(f"--path-map expects OLD=NEW, got {item!r}")
        old, new = item.split("=", 1)
        pm.append((old, new))

    st = prepare(a.params, device=a.device, path_map=pm, output_dir=a.output_dir)
    st.params["_run_settings"] = dict(device=a.device or "auto", batch_size=a.batch_size,
                                      max_rotations=a.max_rotations, path_map=a.path_map)
    n_cores = int(st.params["nCores"])
    jobs = [a.jobref] if a.jobref else list(range(1, n_cores + 1))
    for j in jobs:
        run_job(st, j, device=a.device, batch_size=a.batch_size, max_rotations=a.max_rotations)
    if a.jobref is None or a.jobref == n_cores:
        missing = [j for j in range(1, n_cores + 1)
                   if not (st.scratch_dir / f"search_{zp_int(j)}_arb.dat").exists()]
        if missing:
            print(f"waiting for jobs {missing}; run combine after they finish")
        else:
            res = combine(st)
            if not res["is_complete"] and not a.allow_incomplete:
                print("INCOMPLETE search (debug run): refinement skipped and the run is not marked "
                      "done; use --allow-incomplete to override")
                return 2
            if not a.no_refine:
                from .refine import optimize_particles, refine_particles, select_particles, write_final
                jobs = select_particles(st, res)
                parts = refine_particles(st, jobs, res["mean"], res["sd"], device=a.device)
                opt = optimize_particles(st, parts, device=a.device) if parts else []
                write_final(st, opt)
            # log lines that the RNF213 workflow scripts look for in run.log
            print(f"Done combining search output at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
