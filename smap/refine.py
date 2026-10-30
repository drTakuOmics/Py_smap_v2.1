"""Particle selection and refinement: smappoi_search_global.m lines 864-1990.

* :func:`select_particles`   864-968   cluster above-threshold pixels per defocus plane
                                       (clusterImByThr), suppress near-duplicates
                                       (pdist + pairwiseQD), write opt_toDo files
* :func:`refine_particles`   983-1353  per-particle grid refinement on the search
                                       grid: defocus scan, angular "bump" scan, defocus
                                       scan, full-image peak -> opt_done files
* :func:`optimize_particles` 1418-1562 + optimize_xcorr/_angle/_phase_shifts
                                       (1811-1990): Nelder-Mead sub-pixel/angle
                                       optimisation at the original pixel size with an
                                       exit-wave template
* :func:`write_final`        1566-1704 list_final.txt, particles.{mrc,par,txt,csv},
                                       templates.mrc

Options (``.par`` keys, default = v2.1 behaviour):
  ``fix_optimize_angle 1``  optimize_angle uses the trial rotation (v2.1 ignores it)
  ``refine_local_peak R``   final full-image peak searched within R px of the particle
                            (v2.1: anywhere in the image; 0 = v2.1)
  ``fix_cluster_thresholds 1`` compare distance with dThr and angle with qThr
                            (v2.1 swaps them; identical when both are 10)
  ``refine_continuous 1``   replace optimize_angle by one Nelder-Mead over a minimal,
                            scaled parameter set: rotation vector (3, degrees), shift
                            (2, pixels) and common defocus offset (1, nm); initial
                            simplex = [inc_degrees/2, 0.5 px, df step]. v2.1 searches a
                            4-parameter quaternion (one redundant scale) plus shift with
                            initial steps of ~0.03 deg and 0.025 px, and never defocus.
"""

from __future__ import annotations

import json
import math

import numpy as np
import scipy.fft as sfft
from scipy import ndimage

from . import io as sio
from .ctf import ctf, template_envelope
from .grid import q2r
from .rot import bump_rotations, pairwise_qd, rodrigues, rotm_to_quat, smap2frealign
from .search_global import SearchSetup, _trilinear
from .utils import (apply_phase_shifts_2d, crop_or_pad, crop_patch, ftj, iftj,
                    imresize_bilinear, mask_a_volume, mask_central_cross,
                    matlab_colon, matlab_round, nm, normalize_rm, psd_filter,
                    resize_for_fft, rrj, zp_int)


# ============================================================================ selection

def cluster_im_by_thr(im: np.ndarray, mipi: np.ndarray, thr: float, R: np.ndarray):
    """@smap/clusterImByThr.m -> list of dicts sorted by MaxVal (descending, stable).

    8-connected regions labelled in MATLAB (column-major) order; within a region the
    first maximum in column-major pixel order defines xy (1-based) and the rotation.
    """
    bw = im > thr
    if not bw.any():
        return []
    lab, n = ndimage.label(bw.T, structure=np.ones((3, 3)))   # transpose -> column-major order
    out = []
    for k in range(1, n + 1):
        cs, rs = np.nonzero(lab == k)                         # column-major pixel order
        vals = im[rs, cs]
        j = int(np.argmax(vals))                              # first max (strict > in MATLAB)
        idx = int(mipi[rs[j], cs[j]])
        out.append(dict(MaxVal=float(vals.max()), xy=(int(rs[j]) + 1, int(cs[j]) + 1),
                        q=normalize_rm(R[idx - 1]), area=int(rs.size)))
    order = np.argsort(-np.array([o["MaxVal"] for o in out]), kind="stable")
    return [out[i] for i in order]


def select_particles(st: SearchSetup, comb: dict, write_files: bool = True):
    """Lines 864-968. Returns list (per optimisation job) of particle rows
    [row, col, df1, df2, ang, qInd] exactly as re-read from opt_toDo files."""
    p = st.params
    ff, ff_inds = comb["ff"], comb["ff_inds"]
    nD = ff.shape[2]
    ss, qbest, xy, ind_vec = [], [], [], []
    for i in range(nD):
        cl = cluster_im_by_thr(ff[:, :, i], ff_inds[:, :, i], float(p["optThr"]), st.R_full)
        for c in cl:
            ss.append(c["MaxVal"]); qbest.append(c["q"]); xy.append(c["xy"]); ind_vec.append(i + 1)
    n_el = len(ss)
    n_cores = int(p["nCores"])
    jobs: list[list] = []
    if n_el > 0:
        mv = np.array(ss)
        xy = np.array(xy, dtype=np.float64)
        qbest = np.array(qbest)
        pd = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        qd = pairwise_qd(qbest, qbest)
        upper = np.triu(np.ones((n_el, n_el), bool), 1)       # tril(nan)+x: NaN on/below diag
        if p.get("fix_cluster_thresholds", 0):
            s_mat = upper & (pd < p["dThr"]) & (qd < p["qThr"])
        else:  # QUIRK: v2.1 compares pixel distance with qThr and angle with dThr
            s_mat = upper & (pd < p["qThr"]) & (qd < p["dThr"])
        exclude = []
        for i in range(n_el):
            if s_mat[i].any():
                ntu = [i] + list(np.nonzero(s_mat[i])[0])
                h = ntu[int(np.argmax(mv[ntu]))]
                s_mat[ntu, :] = False
                exclude += [j for j in ntu if j != h]
        keep = [j for j in range(n_el) if j not in set(exclude)]
        n_found = len(keep)
        print(f"starting optimization on {n_found} particles...")
        xy_tu = xy[keep].astype(int)
        ind_tu = np.array(ind_vec)[keep]
        df_tu = np.asarray(p["defocus"])[ind_tu - 1]
        q_tu = np.array([ff_inds[xy_tu[i, 0] - 1, xy_tu[i, 1] - 1, ind_tu[i] - 1]
                         for i in range(n_found)])
        n_opt = min(n_found, n_cores)
        for i in range(n_opt):
            sel = list(range(i, n_found, n_opt))
            lines = ["%4i\t%4i\t%6.2f\t%6.2f\t%5.4f\t%i\n" % (xy_tu[k, 0], xy_tu[k, 1], df_tu[k, 0],
                     df_tu[k, 1], df_tu[k, 2], q_tu[k]) for k in sel]
            if write_files:
                (st.scratch_dir / f"opt_toDo_{zp_int(i + 1)}.txt").write_text("".join(lines))
            rows = [list(map(float, l.split())) for l in lines]   # MATLAB re-reads the text file
            jobs.append(rows)
    else:
        print("nothing to optimize")
    return jobs


# ============================================================================ stage B

class _Scorer:
    """Template scoring on an Npix patch (lines 1101-1118 and repeats), batched on torch."""

    def __init__(self, st: SearchSetup, fPSD_patch: np.ndarray, device):
        import torch
        self.t = torch
        self.dev = torch.device(device)
        N = st.Npix
        self.N = N
        self.cp = N // 2 + 1
        K = int(getattr(st, "oversample", 1) or 1)
        vdt = np.float64 if K == 1 else np.float32
        self.Vr = torch.as_tensor(np.ascontiguousarray(st.V_F.real.astype(vdt)), device=self.dev).double()
        self.Vi = torch.as_tensor(np.ascontiguousarray(st.V_F.imag.astype(vdt)), device=self.dev).double()
        ar = torch.arange(1, N + 1, device=self.dev, dtype=torch.float64)
        rr, cc = torch.meshgrid(ar, ar, indexing="ij")
        self.gx, self.gy = (cc - self.cp) * K, (rr - self.cp) * K
        self.cpV = st.V_F.shape[0] // 2 + 1
        self.fPSD = torch.as_tensor(fPSD_patch.astype(np.float32), device=self.dev)
        m = (rrj(np.ones((N, N))) * N).astype(np.float32) <= 5
        self.mask = torch.as_tensor(m, device=self.dev)

    def proj_corner(self, Rb):
        """projPot_F(idx_small_i{:}) with (1,1)=0, for rotations Rb (B,3,3)."""
        t = self.t
        Rb = t.as_tensor(np.asarray(Rb), device=self.dev)
        px = Rb[:, 0, 0, None, None] * self.gx + Rb[:, 1, 0, None, None] * self.gy + self.cpV
        py = Rb[:, 0, 1, None, None] * self.gx + Rb[:, 1, 1, None, None] * self.gy + self.cpV
        pz = Rb[:, 0, 2, None, None] * self.gx + Rb[:, 1, 2, None, None] * self.gy + self.cpV
        tr, ti = _trilinear(self.Vr, self.Vi, py, px, pz)
        P = t.fft.ifftshift(t.complex(tr, ti), dim=(1, 2)).to(t.complex64)
        P[:, 0, 0] = 0
        return P

    def set_patch(self, patch, pmean, psd):
        t = self.t
        pf = sfft.fftn(np.fft.ifftshift(patch.astype(np.float32)))
        pf = pf / np.float32(np.std(pf, ddof=1))
        self.patchF = t.as_tensor(pf.astype(np.complex64), device=self.dev)
        self.pm = t.as_tensor(pmean.astype(np.float32), device=self.dev)[self.mask]
        self.ps = t.as_tensor(psd.astype(np.float32), device=self.dev)[self.mask]

    def score(self, P, ctf_corner):
        """max over the 5-px disc of (cc - mean)/SD for templates P (B,N,N) x CTFs."""
        t = self.t
        C = t.as_tensor(np.asarray(ctf_corner, dtype=np.float32), device=self.dev)
        temp = P * self.fPSD * C
        flat = temp.reshape(temp.shape[0], -1)
        sd = t.sqrt((t.abs(flat - flat.mean(1, keepdim=True)) ** 2).sum(1) / (flat.shape[1] - 1))
        temp = temp / sd[:, None, None]
        cc = t.fft.fftshift(t.fft.ifft2(self.patchF * t.conj(temp)).real, dim=(1, 2)) * self.N
        v = (cc[:, self.mask] - self.pm) / self.ps
        return v.max(1).values.double().cpu().numpy()


def _ctf_corner_imag(df, N, params):
    C = np.imag(ctf(np.atleast_2d(df), [N, N], params)) * template_envelope(N, params)[:, :, None]
    return np.stack([np.fft.ifftshift(C[:, :, j]) for j in range(C.shape[2])])


def _df_scan(df, half, step):
    d1 = matlab_colon(df[0] - half, step, df[0] + half)
    d2 = matlab_colon(df[1] - half, step, df[1] + half)
    n = min(d1.size, d2.size)
    return np.column_stack([d1[:n], d2[:n], np.full(n, df[2])])


def refine_particles(st: SearchSetup, jobs, mean_im, sd_im, device=None, batch=64,
                     write_files: bool = True, log=print):
    """Lines 983-1353 for every job; returns list of particle dicts (in job order)."""
    import torch
    p = st.params
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N = st.Npix
    N0, N1 = st.Npix_im
    a_search = float(p["aPerPix"])
    meanImage = np.mean(np.asarray(mean_im, dtype=np.float32).reshape(N0, N1, -1), axis=2)
    SDImage = np.mean(np.asarray(sd_im, dtype=np.float32).reshape(N0, N1, -1), axis=2)
    fP = imresize_bilinear(np.fft.fftshift(st.fPSD_full), (N, N)).astype(np.float32)
    fPSD_patch = np.fft.ifftshift(fP)
    fPSD_patch[0, 0] = 0
    R_bump = bump_rotations(float(p["range_degrees"]), float(p["inc_degrees"]))
    log(f"refining over {p['range_degrees']:3.2f} degree range, {p['inc_degrees']:3.2f} "
        f"degree increments ({R_bump.shape[0]} rotations)")
    sc = _Scorer(st, fPSD_patch, device)
    imF_full = sfft.fftn(np.fft.ifftshift(st.imref)) / np.float32(math.sqrt(N0 * N1))
    a_orig = float(p.get("aPerPix_orig", a_search))
    local_r = float(p.get("refine_local_peak", 0) or 0)

    particles = []
    for job, rows in enumerate(jobs, start=1):
        out_vals = []
        for row in rows:
            xy = (int(row[0]), int(row[1]))
            df = np.array(row[2:5])
            qInd = int(row[5])
            R0 = st.R_full[qInd - 1]
            sc.set_patch(crop_patch(st.imref, (N, N), xy), crop_patch(meanImage, (N, N), xy),
                         crop_patch(SDImage, (N, N), xy))
            P0 = sc.proj_corner(R0[None])
            init = sc.score(P0, _ctf_corner_imag(df, N, p))[0]
            # defocus scan 1 (step = 2*aPerPix in nm units, as in v2.1)
            dft = _df_scan(df, 50, a_search * 2)
            c = sc.score(P0.expand(dft.shape[0], -1, -1), _ctf_corner_imag(dft, N, p))
            df_best = dft[int(np.argmax(c))]
            C_best = _ctf_corner_imag(df_best, N, p)
            # angular bump scan
            Rtu = normalize_rm(R_bump @ R0)
            cm = np.concatenate([sc.score(sc.proj_corner(Rtu[b:b + batch]), C_best)
                                 for b in range(0, Rtu.shape[0], batch)])
            q_best = Rtu[int(np.argmax(cm))]
            # defocus scan 2 (step = aPerPix)
            P1 = sc.proj_corner(q_best[None])
            dft = _df_scan(df_best, 50, a_search)
            c2 = sc.score(P1.expand(dft.shape[0], -1, -1), _ctf_corner_imag(dft, N, p))
            df_best = dft[int(np.argmax(c2))]
            C_best = _ctf_corner_imag(df_best, N, p)[0]
            # final value on the full image
            temp = P1[0].cpu().numpy() * fPSD_patch * C_best
            temp = np.fft.fftshift(np.real(sfft.ifftn(temp))).astype(np.float32)
            temp = crop_or_pad(temp, (N0, N1), 0)
            tF = sfft.fftn(np.fft.ifftshift(temp))
            tF = tF / np.float32(np.std(tF, ddof=1))
            ccf = np.fft.fftshift(np.real(sfft.ifftn(imF_full * np.conj(tF)))) * math.sqrt(N0 * N1)
            with np.errstate(divide="ignore", invalid="ignore"):
                cc = (ccf - meanImage) / SDImage
            cc = crop_or_pad(crop_or_pad(cc, st.dims_imref), (N0, N1), 0).astype(np.float64)
            if local_r > 0:  # OPTION: restrict to the neighbourhood of this particle
                rr, cc_ = np.ogrid[1:N0 + 1, 1:N1 + 1]
                cc = np.where((rr - xy[0]) ** 2 + (cc_ - xy[1]) ** 2 <= local_r ** 2, cc, np.nan)
            best = np.nanmax(cc)
            lin = np.nonzero((cc == best).T.ravel())[0][0]          # find(...,1,'first')
            bx, by = lin % N0 + 1, lin // N0 + 1
            log(f"particle {xy}: init {init:.3f} -> defocus {c.max():.3f} -> angle {cm.max():.3f}"
                f" -> defocus {c2.max():.3f} -> final (full) {best:.3f}")
            out_vals.append([best, bx, by, *df_best, *rotm_to_quat(q_best)])
            particles.append(dict(peak=best, xy_search=(bx, by), df=df_best.copy(), q=q_best.copy()))
        if write_files and out_vals:
            fmt = "%6.3f\t%4i\t%4i\t%6.2f\t%6.2f\t%4.3f\t%7.6f\t%7.6f\t%7.6f\t%7.6f\n"
            (st.scratch_dir / f"opt_done_{zp_int(job)}.txt").write_text("".join(fmt % tuple(v) for v in out_vals))

    # coordinates back to the original pixel size (lines 1318-1353)
    if p["aPerPix_search"] != a_orig and particles:
        h0, w0 = p["image_shape_orig"]
        dummy = resize_for_fft(np.zeros((h0, w0)), "pad", 0)
        cp0 = np.array(dummy.shape) // 2 + 1
        cpb = np.array([N0, N1]) // 2 + 1
        for prt in particles:
            Rb, Cb = prt["xy_search"]
            prt["xy"] = (int(matlab_round((Rb - cpb[0]) * (a_search / a_orig) + cp0[0])),
                         int(matlab_round((Cb - cpb[1]) * (a_search / a_orig) + cp0[1])))
    else:
        for prt in particles:
            prt["xy"] = prt["xy_search"]
    return particles


# ============================================================================ stage C

def fminsearch(f, x0, tol_fun=1e-3, tol_x=1e-4, max_iter=None, max_fev=None, step=None):
    """MATLAB fminsearch (Nelder-Mead, Lagarias et al.) with MATLAB's initial simplex,
    coefficients, ordering and stopping rules. ``step`` (port addition): explicit initial
    simplex edge per parameter instead of MATLAB's 5 % / 0.00025 rule."""
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    n = x0.size
    max_iter = max_iter or 200 * n
    max_fev = max_fev or 200 * n
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    v = np.zeros((n + 1, n))
    fv = np.zeros(n + 1)
    v[0] = x0
    fv[0] = f(x0)
    for j in range(n):
        y = x0.copy()
        if step is not None:
            y[j] = y[j] + step[j]
        else:
            y[j] = (1 + 0.05) * y[j] if y[j] != 0 else 0.00025
        v[j + 1] = y
        fv[j + 1] = f(y)
    o = np.argsort(fv, kind="stable"); v, fv = v[o], fv[o]
    it, fev = 1, n + 1
    while fev < max_fev and it < max_iter:
        if (np.max(np.abs(fv[0] - fv[1:])) <= max(tol_fun, 10 * np.spacing(fv[0]))
                and np.max(np.abs(v[1:] - v[0])) <= max(tol_x, 10 * np.spacing(np.max(v[0])))):
            break
        xbar = v[:n].mean(0)
        xr = (1 + rho) * xbar - rho * v[-1]; fxr = f(xr); fev += 1
        shrink = False
        if fxr < fv[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * v[-1]; fxe = f(xe); fev += 1
            if fxe < fxr:
                v[-1], fv[-1] = xe, fxe
            else:
                v[-1], fv[-1] = xr, fxr
        elif fxr < fv[n - 1]:
            v[-1], fv[-1] = xr, fxr
        elif fxr < fv[-1]:
            xc = (1 + psi * rho) * xbar - psi * rho * v[-1]; fxc = f(xc); fev += 1
            if fxc <= fxr:
                v[-1], fv[-1] = xc, fxc
            else:
                shrink = True
        else:
            xcc = (1 - psi) * xbar + psi * v[-1]; fxcc = f(xcc); fev += 1
            if fxcc < fv[-1]:
                v[-1], fv[-1] = xcc, fxcc
            else:
                shrink = True
        if shrink:
            for j in range(1, n + 1):
                v[j] = v[0] + sigma * (v[j] - v[0]); fv[j] = f(v[j])
            fev += n
        o = np.argsort(fv, kind="stable"); v, fv = v[o], fv[o]
        it += 1
    return v[0], fv[0]


class _Optimizer:
    """State of the 'true optimization' (lines 1418-1505) shared by optimize_xcorr."""

    def __init__(self, st: SearchSetup, device=None, log=print):
        import torch
        self.t = torch
        p = st.params
        self.p = p
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        SPV = sio.mr(p["modelFile"])
        if p.get("mask_params") is not None:
            SPV = mask_a_volume(SPV.astype(np.float32), p["mask_params"]).astype(np.float32)
        N = SPV.shape[0]
        self.N, self.cp = N, N // 2 + 1
        V_F = np.fft.fftshift(sfft.fftn(np.fft.ifftshift(SPV.astype(np.float32))))
        V_F[self.cp - 1, self.cp - 1, self.cp - 1] = 0
        # v2.1 stores single(real/imag); interpolation in float64 of single values
        self.Vr = torch.as_tensor(np.ascontiguousarray(V_F.real.astype(np.float32)), device=self.dev).double()
        self.Vi = torch.as_tensor(np.ascontiguousarray(V_F.imag.astype(np.float32)), device=self.dev).double()
        del V_F, SPV
        ar = torch.arange(1, N + 1, device=self.dev, dtype=torch.float64)
        rr, cc = torch.meshgrid(ar, ar, indexing="ij")
        self.gx, self.gy = cc - self.cp, rr - self.cp
        self.inds_mask = (rrj(np.ones((N, N))) * N) > (N / 2)
        g = (np.arange(N) - N / 2) / N
        self.gr = g[:, None]   # xd' (varies along rows)
        self.gc = g[None, :]   # yd' (varies along cols)

        nf = sio.mr(p["imageFile"]).astype(np.float32)
        n_orig = nf.shape
        nf = resize_for_fft(nf, "pad", 0).astype(np.float64)
        self.nfIm_unmod = nf.copy()
        nf = nf - nf.mean()
        fPSD, nf = psd_filter(nf, "sqrt")
        fPSD_patch = imresize_bilinear(fPSD, (N, N)).astype(np.float32)
        if p["maskCrossFlag"]:
            nf = mask_central_cross(nf)
        fPSD_patch[self.cp - 1, self.cp - 1] = 0
        npix_im = nf.shape
        self.n_orig, self.npix_im = tuple(n_orig), tuple(npix_im)
        nf = crop_or_pad(nf, n_orig)
        nf = nm(nf)
        self.nfIm = crop_or_pad(nf, npix_im, 0).astype(np.float64)
        self.fPSD = fPSD.astype(np.float32)
        self.fPSD_patch = fPSD_patch

        mi = np.mean(sio.mr(st.out_dir / "search_mean.mrc").reshape(*st.Npix_im, -1), axis=2)
        si = np.mean(sio.mr(st.out_dir / "search_SD.mrc").reshape(*st.Npix_im, -1), axis=2)
        mi = imresize_bilinear(crop_or_pad(mi, st.dims_imref), n_orig)
        self.meanImage = crop_or_pad(mi, self.nfIm.shape, 0).astype(np.float64)
        si = imresize_bilinear(crop_or_pad(si, st.dims_imref), n_orig)
        self.SDImage = crop_or_pad(si, self.nfIm.shape, 1).astype(np.float64)

    # ---- template: exit wave |w|^2 (optimize_xcorr / optimize_angle)
    def template(self, R, CTF):
        t = self.t
        Rb = t.as_tensor(np.asarray(R)[None], device=self.dev)
        # xyz_r = (R*xyz')' : X = R[0,:].v + cp ...  (note: R, not R')
        px = Rb[:, 0, 0, None, None] * self.gx + Rb[:, 0, 1, None, None] * self.gy + self.cp
        py = Rb[:, 1, 0, None, None] * self.gx + Rb[:, 1, 1, None, None] * self.gy + self.cp
        pz = Rb[:, 2, 0, None, None] * self.gx + Rb[:, 2, 1, None, None] * self.gy + self.cp
        tr, ti = _trilinear(self.Vr, self.Vi, py, px, pz)
        proj = t.complex(tr, ti)[0].cpu().numpy()
        ew = np.exp(1j * sfft.ifftn(np.fft.ifftshift(proj)))
        w = np.fft.fftshift(sfft.ifftn(sfft.fftn(ew) * CTF))
        return np.real(w * np.conj(w))

    def cc_value_at_shift(self, cc_F, shifts):
        """optimize_phase_shifts / tail of optimize_angle: cc at sub-pixel offset."""
        dphs = self.gr * (-shifts[0]) + self.gc * (-shifts[1])
        d = np.real(sfft.ifftn(np.fft.ifftshift(cc_F * np.exp(1j * 2 * np.pi * dphs))))
        return float(d[0, 0] * self.N)

    def optimize_xcorr(self, prt):
        p, N = self.p, self.N
        xy = prt["xy"]
        Rref = normalize_rm(prt["q"].T)
        CTF = np.fft.ifftshift(ctf(np.atleast_2d(prt["df"]), [N, N], p)[:, :, 0] * template_envelope(N, p))
        imref = nm(crop_patch(self.nfIm, (N, N), xy))
        imref_F = ftj(imref)
        meanref = crop_patch(self.meanImage, (N, N), xy)
        SDref = crop_patch(self.SDImage, (N, N), xy)
        tmpl = self.template(Rref, CTF)
        bg = np.nanmedian(tmpl[self.inds_mask])
        tmpl = tmpl - bg
        tF = ftj(tmpl) * self.fPSD_patch
        tF = tF / np.std(tF, ddof=1)
        cc_corr = (iftj(imref_F * np.conj(tF)) - meanref) / SDref
        cc_F = ftj(cc_corr)
        shifts, _ = fminsearch(lambda s: -self.cc_value_at_shift(cc_F, s), [0.0, 0.0])

        imref_s = apply_phase_shifts_2d(imref, shifts[::-1])
        mean_s = apply_phase_shifts_2d(self.meanImage, shifts[::-1])
        sd_s = apply_phase_shifts_2d(self.SDImage, shifts[::-1])
        meanref = crop_patch(mean_s, (N, N), xy)
        SDref = crop_patch(sd_s, (N, N), xy)
        imref_F = ftj(imref_s)
        fix = bool(self.p.get("fix_optimize_angle", 0))
        cache = {}

        def obj(x):
            Rt = q2r(x[:4]) @ Rref if fix else Rref        # QUIRK: v2.1 ignores x[:4]
            key = Rt.tobytes()
            if key not in cache:
                tm = self.template(Rt, CTF) - bg
                f2 = ftj(tm) * self.fPSD_patch
                cache[key] = f2 / np.std(f2, ddof=1)
            tmp = (iftj(imref_F * np.conj(cache[key])) - meanref) / SDref
            return -self.cc_value_at_shift(ftj(tmp), x[4:6] * 100)

        df_out = np.asarray(prt["df"], dtype=np.float64).copy()
        if int(self.p.get("refine_continuous", 0) or 0):
            RM, dsh, ddf = self._continuous(Rref, prt["df"], imref_F, meanref, SDref, bg)
            shifts = shifts + dsh
            df_out[:2] += ddf
            CTF = np.fft.ifftshift(ctf(np.atleast_2d(df_out), [N, N], p)[:, :, 0] * template_envelope(N, p))
        else:
            ang, _ = fminsearch(obj, [1, 0, 0, 0, 0, 0])
            RM = q2r(ang[:4]) @ Rref
            shifts = shifts + ang[4:6] * 100
        xy_opt = (xy[0] - shifts[0], xy[1] - shifts[1])

        tm = self.template(RM, CTF) - bg
        nf_s = apply_phase_shifts_2d(self.nfIm, shifts[::-1])
        unmod_s = apply_phase_shifts_2d(self.nfIm_unmod, shifts[::-1])
        patchref = crop_patch(unmod_s, (N, N), xy)
        mean_s = apply_phase_shifts_2d(self.meanImage, shifts[::-1])
        sd_s = apply_phase_shifts_2d(self.SDImage, shifts[::-1])
        tmf = crop_or_pad(tm, (self.nfIm.shape[0],) * 2, 0)
        tF = ftj(tmf) * self.fPSD
        tF = tF / np.std(tF, ddof=1)
        cc = (iftj(ftj(nf_s) * np.conj(tF)) - mean_s) / sd_s
        peak_opt = float(cc[xy[0] - 1, xy[1] - 1])
        return dict(q_opt=RM.T, xy_opt=xy_opt, peak_opt=peak_opt, patch=patchref, template=tm, df=df_out)

    def _continuous(self, Rref, df, imref_F, meanref, SDref, bg):
        """OPTION refine_continuous: x = [rotvec (deg, 3), shift (px, 2), d_defocus (nm)].
        The template rotation is rodrigues(x[:3]) @ Rref (same composition as v2.1's
        q2r(x[:4]) @ Rref). Returns (R, shift, d_defocus)."""
        p, N = self.p, self.N
        df0 = np.asarray(df, dtype=np.float64)
        ctf_cache, tm_cache = {}, {}

        def ctf_for(ddf):
            k = round(float(ddf), 4)
            if k not in ctf_cache:
                d = df0.copy()
                d[:2] += k
                ctf_cache[k] = np.fft.ifftshift(ctf(np.atleast_2d(d), [N, N], p)[:, :, 0]
                                                * template_envelope(N, p))
            return ctf_cache[k]

        def obj(x):
            Rt = rodrigues(np.radians(x[:3])) @ Rref
            key = (Rt.tobytes(), round(float(x[5]), 4))
            if key not in tm_cache:
                tm = self.template(Rt, ctf_for(x[5])) - bg
                f2 = ftj(tm) * self.fPSD_patch
                tm_cache[key] = f2 / np.std(f2, ddof=1)
            tmp = (iftj(imref_F * np.conj(tm_cache[key])) - meanref) / SDref
            return -self.cc_value_at_shift(ftj(tmp), x[3:5])

        inc = float(p.get("inc_degrees") or 0.5)
        df_step = 2.0 * float(p["aPerPix"])           # nm; same as the stage-B defocus scan
        x, _ = fminsearch(obj, np.zeros(6), tol_fun=1e-4, tol_x=1e-3, max_fev=900,
                          step=[inc / 2] * 3 + [0.5, 0.5] + [df_step])
        return rodrigues(np.radians(x[:3])) @ Rref, x[3:5], float(x[5])


def optimize_particles(st: SearchSetup, particles, device=None, log=print):
    """Lines 1418-1505: run optimize_xcorr for every particle (params at original pixel)."""
    p = st.params
    p["aPerPix"] = float(p.get("aPerPix_orig", p["aPerPix"]))    # line 1349
    opt = _Optimizer(st, device=device, log=log)
    off = [(opt.npix_im[k] // 2 + 1) - (opt.n_orig[k] // 2 + 1) for k in (0, 1)]
    out = []
    for i, prt in enumerate(particles, 1):
        r = opt.optimize_xcorr(prt)
        # zero-based original-image (x, y), undoing resizeForFFT padding of the image
        r["xy_orig0_est"] = (r["xy_opt"][1] - 1 - off[1], r["xy_opt"][0] - 1 - off[0])
        log(f"particle {i}: peak {prt['peak']:.3f} -> peak_opt {r['peak_opt']:.3f}")
        out.append({**prt, **r})
    return out


# ============================================================================ outputs

def write_final(st: SearchSetup, optimized, jobs_sizes=None):
    """list_final.txt (from opt_done files), particles.{mrc,par,txt,csv}, templates.mrc."""
    p, out = st.params, st.out_dir
    z = []
    for f in sorted(st.scratch_dir.glob("opt_done_*.txt")):
        txt = f.read_text().split()
        z += [float(v) for v in txt]
    z = np.array(z).reshape(-1, 10) if z else np.zeros((0, 10))
    with open(out / "list_final.txt", "w") as fh:
        for r in z:
            fh.write("%6.3f\t%i\t%i\t%6.2f\t%6.2f\t%4.3f\t%7.6f\t%7.6f\t%7.6f\t%7.6f\n" % tuple(r))
    if not optimized:
        return
    a = float(p["aPerPix"])
    sio.mw(np.stack([o["patch"] for o in optimized], axis=2), out / "particles.mrc", a)
    sio.mw(np.stack([o["template"] for o in optimized], axis=2), out / "templates.mrc", a)

    q_offset = np.diag([1.0, -1.0, -1.0])
    mag = 5000.0 / (a / 10)
    pv_file = out / "search_pval.mrc"
    pv = sio.mr(pv_file) if pv_file.exists() else None
    with open(out / "particles.par", "w") as fr, open(out / "particles.txt", "w") as fc, \
            open(out / "particles.csv", "w") as fcsv:
        fr.write("C           PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1"
                 "      DF2  ANGAST     OCC      LogP      SIGMA   SCORE  CHANGE\n")
        fc.write("#\tCC\n")
        fcsv.write("num_id,score_final,score_refined,x_orig0_est,y_orig0_est,row,col,row_opt,"
                   "col_opt,row_search,col_search,df1_nm,df2_nm,ast_rad,"
                   + ",".join(f"q_opt_{i}{j}" for i in range(3) for j in range(3)) + ",neg_log_p_search\n")
        for k, o in enumerate(optimized, 1):
            Q = normalize_rm(o["q_opt"] @ q_offset)
            psi_, th_, phi_ = smap2frealign(Q[None])[0]
            df = o["df"]
            dfs = [df[0] * 10, df[1] * 10, df[2] * (-180 / math.pi)]
            fr.write(str(k).rjust(7) + "%8.2f%8.2f%8.2f" % (psi_, th_, phi_) + "  "
                     + "0.00".rjust(8) + "  " + "0.00".rjust(8) + "  " + ("%5.0f" % mag).rjust(6)
                     + "  " + "1".rjust(4) + "  " + ("%3.1f" % dfs[0]).rjust(7) + "  "
                     + ("%3.1f" % dfs[1]).rjust(7) + "  " + ("%3.2f" % dfs[2]).rjust(6) + "  "
                     + "100.00".rjust(6) + "  " + "0".rjust(8) + "  " + "0.50".rjust(9) + "  "
                     + "0.00".rjust(6) + "  " + "0.00".rjust(6) + "\n")
            fc.write(str(k).rjust(7) + "  %6.3f\n" % o["peak_opt"])
            fcsv.write(",".join(str(v) for v in [k, o["peak_opt"], o["peak"], *o["xy_orig0_est"],
                                                  *o["xy"], *o["xy_opt"], *o["xy_search"], *df,
                                                  *o["q_opt"].ravel(), _pv_at(pv, o["xy_search"])]) + "\n")
    (out / "particles_schema.json").write_text(json.dumps(PARTICLES_SCHEMA, indent=1))


def _pv_at(pv, rc) -> float:
    """-ln p of the search-grid pixel (1-based row, col); nan without search_pval.mrc."""
    if pv is None:
        return float("nan")
    r, c = int(round(rc[0])) - 1, int(round(rc[1])) - 1
    return float(pv[r, c]) if 0 <= r < pv.shape[0] and 0 <= c < pv.shape[1] else float("nan")


PARTICLES_SCHEMA = {
    "file": "particles.csv (one row per optimised candidate; same order as particles.txt/.par/.mrc)",
    "columns": {
        "num_id": "1-based index; row j <-> particles.txt row j <-> j@particles.mrc",
        "score_final": "final optimised, pixelwise flat-fielded SMAP SNR (= particles.txt CC); "
                       "score_kind=flat-fielded normalised CC, score_stage=optimize_xcorr",
        "score_refined": "SNR after grid refinement on the search grid (= list_final.txt col 1)",
        "x_orig0_est, y_orig0_est": "zero-based original-image pixel centre (x=column, y=row), "
                                    "estimated from the optimised shift; register the saved patch "
                                    "(rnf213_workflow_v3.register_patch) for final coordinates",
        "row, col": "1-based original-pixel position passed to optimisation (padded image frame)",
        "row_opt, col_opt": "1-based sub-pixel position after optimisation (padded image frame)",
        "row_search, col_search": "1-based search-grid position (= list_final.txt cols 2-3)",
        "df1_nm, df2_nm, ast_rad": "refined defocus (nm) and astigmatism angle (rad, SMAP sign)",
        "q_opt_ij": "rotation matrix q_opt (MATLAB tableref.q_opt); particles.par holds the "
                    "Frealign angles of q_opt*diag(1,-1,-1); RELION angles = ZYZ of q_opt "
                    "(smap.export)",
        "neg_log_p_search": "-ln p (anisotropic-Gaussian p-value, Zhang et al. 2025) of the "
                            "search-grid pixel from search_pval.mrc; nan if write_mip 0",
    },
    "threshold_operator": "particle selection downstream uses score_final >= cutoff",
    "units": {"pixels": "original sampling (aPerPix)", "defocus": "nm"},
}
