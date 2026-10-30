"""E1: accuracy of SMAP's Fourier-slice templates vs analytic projections."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from smap.search_global import _templates
from smap.utils import normalize_rm
rng = np.random.default_rng(1)
N, s = 96, 1.2
pts = rng.uniform(-1, 1, (20000, 3)); pts = pts[np.linalg.norm(pts, axis=1) < 1][:1500] * (N * 0.33)
cp = N // 2 + 1
ax = np.arange(1, N + 1) - cp
Y, X, Z = np.meshgrid(ax, ax, ax, indexing='ij')           # rows=y, cols=x, pages=z
vol = np.zeros((N, N, N))
for (x, y, z) in pts:
    vol += np.exp(-((X - x)**2 + (Y - y)**2 + (Z - z)**2) / (2 * s * s))

def vf(v, pad=1, grid=False):
    M = N * pad
    if grid:  # pre-compensate trilinear kernel: divide by sinc^2(x/M) per axis
        g = np.sinc(ax / M) ** 2
        v = v / (g[:, None, None] * g[None, :, None] * g[None, None, :])
    vp = np.zeros((M, M, M)); o = (M - N) // 2
    vp[o:o+N, o:o+N, o:o+N] = v
    F = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vp)))
    c = M // 2; F[c, c, c] = 0
    return torch.as_tensor(F.real), torch.as_tensor(F.imag), M

def templ(F, R, pad):
    Vr, Vi, M = F
    a = torch.arange(1, N + 1, dtype=torch.float64)
    rr, cc = torch.meshgrid(a, a, indexing='ij')
    if pad == 1:
        return _templates(Vr, Vi, torch.as_tensor(R), cc - cp, rr - cp, cp).numpy()
    # sample padded transform at 2k (same N x N output) -> projection of padded volume, period N
    cpm = M // 2 + 1
    return _templates(Vr, Vi, torch.as_tensor(R), (cc - cp) * pad, (rr - cp) * pad, cpm).numpy() * 1.0

def exact(R):
    a, b = R[:, 0, :], R[:, 1, :]
    out = []
    for i in range(R.shape[0]):
        u = pts @ a[i]; v = pts @ b[i]
        t = np.zeros((N, N))
        for uu, vv in zip(u, v):
            t += np.exp(-((ax[None, :] - uu)**2 + (ax[:, None] - vv)**2) / (2 * s * s))
        out.append(t)
    return np.array(out)

def corr(a, b, hp=None):
    A, B = np.fft.fft2(a), np.fft.fft2(b)
    k = np.sqrt(np.fft.fftfreq(N)[:, None]**2 + np.fft.fftfreq(N)[None, :]**2)
    w = (k > 0) if hp is None else (k > hp)
    num = np.real(np.sum(A[w] * np.conj(B[w])))
    return num / np.sqrt(np.sum(np.abs(A[w])**2) * np.sum(np.abs(B[w])**2))

Rs = []
for _ in range(6):
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    Rs.append(Q * np.sign(np.linalg.det(Q)))
R = normalize_rm(np.array(Rs)); R[0] = np.eye(3)
E = exact(R)
for name, pad, grid in [('v2.1 (no pad, trilinear)', 1, False), ('gridding corr., no pad', 1, True),
                        ('2x pad', 2, False), ('2x pad + gridding corr.', 2, True)]:
    F = vf(vol, pad, grid)
    T = templ(F, R, pad)
    c_all = [corr(T[i], E[i]) for i in range(len(R))]
    c_hi = [corr(T[i], E[i], 0.25) for i in range(len(R))]
    print(f"{name:28s} CC all-k: R=I {c_all[0]:.4f}, random mean {np.mean(c_all[1:]):.4f} | "
          f"CC k>0.25/px: R=I {c_hi[0]:.4f}, random mean {np.mean(c_hi[1:]):.4f}")
