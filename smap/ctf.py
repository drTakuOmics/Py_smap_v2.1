"""CTF and physical constants (@smap/ctf.m, @smap/def_consts.m)."""

from __future__ import annotations

import math

import numpy as np

from .utils import get_ks, rrj


def def_consts() -> dict:
    """@smap/def_consts.m (values kept exactly, including h = 6.636e-34)."""
    c = dict(V=300e3, m_e=9.10938215e-31, h=6.636e-34, q_e=1.602e-19, wl=0.00197e-9,
             Cs=2.7e-3, Cc=2.7e-3, a_i=0.05e-3, dE=0.7, c=2.99792458e8)
    c["hbar"] = c["h"] / (2 * math.pi)
    c["IC"] = (2 * c["m_e"] / c["hbar"] ** 2) * c["q_e"]
    c["k"] = 2 * math.pi / c["wl"]
    c["c_v"] = c["c"]
    return c


def wavelength(V_acc: float) -> float:
    """Relativistic electron wavelength [m] as written in ctf.m."""
    cc = def_consts()
    return cc["h"] / ((2 * cc["m_e"] * cc["q_e"] * V_acc)
                      * (1 + cc["q_e"] * V_acc / (2 * cc["m_e"] * cc["c"] ** 2))) ** 0.5


def ctf(defocus, edge_size, params: dict) -> np.ndarray:
    """@smap/ctf.m (params branch) -> complex CTF stack, shape (N, N, nDf), centred.

    defocus: (nDf, 3) rows of [df1 nm, df2 nm, astig angle rad].
    Uses params Cs, Cc, V_acc, deltaE, a_i, aPerPix, F_abs.
    """
    df = np.atleast_2d(np.asarray(defocus, dtype=np.float64))
    Cs, Cc, V_acc = params["Cs"], params["Cc"], params["V_acc"]
    deltaE, a_i = params["deltaE"], params["a_i"]
    pixel_size = params["aPerPix"]
    F_abs = params.get("F_abs", 0) or 0
    lam = wavelength(V_acc)

    n = int(max(np.atleast_1d(edge_size)))
    dummy = np.ones((n, n), dtype=np.float32)
    k_2d, cp = get_ks(dummy, pixel_size)
    R = rrj(dummy).astype(np.float32) * (n // 2 / 0.5)
    vec = np.arange(-R[0, cp - 1], R[-1, cp - 1] + 0.5, 1.0)[:n]
    X, Y = np.meshgrid(vec, vec)
    Y = -Y
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.arctan(np.abs(Y) / np.abs(X))
    t[np.isnan(t)] = 0
    alpha = np.zeros((n, n))
    q1 = (X >= 0) & (Y >= 0)
    q2 = (X < 0) & (Y >= 0)
    q3 = (X < 0) & (Y < 0)
    q4 = (X >= 0) & (Y < 0)
    alpha[q1] = t[q1]
    alpha[q2] = np.pi - t[q2]
    alpha[q3] = np.pi + t[q3]
    alpha[q4] = 2 * np.pi - t[q4]
    o = (X == 0) & (Y == 0)
    alpha[o] = t[o]

    freq = k_2d * 1e10
    out = np.zeros((n, n, df.shape[0]), dtype=np.complex128)
    w1, w2 = F_abs, 1 - F_abs
    for i in range(df.shape[0]):
        df1, df2, ang = df[i, 0] * 1e-9, df[i, 1] * 1e-9, df[i, 2]
        df_ast = 0.5 * (df1 + df2 + (df1 - df2) * np.cos(2 * (alpha - ang)))
        chi = (np.pi * lam * freq**2) * (df_ast - (Cs * (lam**2) * freq**2 / 2))
        c = (w1 * np.sin(chi) - w2 * np.cos(chi)) + 1j * (-w1 * np.cos(chi) - w2 * np.sin(chi))
        t1 = -((np.pi * lam * freq**2 * Cc * deltaE) / (4 * V_acc * math.sqrt(math.log(2)))) ** 2
        t2 = -((np.pi * Cs * lam**2 * freq**3 - np.pi * df_ast * freq) ** 2) * a_i**2 / math.log(2)
        out[:, :, i] = c * np.exp(t1) * np.exp(t2)
    return out
