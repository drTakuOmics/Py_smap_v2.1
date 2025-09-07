import numpy as np
from .constants import def_consts
from .ks import get_ks


def ctf(df, edge_size, params):
    """Generate a contrast transfer function (CTF) image.

    Parameters
    ----------
    df : array_like, shape (n,3)
        Defocus parameters ``(df1, df2, alpha_ast)`` in nanometers and radians.
    edge_size : int
        Size of the (square) output CTF image.
    params : dict
        Microscope parameters with keys ``Cs``, ``Cc``, ``V_acc``, ``deltaE``,
        ``a_i`` and ``aPerPix``. Optional key ``F_abs`` specifies the amplitude
        contrast fraction (default 0).

    Returns
    -------
    ndarray
        CTF image of shape ``(edge_size, edge_size)`` or ``(..., n)`` if multiple
        defocus sets are provided.
    """
    df = np.atleast_2d(np.asarray(df, dtype=float))
    n_ctf = df.shape[0]

    Cs = params["Cs"]
    Cc = params["Cc"]
    V_acc = params["V_acc"]
    deltaE = params["deltaE"]
    a_i = params["a_i"]
    pixel_size = params["aPerPix"]
    F_abs = params.get("F_abs", 0.0)

    cc = def_consts()
    h = cc["h"]
    e = cc["q_e"]
    c_v = cc["c"]
    m_e = cc["m_e"]
    lam = h / np.sqrt(2 * m_e * e * V_acc * (1 + e * V_acc / (2 * m_e * c_v ** 2)))

    edge_size = int(np.max(np.atleast_1d(edge_size)))
    dummy = np.ones((edge_size, edge_size), dtype=np.float32)
    k_2d, center = get_ks(dummy, pixel_size)

    x = np.arange(-center, edge_size - center)
    X, Y = np.meshgrid(x, x)
    Y = -Y
    alpha_g = np.arctan2(Y, X)
    alpha_g[alpha_g < 0] += 2 * np.pi

    freq = k_2d * 1e10
    CTF = np.zeros((edge_size, edge_size, n_ctf), dtype=np.complex64)
    for i in range(n_ctf):
        df1, df2, alpha_ast = df[i]
        df1 *= 1e-9
        df2 *= 1e-9
        ddf = df1 - df2
        df_ast = 0.5 * (df1 + df2 + ddf * np.cos(2 * (alpha_g - alpha_ast)))
        chi = (np.pi * lam * freq**2) * (df_ast - (Cs * (lam**2) * (freq**2) / 2))
        w1 = F_abs
        w2 = 1 - w1
        ctf_temp = (w1 * np.sin(chi) - w2 * np.cos(chi)) + 1j * (
            -w1 * np.cos(chi) - w2 * np.sin(chi)
        )
        term_one = -(
            (
                np.pi
                * lam
                * (freq**2)
                * Cc
                * deltaE
                / (4 * V_acc * np.sqrt(np.log(2)))
            )
            ** 2
        )
        term_two = -(
            (np.pi * Cs * (lam**2) * (freq**3) - np.pi * df_ast * freq) ** 2
        ) * (a_i**2) / np.log(2)
        CTF[:, :, i] = ctf_temp * np.exp(term_one) * np.exp(term_two)

    return CTF.squeeze()
