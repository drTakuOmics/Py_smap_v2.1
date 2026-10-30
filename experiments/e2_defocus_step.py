"""E2: CC retained when the true defocus lies between search planes (sample .par physics)."""
import numpy as np
lam = 0.019687  # A, 300 kV
Cs = 1e-6 * 1e10  # 0.000001 m from sample .par (Cs-corrected), in A
apix = 1.5
k = np.linspace(1e-4, 1 / (2 * apix), 4000)          # radial frequency, A^-1
df0 = 3798.0                                          # A, mean of 4407/3189
def sinchi(df):
    chi = np.pi * lam * k**2 * (df - Cs * lam**2 * k**2 / 2)
    return np.sin(chi)
for B in (0, 50, 100):
    S2 = np.exp(-B * k**2 / 2) * k                     # signal power x 2-D ring weight
    def rho(d):
        a, b = sinchi(df0), sinchi(df0 + d)
        return np.sum(S2 * a * b) / np.sqrt(np.sum(S2 * a * a) * np.sum(S2 * b * b))
    for step in (500, 250, 100):  # A  (50, 25, 10 nm)
        offs = np.linspace(-step / 2, step / 2, 201)
        r = np.array([rho(d) for d in offs])
        print(f"B={B:3d} A^2  df_inc={step/10:4.0f} nm: worst-case CC {r.min():.3f}, mean CC {r.mean():.3f}")
