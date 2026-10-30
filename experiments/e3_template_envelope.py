"""E3: matched-filter efficiency when the template omits the image envelope (MTF, dose, motion)."""
import numpy as np
lam, apix, df0 = 0.019687, 1.5, 3798.0
k = np.linspace(1e-4, 1 / (2 * apix), 4000)
sin2 = np.sin(np.pi * lam * k**2 * df0) ** 2
T2 = k * sin2                         # template power (flat atom spectrum) x ring weight
for B in (50, 100, 150):              # envelope exp(-B k^2/4) on amplitude
    E = np.exp(-B * k**2 / 4)
    eff = np.sum(T2 * E) / np.sqrt(np.sum(T2) * np.sum(T2 * E**2))
    print(f"envelope B={B:3d} A^2: CC efficiency of envelope-free template = {eff:.3f}")
