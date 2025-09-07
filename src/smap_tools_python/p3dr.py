import numpy as np


def p3dr(coords):
    """Plot 3-D points as red dots using matplotlib (if available)."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for p3dr") from exc
    pts = np.asarray(coords)
    ax = plt.gca(projection="3d")
    ax.plot(pts[0], pts[1], pts[2], "r.")
