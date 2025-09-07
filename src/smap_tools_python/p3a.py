import numpy as np


def p3a(scale=100):
    """Plot 3-D coordinate axes using matplotlib (if available)."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for p3a") from exc
    origin = np.zeros(3)
    ax = np.eye(3) * scale
    ax = plt.gca(projection="3d")
    ax.plot([0, scale], [0, 0], [0, 0], linewidth=2)
    ax.plot([0, 0], [0, scale], [0, 0], linewidth=2)
    ax.plot([0, 0], [0, 0], [0, scale], linewidth=2)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
