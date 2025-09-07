
def ahl(val):
    """Plot a horizontal line at ``val`` using matplotlib if available."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for ahl") from exc
    xl = plt.xlim()
    plt.plot([xl[0], xl[1]], [val, val], "r--")
