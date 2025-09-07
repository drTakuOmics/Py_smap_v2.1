
def avl(val):
    """Plot a vertical line at ``val`` using matplotlib if available."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for avl") from exc
    yl = plt.ylim()
    plt.plot([val, val], [yl[0], yl[1]], "r--")
