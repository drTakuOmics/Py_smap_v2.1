import numpy as np


def p3d(xyzref, ctu="k.", ax=None):
    """Plot 3-D points in an equal-aspect scatter plot.

    Parameters
    ----------
    xyzref : array_like, shape (N, 3) or (3, N)
        Coordinates to plot.
    ctu : str, optional
        Matplotlib format string specifying colour and marker. Defaults to
        ``"k."`` (black dots).
    ax : :class:`matplotlib.axes.Axes`, optional
        Existing 3-D axis to draw on. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the scatter plot.
    """
    pts = np.asarray(xyzref, dtype=float)
    if pts.shape[0] != 3:
        pts = pts.T
    if ax is None:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[0], pts[1], pts[2], ctu)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True)
    ax.set_aspect("auto")
    import matplotlib.pyplot as plt
    plt.pause(0.01)
    return ax
