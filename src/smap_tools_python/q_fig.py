"""Export a figure with consistent fonts and size.

This mirrors the MATLAB ``qFig`` utility used throughout SMAP.  The current
figure is resized, tick styling is applied, and the result is saved as an EPS
file.  Only a small subset of the original functionality is implemented: font
replacement is limited to setting the ``Arial`` family for all text objects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _set_font(obj, font: str, size: float) -> None:
    if hasattr(obj, "set_fontname"):
        try:
            obj.set_fontname(font)
        except Exception:
            pass
    if hasattr(obj, "set_fontsize"):
        obj.set_fontsize(size)


def q_fig(filename: str | Path, size: Iterable[float] = (4, 3), fontsize: float = 9) -> None:
    """Save the current figure to ``filename`` using a clean style.

    The import of :mod:`matplotlib` is deferred so that the rest of the package
    remains importable even when the dependency is absent.  A clear ``ImportError``
    is raised if the function is invoked without ``matplotlib`` installed.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - simple error propagation
        raise ImportError("matplotlib is required for q_fig") from exc

    fig = plt.gcf()
    fig.set_size_inches(*size)

    for ax in fig.get_axes():
        ax.tick_params(direction="out", length=4, width=1, top=False, right=False)
        ax.minorticks_on()
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            _set_font(item, "Arial", fontsize)

    Path(filename).with_suffix(".eps")
    fig.savefig(str(filename), format="eps", dpi=300)


__all__ = ["q_fig"]
