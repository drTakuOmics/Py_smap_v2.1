import numpy as np

from .templates import templates


def templates_gpu(volume, rotations, dfs=None, pixel_size=1.0, edge_size=None, params=None):
    """Generate projection templates and return them on the GPU when possible.

    This is a lightweight wrapper around :func:`templates` that transfers the
    resulting stack to CuPy when available. If CuPy is not installed, the CPU
    result is returned unchanged.
    """
    out = templates(volume, rotations, dfs, pixel_size, edge_size, params)
    try:
        import cupy as cp  # type: ignore

        return cp.asarray(out)
    except Exception:  # pragma: no cover - CuPy may be absent
        return out


def templates_half_gpu(volume, rotations, dfs=None, pixel_size=1.0, edge_size=None, params=None):
    """Generate templates at full resolution and downsample by two.

    The computation is delegated to :func:`templates_gpu`; the returned stack is
    decimated by taking every other pixel in both dimensions. This mimics the
    MATLAB ``templates_half_gpu`` helper that produced half-sized templates for
    coarse searches.
    """
    full = templates_gpu(volume, rotations, dfs, pixel_size, edge_size, params)
    return full[::2, ::2]


__all__ = ["templates_gpu", "templates_half_gpu"]
