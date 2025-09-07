import inspect


def gpu_whos(namespace=None):
    """List GPU arrays in the given namespace.

    The function searches for CuPy or PyTorch CUDA arrays and reports their
    sizes in bytes. It mirrors MATLAB's ``gpuwhos`` utility but returns Python
    structures.

    Parameters
    ----------
    namespace : dict, optional
        Dictionary of variables to inspect. When ``None`` the caller's local
        variables are examined.

    Returns
    -------
    tuple
        ``(summary, info, total_bytes)`` where ``summary`` is a human-readable
        string, ``info`` is a list of dictionaries with ``name``, ``class`` and
        ``size`` keys, and ``total_bytes`` is the aggregate memory usage.
    """
    if namespace is None:
        frame = inspect.currentframe().f_back
        namespace = frame.f_locals

    try:
        import cupy as cp  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        cp = None
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        torch = None

    info = []
    lines = []
    total = 0
    for name, val in namespace.items():
        size = 0
        cls = None
        if cp is not None and isinstance(val, cp.ndarray):
            size = val.nbytes
            cls = "cupy.ndarray"
        elif torch is not None and isinstance(val, torch.Tensor) and val.is_cuda:
            size = val.element_size() * val.nelement()
            cls = "torch.Tensor"
        if size:
            info.append({"name": name, "class": cls, "size": size})
            lines.append(f"{name:20s}\t{size:12d}\t{cls}")
            total += size
    if info:
        lines.append(f"total is {total/1e9:6.4f} GB")
    return "\n".join(lines), info, total

# MATLAB compatibility alias
gpuwhos = gpu_whos

