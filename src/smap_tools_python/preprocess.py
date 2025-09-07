
def preprocess(obj):
    """Run the standard SMAP preprocessing pipeline.

    The ``obj`` argument is updated in-place by sequentially applying gain
    correction, motion correction, frame summation, and CTF estimation if the
    corresponding steps have not yet been performed (as indicated by flags in
    ``obj.ID``).
    """
    from .gain_corr import gain_corr
    from .sum_frames import sum_frames
    from .run_ctffind import run_ctffind
    try:  # optional dependency
        from .motion_corr import motion_corr  # type: ignore
    except Exception:  # pragma: no cover - motion correction may not be available
        motion_corr = None

    steps = ["GC", "MC", "SF", "CTF"]
    calls = [gain_corr, motion_corr, sum_frames, run_ctffind]
    for flag, func in zip(steps, calls):
        if func is None:
            continue
        if not getattr(obj.ID, flag, False):
            obj = func(obj)
    return obj


__all__ = ["preprocess"]
