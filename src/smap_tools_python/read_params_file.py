"""Parse SMAP parameter files.

This is a lightweight reimplementation of MATLAB's ``readParamsFile``.  It
supports the three canonical ``function`` types used throughout the toolbox:
``search_global``, ``search_local``, and ``calculate_SP``.  The parser returns a
``(params, fn_type)`` tuple where ``params`` is a dictionary mapping parameter
names to either floats, strings, or lists of floats.

Lines beginning with ``#`` are treated as comments and stripped before parsing.
Values consisting solely of numbers are converted to floats; if multiple
numerical tokens are present, a list of floats is returned.  Otherwise the value
is left as a stripped string.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple


def read_params_file(path: str | Path) -> Tuple[dict[str, Any], str]:
    """Read a ``.par`` file and return parameters with its function type.

    Parameters
    ----------
    path
        Path to the parameter file.

    Returns
    -------
    params, fn_type
        ``params`` is a dictionary of parsed key/value pairs and ``fn_type`` is
        the function type declared on the first non-comment line.
    """
    lines: list[str] = []
    for raw in Path(path).read_text().splitlines():
        clean = raw.split("#", 1)[0].strip()
        if clean:
            lines.append(clean)

    if not lines:
        return {}, ""

    first = lines.pop(0)
    fn_type = first.split()[1] if first.lower().startswith("function") else ""

    params: dict[str, Any] = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        key, values = parts[0], parts[1:]
        if not values:
            params[key] = None
            continue
        try:
            nums = [float(v) for v in values]
        except ValueError:
            params[key] = " ".join(values)
        else:
            params[key] = nums if len(nums) > 1 else nums[0]
    return params, fn_type


__all__ = ["read_params_file"]
