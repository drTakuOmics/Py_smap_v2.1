#!/usr/bin/env python3
"""Python wrapper for running SMAP modules based on parameter file."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the appropriate ``smappoi_*`` module.

    Parameters
    ----------
    argv: list[str] | None
        Command-line arguments. If ``None`` ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        Exit status code.
    """
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: run_smappoi.py <paramsFile> [boardIndex]")
        return 1
    params_file = Path(argv[0])
    board = argv[1] if len(argv) > 1 else "1"

    fxn_to_run: str | None = None
    with params_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if line.startswith("function"):
                fxn_to_run = line.split()[-1]
                break
    if not fxn_to_run:
        print(f"No function specified in {params_file}")
        return 1

    module_name = f"smap_tools_python.smappoi_{fxn_to_run}"
    module = importlib.import_module(module_name)
    if hasattr(module, "main"):
        return module.main([str(params_file), board])
    # Fallback to calling the function directly if ``main`` is missing
    func = getattr(module, f"smappoi_{fxn_to_run}")
    func(str(params_file), int(board))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
