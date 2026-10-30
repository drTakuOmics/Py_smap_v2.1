"""Provenance record written into search_status.json (input hashes, code, settings)."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

from . import __version__


def sha256(path, chunk=1 << 22) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def code_commit() -> str | None:
    root = Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        commit = out.stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "smap"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{commit}{'+modified' if dirty else ''}" if commit else None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(params: dict) -> dict:
    import numpy
    import scipy
    try:
        import torch
        tv, cuda = torch.__version__, (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:  # pragma: no cover
        tv, cuda = None, None
    files = {k: params.get(k) for k in ("imageFile", "structureFile", "modelFile", "rotationsFile")
             if params.get(k)}
    return dict(
        smap_port_version=__version__, code_commit=code_commit(),
        python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
        torch=tv, gpu=cuda, run_settings=params.get("_run_settings"),
        params_file=params.get("_params_file"), params_text=params.get("_params_text"),
        input_sha256={k: {"path": v, "sha256": sha256(v)} for k, v in files.items()},
        upstream="SMAP v2.1, J. Peter Rickgauer (HHMI Janelia), MIT License",
    )
