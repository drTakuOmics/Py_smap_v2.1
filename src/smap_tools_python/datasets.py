from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


def get_dataset(path: str | Path) -> Dict[str, Any]:
    """Load a dataset stored as JSON."""
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def put_dataset(path: str | Path, data: Mapping[str, Any]) -> None:
    """Save ``data`` to ``path`` as JSON."""
    path = Path(path)
    with path.open("w") as f:
        json.dump(data, f)


def get_datasets(directory: str | Path, pattern: str = "*.json") -> Dict[str, Dict[str, Any]]:
    """Load all dataset JSON files in ``directory`` matching ``pattern``."""
    directory = Path(directory)
    return {
        p.stem: get_dataset(p)
        for p in directory.glob(pattern)
        if p.is_file()
    }

# MATLAB-style aliases
getDataset = get_dataset
putDataset = put_dataset
getDatasets = get_datasets
