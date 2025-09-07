from __future__ import annotations

from pathlib import Path
from typing import Dict


def read_output_files(directory: str | Path, pattern: str = "*") -> Dict[str, str]:
    """Read multiple text files from ``directory``.

    Parameters
    ----------
    directory:
        Folder containing the files of interest.
    pattern:
        Glob pattern used to select files.  Defaults to all files.

    Returns
    -------
    dict
        Mapping of file names to their full textual contents.
    """
    directory = Path(directory)
    result: Dict[str, str] = {}
    for path in directory.glob(pattern):
        if path.is_file():
            result[path.name] = path.read_text()
    return result
