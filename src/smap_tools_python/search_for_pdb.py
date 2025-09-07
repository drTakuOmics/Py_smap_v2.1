from __future__ import annotations

from pathlib import Path
from typing import List


def search_for_pdb(directory: str | Path, pdb_id: str) -> List[Path]:
    """Search ``directory`` for PDB files containing ``pdb_id`` in the name.

    Parameters
    ----------
    directory:
        Directory to search.
    pdb_id:
        Substring of the desired PDB identifier.  The search is case-insensitive.

    Returns
    -------
    list of :class:`pathlib.Path`
        Paths to matching PDB files.
    """
    directory = Path(directory)
    pattern = pdb_id.lower()
    matches: List[Path] = []
    for path in directory.glob("*.pdb"):
        if pattern in path.name.lower():
            matches.append(path)
    return matches
