import numpy as np
from typing import Sequence, Tuple, List, Optional


def read_cif_file(filename: str, chains: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[str]]:
    """Parse a minimal subset of a ``.cif`` file.

    Parameters
    ----------
    filename:
        Path to a CIF file.
    chains:
        Optional iterable of chain identifiers to keep.  All chains are
        returned when ``None``.

    Returns
    -------
    xyz, atom_nums, atom_list, b_factor, chain_ids
        ``xyz`` is a 3×N array of coordinates in Å and the other outputs are
        per-atom properties mirroring the MATLAB ``read_cif_file`` helper.
    """
    atom_nums: List[int] = []
    atom_list: List[str] = []
    chain_ids: List[str] = []
    b_factor: List[float] = []
    coords: List[List[float]] = []

    chain_set = set(chains) if chains else None

    with open(filename, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            tokens = line.split()
            if len(tokens) < 15:
                # Not enough columns to parse
                continue
            # Chain identifier appears in different columns depending on the
            # originating software.  Try a few possibilities, falling back to
            # the label asym id (token 6) used by PyMOL exports in this repo.
            chain = ""
            if len(tokens) >= 7:
                chain = tokens[6]
            if chain in {"", "."}:
                if len(tokens) >= 19:
                    chain = tokens[18]
                elif len(tokens) >= 17:
                    chain = tokens[16]
            if chain_set and chain not in chain_set:
                continue
            atom_nums.append(int(tokens[1]))
            atom_list.append(tokens[3])
            chain_ids.append(chain)
            coords.append([float(tokens[10]), float(tokens[11]), float(tokens[12])])
            b_factor.append(float(tokens[14]))

    xyz = np.asarray(coords, dtype=float).T if coords else np.empty((3, 0))
    atom_nums_arr = np.asarray(atom_nums, dtype=int)
    b_factor_arr = np.asarray(b_factor, dtype=float)
    return xyz, atom_nums_arr, atom_list, b_factor_arr, chain_ids
