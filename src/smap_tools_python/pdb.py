from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def _atom_record(atom):
    """Return record type (ATOM/HETATM) for an atom."""
    hetero_flag = atom.get_parent().id[0]
    return "HETATM" if hetero_flag.strip() else "ATOM"


def read_pdb_file(filename: str) -> Dict[str, Any]:
    """Read a PDB or mmCIF file using :mod:`Bio.PDB` utilities.

    Parameters
    ----------
    filename:
        Path to a ``.pdb`` or ``.cif``/``.mmcif`` file.

    Returns
    -------
    dict
        Dictionary containing coordinates, atom annotations and assorted
        metadata mirroring the previous MATLAB helper.
    """
    path = Path(filename)
    split_lines = path.read_text().splitlines()

    is_cif = path.suffix.lower() in {".cif", ".mmcif"}
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, filename)
    metadata = MMCIF2Dict(filename) if is_cif else parser.get_header()

    atoms = list(structure.get_atoms())
    coords = np.asarray([a.coord for a in atoms], dtype=float)
    xyz = coords.T if atoms else np.empty((3, 0))

    atom_list = [a.get_name().strip() for a in atoms]
    chain_ids = [a.get_parent().get_parent().id.strip() for a in atoms]
    res_name = [a.get_parent().get_resname().strip() for a in atoms]
    b_factor = [a.get_bfactor() for a in atoms]
    occ = [a.get_occupancy() if a.get_occupancy() is not None else np.nan for a in atoms]
    alt_loc = [a.get_altloc().strip() for a in atoms]
    element = [a.element.strip() if a.element else "" for a in atoms]

    pdb_data = {
        "atomType": [_atom_record(a) for a in atoms],
        "atomNum": [str(a.serial_number) for a in atoms],
        "atomName": atom_list,
        "resName": res_name,
        "chain": chain_ids,
        "resNum": [str(a.get_parent().id[1]) for a in atoms],
        "filler": [""] * len(atoms),
        "X": [f"{c[0]:8.3f}" for c in coords],
        "Y": [f"{c[1]:8.3f}" for c in coords],
        "Z": [f"{c[2]:8.3f}" for c in coords],
        "occ": ["" if np.isnan(o) else f"{o:6.2f}" for o in occ],
        "b_factor": [f"{b:6.2f}" for b in b_factor],
        "comment": [""] * len(atoms),
        "element": element,
        "altLoc": alt_loc,
        "line": split_lines,
    }

    header_inds: List[int] = []
    header: List[str] = []
    inds: List[int] = []
    line_type: List[int] = []
    for idx, line in enumerate(split_lines, start=1):
        if line.startswith(("ATOM", "HETATM", "TER")):
            inds.append(idx)
            line_type.append(1)
        else:
            header_inds.append(idx)
            header.append(line)
            line_type.append(0)

    return {
        "xyz": xyz,
        "atomList": atom_list,
        "bFactor": np.asarray(b_factor, dtype=float),
        "chainIDs": chain_ids,
        "resName": res_name,
        "occ": np.asarray(occ, dtype=float),
        "altLoc": alt_loc,
        "element": element,
        "inds": inds,
        "header": header,
        "headerInds": header_inds,
        "lineType": line_type,
        "PDBdata": pdb_data,
        "splitLines": split_lines,
        "metadata": metadata,
    }
