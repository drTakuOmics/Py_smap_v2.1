from __future__ import annotations

import numpy as np
from typing import Dict, List, Any


def read_pdb_file(filename: str) -> Dict[str, Any]:
    """Read an ASCII ``.pdb`` file.

    This is a lightweight parser that extracts atom records and a handful of
    commonly used fields.  The return value mirrors MATLAB's ``read_pdb_file``
    helper and provides a dictionary containing atom coordinates and assorted
    metadata.
    """
    with open(filename, "r") as fh:
        split_lines = fh.read().splitlines()

    atom_list: List[str] = []
    b_factor: List[float] = []
    chain_ids: List[str] = []
    res_name: List[str] = []
    occ: List[float] = []
    inds: List[int] = []
    header: List[str] = []
    header_inds: List[int] = []
    line_type: List[int] = []
    xyz: List[List[float]] = []

    pdb_data = {
        "atomType": [],
        "atomNum": [],
        "atomName": [],
        "resName": [],
        "chain": [],
        "resNum": [],
        "X": [],
        "Y": [],
        "Z": [],
        "b_factor": [],
        "comment": [],
        "filler": [],
        "occ": [],
        "line": split_lines,
    }

    for idx, line in enumerate(split_lines, start=1):
        record = line[0:6].strip()
        if record in {"ATOM", "HETATM", "TER"}:
            try:
                pdb_data["atomType"].append(line[0:6])
                pdb_data["atomNum"].append(line[6:11])
                pdb_data["atomName"].append(line[12:16])
                pdb_data["resName"].append(line[17:20])
                pdb_data["chain"].append(line[21:22])
                pdb_data["resNum"].append(line[22:26])
                pdb_data["filler"].append(line[27:30])
                pdb_data["X"].append(line[30:38])
                pdb_data["Y"].append(line[38:46])
                pdb_data["Z"].append(line[46:54])
                pdb_data["occ"].append(line[54:60])
                pdb_data["b_factor"].append(line[60:66])
                pdb_data["comment"].append(line[66:])

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                xyz.append([x, y, z])
                atom_list.append(line[12:16].strip())
                chain_ids.append(line[21:22].strip())
                res_name.append(line[17:20].strip())
                b_factor.append(float(line[60:66]))
                occ.append(float(line[54:60]))
                inds.append(idx)
                line_type.append(1)
            except ValueError:
                header.append(line)
                header_inds.append(idx)
                line_type.append(0)
        else:
            header.append(line)
            header_inds.append(idx)
            line_type.append(0)

    xyz_arr = np.asarray(xyz, dtype=float).T if xyz else np.empty((3, 0))
    b_factor_arr = np.asarray(b_factor, dtype=float)
    occ_arr = np.asarray(occ, dtype=float)

    return {
        "xyz": xyz_arr,
        "atomList": atom_list,
        "bFactor": b_factor_arr,
        "chainIDs": chain_ids,
        "resName": res_name,
        "occ": occ_arr,
        "inds": inds,
        "header": header,
        "headerInds": header_inds,
        "lineType": line_type,
        "PDBdata": pdb_data,
        "splitLines": split_lines,
    }
