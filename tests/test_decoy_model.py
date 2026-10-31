"""tools/make_decoy_model.py: atom-count/shape matched trimming, output readable by SMAP."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from smap.structure import read_cif_file

spec = importlib.util.spec_from_file_location(
    "make_decoy_model", Path(__file__).resolve().parents[1] / "tools" / "make_decoy_model.py")
mdm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdm)

COLS = ("group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id label_entity_id "
        "label_seq_id pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv pdbx_formal_charge "
        "auth_seq_id auth_comp_id auth_asym_id auth_atom_id pdbx_PDB_model_num").split()


def write_cif(path, centres, atoms_per_res, chain="A", seed=0):
    rng = np.random.default_rng(seed)
    lines = ["data_x", "loop_"] + [f"_atom_site.{c}" for c in COLS]
    k = 0
    for r, c in enumerate(centres, 1):
        for a in range(atoms_per_res):
            k += 1
            x, y, z = c + rng.normal(0, 1.0, 3)
            el = "C" if a else "N"
            lines.append(f"ATOM {k} {el} {el}{a} . ALA {chain} 1 {r} ? {x:.3f} {y:.3f} {z:.3f} 1.00 30.00 ? "
                         f"{r} ALA {chain} {el}{a} 1")
    lines.append("#")
    Path(path).write_text("\n".join(lines) + "\n")


def test_decoy_matched_to_target(tmp_path):
    rng = np.random.default_rng(1)
    target = rng.normal(0, 1, (300, 3)) * [40, 20, 10]          # elongated target
    decoy = rng.normal(0, 1, (900, 3)) * [35, 30, 25]           # larger, rounder decoy
    write_cif(tmp_path / "t.cif", target, 8)
    write_cif(tmp_path / "d.cif", decoy, 5, chain="B")           # truncated side chains
    mdm.main(["--decoy", str(tmp_path / "d.cif"), "--match", str(tmp_path / "t.cif"),
              "--out", str(tmp_path / "out" / "dm")])
    rep = json.loads((tmp_path / "out" / "dm.json").read_text())
    n_t, n_d = rep["target"]["atoms"], rep["decoy_after"]["atoms"]
    assert n_t == 2400 and n_t <= n_d < n_t + 5                  # whole residues of 5 atoms
    assert any("side chains truncated" in w for w in rep["warnings"])
    sp_d = rep["decoy_after"]["pc_span_A"]
    assert sp_d[0] / sp_d[2] > 1.8                               # proportions follow the target
    at = read_cif_file(tmp_path / "out" / "dm.cif")
    assert at.xyz.shape[1] == n_d and set(at.chain_ids) == {"B"}
