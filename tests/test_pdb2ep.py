import os
from smap_tools_python import pdb2ep


def test_pdb2ep_writes_files(tmp_path):
    pdb_file = tmp_path / "model.pdb"
    pdb_file.write_text("HEADER test\nEND\n")
    out_dir = tmp_path / "out"
    ep_path = pdb2ep(str(pdb_file), 1.23, str(out_dir), tem_simulator="true")
    cfg = out_dir / f"{pdb_file.stem}_PDB2EP.txt"
    assert cfg.exists()
    assert (out_dir / "arb_location.txt").exists()
    assert (out_dir / "arb_rotations.txt").exists()
    text = cfg.read_text()
    assert f"pdb_file_in = {pdb_file}" in text
    assert ep_path == str(out_dir / f"{pdb_file.stem}_EP.mrc")
