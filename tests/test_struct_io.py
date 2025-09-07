import numpy as np
import pathlib

from smap_tools_python import read_cif_file, read_pdb_file

MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "model"


def test_read_cif_file_basic():
    cif_path = MODEL_DIR / "6ek0_LSU.cif"
    xyz, atom_nums, atom_list, bfac, chains = read_cif_file(str(cif_path), chains=["A"])
    assert xyz.shape[0] == 3
    assert len(atom_nums) == xyz.shape[1]
    assert atom_nums[0] == 1
    assert atom_list[0] == "C5'"
    assert chains[0] == "A"
    assert np.allclose(xyz[:, 0], [-20.637, 30.989, 98.976])
    assert np.isclose(bfac[0], 109.77)


def test_read_pdb_file_basic():
    pdb_path = MODEL_DIR / "5j5b_monster.pdb"
    data = read_pdb_file(str(pdb_path))
    xyz = data["xyz"]
    assert xyz.shape[0] == 3
    assert data["atomList"][0] == "N"
    assert data["chainIDs"][0] == "V"
    assert np.allclose(xyz[:, 0], [29.342, -33.809, -75.427])
    assert np.isclose(data["bFactor"][0], 142.22)
    assert np.isclose(data["occ"][0], 1.00)


def test_read_pdb_file_mmcif():
    cif_path = MODEL_DIR / "6ek0_LSU.cif"
    data = read_pdb_file(str(cif_path))
    xyz = data["xyz"]
    assert xyz.shape[0] == 3
    assert data["atomList"][0] == "C5'"
    assert data["chainIDs"][0] == "L5"
    assert np.allclose(xyz[:, 0], [-20.637, 30.989, 98.976])
    assert np.isclose(data["bFactor"][0], 109.77)
    assert np.isclose(data["occ"][0], 1.0)


def test_read_pdb_file_altloc(tmp_path):
    pdb_content = (
        "ATOM      1  CA AASN A   1      11.104  13.207  10.718  0.50 20.00           C\n"
        "ATOM      1  CA BASN A   1      11.204  13.307  10.818  0.50 25.00           C\n"
        "TER\n"
        "END\n"
    )
    pdb_path = tmp_path / "altloc.pdb"
    pdb_path.write_text(pdb_content)
    data = read_pdb_file(str(pdb_path))
    assert np.allclose(data["xyz"][:, 0], [11.104, 13.207, 10.718])
    assert np.isclose(data["occ"][0], 0.5)
    assert data["altLoc"][0] == "A"
