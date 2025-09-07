import os
import subprocess
from datetime import datetime


def pdb2ep(pdb_path, nm_per_voxel, out_dir, tem_simulator="TEM-simulator"):
    """Generate electrostatic potential from a PDB file using TEM-simulator.

    The function prepares the configuration and auxiliary files required by the
    external ``TEM-simulator`` program.  If the simulator binary is available it
    is invoked; otherwise the configuration files are still written and the
    expected output filename is returned.

    Parameters
    ----------
    pdb_path : str
        Path to the input PDB file.
    nm_per_voxel : float
        Desired voxel size for the output potential in nanometres.
    out_dir : str
        Directory where the configuration files and output map should reside.
    tem_simulator : str, optional
        Executable used to perform the conversion.  Defaults to
        ``"TEM-simulator"``.

    Returns
    -------
    str
        Path to the expected electrostatic potential MRC file.
    """

    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.abspath(pdb_path)
    model_name = os.path.splitext(os.path.basename(pdb_path))[0]

    config_path = os.path.join(out_dir, f"{model_name}_PDB2EP.txt")
    log_line = f"log_file = {model_name}_{datetime.now():%Y%m%dT%H%M%S}.log"
    lines = [
        "=== simulation ===",
        "generate_micrographs = yes",
        log_line,
        "",
        "=== sample ===",
        "diameter = 300",
        "thickness_center = 0",
        "thickness_edge = 0",
        "",
        "=== particle arb ===",
        "source = pdb",
        f"pdb_file_in = {pdb_path}",
        "add_hydrogen = no",
        f"voxel_size = {nm_per_voxel}",
        f"map_file_re_out = {model_name}_EP.mrc",
        "",
        "=== particleset ===",
        "particle_type = arb",
        "particle_coords = file",
        "coord_file_in = arb_location.txt",
        "",
        "=== geometry ===",
        "gen_tilt_data = yes",
        "tilt_mode = single_particle",
        "geom_file_in = arb_rotations.txt",
        "ntilts = 1",
        "theta_start = 0",
        "theta_incr = 0",
        "",
        "=== electronbeam ===",
        "acc_voltage = 300",
        "gen_dose = yes",
        "dose_per_im = 1000",
        "",
        "=== optics ===",
        "cs = 2.7",
        "cond_ap_angle = 0.00",
        "gen_defocus = yes",
        "defocus_nominal = 0.07",
        "",
        "=== detector ===",
        "det_pix_x = 512",
        "det_pix_y = 512",
        "pixel_size = 5",
        "gain = 1",
        "image_file_out = trash.mrc",
    ]
    with open(config_path, "w") as fh:
        fh.write("\n".join(lines))

    # coordinate and rotation files
    with open(os.path.join(out_dir, "arb_location.txt"), "w") as fh:
        fh.write("1  6\n#\t x\t y\t z\t phi\t theta\t psi\n\t0\t0\t0\t0\t0\t0\n")
    with open(os.path.join(out_dir, "arb_rotations.txt"), "w") as fh:
        fh.write("1  3\n#\t phi\t theta\t psi\n\t0\t0\t0\n")

    try:
        subprocess.run([tem_simulator, config_path], check=True)
    except (OSError, FileNotFoundError):
        # simulator not available; proceed without running it
        pass

    return os.path.join(out_dir, f"{model_name}_EP.mrc")


__all__ = ["pdb2ep"]
