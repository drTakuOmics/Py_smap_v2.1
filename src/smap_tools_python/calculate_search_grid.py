import numpy as np
from .rotate import normalize_rotation_matrices


_def_psi_max = 358


def calculate_search_grid(symmetry_symbol, angular_step_size, psi_step, psi_max=_def_psi_max):
    """Generate Euler angles and rotation matrices for a symmetry-limited grid.

    Parameters
    ----------
    symmetry_symbol : str
        Symmetry specifier (e.g., ``'C1'``, ``'D2'``, ``'T'``, ``'O'``, ``'I'``).
    angular_step_size : float
        Sampling step for polar angles in degrees.
    psi_step : float
        Sampling step for ``psi`` angle in degrees.
    psi_max : float, optional
        Maximum ``psi`` angle. Defaults to 358 to mimic MATLAB behavior.

    Returns
    -------
    tuple
        ``(RM, EA)`` where ``RM`` are rotation matrices of shape ``(3,3,N)`` and
        ``EA`` are Euler angles in degrees as a ``3×N`` array ``[phi; theta; psi]``.
    """
    symmetry_symbol = symmetry_symbol.upper()
    symmetry_number = 1
    if len(symmetry_symbol) > 1:
        try:
            symmetry_number = int(symmetry_symbol[1:])
        except ValueError:
            pass
    sym_type = symmetry_symbol[0]
    phi_start = 0

    if sym_type == "C":
        phi_max = 360.0 / symmetry_number
        theta_max = 90.0
        test_mirror = True
    elif sym_type == "D":
        phi_max = 360.0 / symmetry_number
        theta_max = 90.0
        test_mirror = False
    elif sym_type == "T":
        phi_max = 180.0
        theta_max = 54.7
        test_mirror = False
    elif sym_type == "O":
        phi_max = 90.0
        theta_max = 54.7
        test_mirror = False
    elif sym_type == "I":
        phi_max = 180.0
        theta_max = 31.7
        test_mirror = False
    else:
        raise ValueError("Unrecognized symmetry symbol")

    psi_vector = np.arange(0, psi_max + 1e-3, psi_step)

    theta_step = theta_max / np.floor(theta_max / angular_step_size + 0.5)
    theta_values = np.arange(0, theta_max + theta_step / 2 + 1e-3, theta_step)

    euler_angles = []
    for theta in theta_values:
        if theta in (0, 180):
            phi_step = phi_max
        else:
            phi_step = angular_step_size / np.sin(np.deg2rad(theta))
            if phi_step > phi_max:
                phi_step = phi_max
            phi_step = phi_max / np.floor(phi_max / phi_step + 0.5)
        phi_values = np.arange(0, phi_max + 1e-3, phi_step)
        for phi in phi_values:
            for psi in psi_vector:
                euler_angles.append([phi + phi_start, theta, psi])
    EA = np.array(euler_angles, dtype=float).T

    phi = np.deg2rad(EA[0])
    theta = np.deg2rad(EA[1])
    psi = np.deg2rad(EA[2])

    n = EA.shape[1]
    RM = np.zeros((3, 3, n), dtype=float)
    c_phi, c_theta, c_psi = np.cos(phi), np.cos(theta), np.cos(psi)
    s_phi, s_theta, s_psi = np.sin(phi), np.sin(theta), np.sin(psi)
    RM[0, 0, :] = c_phi * c_theta * c_psi - s_phi * s_psi
    RM[0, 1, :] = s_phi * c_theta * c_psi + c_phi * s_psi
    RM[0, 2, :] = -s_theta * c_psi
    RM[1, 0, :] = -c_phi * c_theta * s_psi - s_phi * c_psi
    RM[1, 1, :] = -s_phi * c_theta * s_psi + c_phi * c_psi
    RM[1, 2, :] = s_theta * s_psi
    RM[2, 0, :] = s_theta * c_phi
    RM[2, 1, :] = s_theta * s_phi
    RM[2, 2, :] = c_theta

    RM = normalize_rotation_matrices(np.transpose(RM,(2,0,1))).transpose(1,2,0)

    if test_mirror:
        R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        RM_mirror = np.zeros_like(RM)
        for j in range(RM.shape[2]):
            RM_mirror[:, :, j] = RM[:, :, j] @ R_flip
        RM_mirror = normalize_rotation_matrices(np.transpose(RM_mirror,(2,0,1))).transpose(1,2,0)
        RM = np.concatenate([RM, RM_mirror], axis=2)

    return RM, EA
