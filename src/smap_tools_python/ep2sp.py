import numpy as np

from .crop_pad import extendj
from .resize_f import resize_F
from .particle_diameter import particle_diameter
from .phase_shift import apply_phase_shifts
from .constants import def_consts


def ep2sp(pv, nm_per_voxel_ep, nm_per_voxel_sp):
    """Convert electrostatic potential to scattering potential.

    Parameters
    ----------
    pv : array_like
        Input electrostatic potential volume.
    nm_per_voxel_ep : float
        Voxel size of the input volume in nanometres.
    nm_per_voxel_sp : float
        Desired voxel size of the output scattering potential in nanometres.

    Returns
    -------
    tuple
        ``(spv, a_per_pix)`` where ``spv`` is the scattering potential volume
        and ``a_per_pix`` is the final voxel size in Ångström.
    """

    pv = np.asarray(pv, dtype=float)

    # ensure cubic volume by padding with the minimum value
    min_dim = min(pv.shape)
    max_dim = max(pv.shape)
    if min_dim < max_dim:
        pv_cube = extendj(pv, (max_dim, max_dim, max_dim), float(pv.min()))
    else:
        pv_cube = pv

    # subtract background value
    pv_cube = pv_cube - pv_cube.flat[0]

    # resize to the desired voxel size
    scale = float(nm_per_voxel_ep) / float(nm_per_voxel_sp)
    spv = resize_F(pv_cube, scale)

    # determine padding size from estimated particle diameter
    pd = particle_diameter(spv, 0.05)
    edge = int(max(2.5 * pd, max(spv.shape)))
    spv = extendj(spv, (edge, edge, edge), 0)

    consts = def_consts()
    a_per_pix = float(nm_per_voxel_sp) * 10.0  # convert nm to Å
    dx = a_per_pix / 1e10  # metres

    spv = spv * consts["IC"] * dx / (2.0 * consts["k"])

    # reverse z-axis to preserve handedness and apply phase shift
    spv = spv[:, :, ::-1]
    spv = apply_phase_shifts(spv, (0, 0, 1)).real.astype(np.float32)

    return spv, a_per_pix


__all__ = ["ep2sp"]
