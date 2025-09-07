"""MATLAB-style ``dw`` helper for writing binary volumes."""

from .dat_io import write_dat


def dw(array, filename):
    """Write ``array`` to ``filename`` as a ``.dat`` with header.

    This is a thin wrapper around :func:`write_dat` to mirror MATLAB's
    ``dw`` utility.  The data are stored as 32-bit floats and a companion
    ``.hdr`` file records the array shape separated by tabs.
    """

    write_dat(array, filename)
