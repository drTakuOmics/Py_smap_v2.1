import numpy as np


def assign_jobs(n_indices, n_servers, server_id):
    """Return the 1-based indices assigned to a server.

    Parameters
    ----------
    n_indices : int
        Total number of available indices.
    n_servers : int
        Number of servers sharing the work.
    server_id : int
        1-based identifier of the server requesting its slice.

    Returns
    -------
    numpy.ndarray
        1-based indices allocated to the requested server.  The array is
        empty if ``server_id`` maps to no work.
    """
    if n_servers < 1:
        raise ValueError("n_servers must be positive")
    if server_id < 1 or server_id > n_servers:
        raise ValueError("server_id must be between 1 and n_servers inclusive")

    base_jobs = int(np.ceil(n_indices / n_servers))
    jobs_per_server = np.full(n_servers, base_jobs, dtype=int)
    jobs_with_base = np.cumsum(jobs_per_server)
    start_inds = jobs_with_base - base_jobs + 1
    jobs_with_base[-1] = min(jobs_with_base[-1], n_indices)
    if server_id - 1 >= len(start_inds):
        return np.array([], dtype=int)
    inds = np.arange(start_inds[server_id - 1], jobs_with_base[server_id - 1] + 1)
    return inds[inds <= n_indices]
