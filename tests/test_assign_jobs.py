import numpy as np
from smap_tools_python import assign_jobs


def test_assign_jobs_even_split():
    assert np.array_equal(assign_jobs(10, 2, 1), np.arange(1, 6))
    assert np.array_equal(assign_jobs(10, 2, 2), np.arange(6, 11))


def test_assign_jobs_with_remainder():
    a = assign_jobs(10, 3, 1)
    b = assign_jobs(10, 3, 2)
    c = assign_jobs(10, 3, 3)
    assert np.array_equal(a, np.arange(1, 5))
    assert np.array_equal(b, np.arange(5, 9))
    assert np.array_equal(c, np.arange(9, 11))


def test_assign_jobs_extra_servers():
    assert assign_jobs(3, 5, 4).size == 0
