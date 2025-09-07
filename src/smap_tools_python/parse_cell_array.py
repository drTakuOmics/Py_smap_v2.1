from typing import Iterable, List, Tuple


def parse_cell_array(values: Iterable[str], substr: str) -> Tuple[List[int], List[str]]:
    """Return indices and entries containing ``substr``.

    Mimics the MATLAB ``parseCellArray`` helper by performing a substring
    search over ``values``. Indices are 1-based to ease translation of existing
    MATLAB code.

    Parameters
    ----------
    values : iterable of str
        Collection of strings to search.
    substr : str
        Substring to look for (case sensitive).

    Returns
    -------
    tuple of (list of int, list of str)
        ``inds`` are the 1-based indices where ``substr`` occurs and ``entries``
        are the corresponding strings.
    """

    inds: List[int] = []
    entries: List[str] = []
    for i, val in enumerate(values, start=1):
        if substr in str(val):
            inds.append(i)
            entries.append(val)
    return inds, entries
