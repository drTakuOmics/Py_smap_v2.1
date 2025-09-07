import collections.abc as _abc


def get_pref(prefs, key):
    """Retrieve preference ``key`` from ``prefs``.

    Parameters
    ----------
    prefs : mapping or iterable of str
        Source of preferences. If a mapping is provided it is used directly.
        Otherwise it is interpreted as an iterable of ``"name:value"`` strings.
    key : str
        Preference name to retrieve. Use ``"all"`` to obtain a dictionary of
        all preferences.

    Returns
    -------
    str or dict
        The requested preference value or a dict of all preferences when
        ``key`` is ``"all"``. Missing keys return an empty string.
    """
    if isinstance(prefs, _abc.Mapping):
        if key == "all":
            return dict(prefs)
        return prefs.get(key, "")

    out = {}
    for item in prefs:
        if not isinstance(item, str) or ":" not in item:
            continue
        k, v = item.split(":", 1)
        out[k.strip()] = v.strip()
    if key == "all":
        return out
    return out.get(key, "")
