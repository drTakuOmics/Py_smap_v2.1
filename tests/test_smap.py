import smap_tools_python as stp


def test_smap_placeholder_holds_prefs():
    obj = stp.Smap()
    assert obj.prefs is None
    obj2 = stp.Smap({'alpha': 1})
    assert obj2.prefs['alpha'] == 1
