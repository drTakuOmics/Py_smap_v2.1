from smap_tools_python import parse_cell_array


def test_parse_cell_array_returns_indices_and_entries():
    data = ['alpha', 'beta', 'alphabet']
    idx, entries = parse_cell_array(data, 'alpha')
    assert idx == [1, 3]
    assert entries == ['alpha', 'alphabet']


def test_parse_cell_array_handles_missing():
    idx, entries = parse_cell_array(['foo', 'bar'], 'baz')
    assert idx == []
    assert entries == []
