import os
from pathlib import Path

from smap_tools_python import (
    parse_input_file,
    read_output_files,
    search_for_pdb,
    get_dataset,
    put_dataset,
    get_datasets,
)


def test_parse_input_file(tmp_path):
    content = """
# comment
alpha = 1
beta: two
gamma three
% ignore
"""
    file = tmp_path / "input.txt"
    file.write_text(content)
    result = parse_input_file(file)
    assert result == {"alpha": 1, "beta": "two", "gamma": "three"}


def test_read_output_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.log").write_text("world")
    result = read_output_files(tmp_path, "*.txt")
    assert result == {"a.txt": "hello"}


def test_search_for_pdb(tmp_path):
    (tmp_path / "1abc.pdb").write_text("PDB DATA")
    (tmp_path / "2xyz.pdb").write_text("PDB DATA")
    matches = search_for_pdb(tmp_path, "1a")
    assert len(matches) == 1
    assert matches[0].name == "1abc.pdb"


def test_dataset_roundtrip(tmp_path):
    data = {"a": 1, "b": 2}
    file = tmp_path / "dataset.json"
    put_dataset(file, data)
    assert file.exists()
    loaded = get_dataset(file)
    assert loaded == data
    all_datasets = get_datasets(tmp_path)
    assert all_datasets == {"dataset": data}
