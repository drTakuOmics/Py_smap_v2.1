from pathlib import Path

from smap_tools_python import smappoi_search_global


def test_smappoi_search_global_stdout(capsys):
    smappoi_search_global(Path('sample_search.par'))
    captured = capsys.readouterr()
    expected = Path('tests/fixtures/global_stdout.txt').read_text().strip()
    assert captured.out.strip() == expected
