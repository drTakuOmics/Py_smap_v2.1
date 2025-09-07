import numpy as np
import pytest

from smap_tools_python import mask_volume, mask_a_volume, parse_excel_file, parseExcelFile


def test_mask_a_volume_alias():
    vol = np.zeros((4, 4, 4), dtype=float)
    vol[1:3, 1:3, 1:3] = 1
    out1, mask1, _ = mask_volume(vol, (1, 1))
    out2, mask2, _ = mask_a_volume(vol, (1, 1))
    assert np.allclose(out1, out2)
    assert np.allclose(mask1, mask2)


def test_parse_excel_file(tmp_path):
    Workbook = pytest.importorskip("openpyxl").Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(["id", "value"])
    ws.append(["", ""])
    ws.append(["a", 1])
    ws.append(["b", 2])
    file = tmp_path / "test.xlsx"
    wb.save(file)

    records = parse_excel_file(file)
    assert len(records) == 2
    assert records[0]["id"] == "a"

    rec = parse_excel_file(file, "b")
    assert rec["id"] == "b"

    rec2 = parseExcelFile(file, "a")
    assert rec2["id"] == "a"
