import re
from smap_tools_python import ts


def test_ts_format():
    stamp = ts()
    assert re.fullmatch(r"_\d{6}_\d{6}", stamp)
