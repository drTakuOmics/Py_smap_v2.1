import re
from smap_tools_python import whoami


def test_whoami_format():
    sid = whoami()
    assert sid.startswith(".")
    assert re.fullmatch(r"(\.[0-9a-f]{12})+", sid), sid
