from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Dict, Any


def parse_input_file(path: str | Path) -> Dict[str, Any]:
    """Parse a simple key/value text file.

    Each non-empty, non-comment line of ``path`` is interpreted as a key/value
    pair.  Lines may separate the key and value with whitespace, ``=`` or
    ``:``.  Values are converted using :func:`ast.literal_eval` when possible to
    recover numeric types and other Python literals; otherwise they are kept as
    strings.

    Parameters
    ----------
    path:
        Path to the text file to parse.

    Returns
    -------
    dict
        Mapping of parsed keys to values.
    """
    path = Path(path)
    result: Dict[str, Any] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                key, value = parts
            key = key.strip()
            value = value.strip()
            try:
                value = literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            result[key] = value
    return result
