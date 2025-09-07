"""Utility for generating timestamp strings.

Mirrors the MATLAB ``ts`` helper which returns strings of the form
``_yymmdd_HHMMSS`` representing the current date and time.
"""

from __future__ import annotations

from datetime import datetime


def ts():
    """Return a timestamp string ``_yymmdd_HHMMSS``."""
    return datetime.now().strftime("_%y%m%d_%H%M%S")
