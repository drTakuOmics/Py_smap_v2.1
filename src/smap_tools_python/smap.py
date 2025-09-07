"""Minimal placeholder class mirroring MATLAB's ``smap`` handle.

The original MATLAB ``smap.m`` class is essentially a thin namespace that
exposes many stand‑alone helper routines as static methods.  The Python port
already provides those helpers as top‑level functions within the
``smap_tools_python`` package.  For compatibility with existing scripts that
expect an object, this module defines :class:`Smap`, which merely stores an
optional ``prefs`` attribute.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Smap:
    """Light‑weight container matching the MATLAB ``smap`` interface.

    Parameters
    ----------
    prefs : Any, optional
        User preference data to attach to the instance.  The class does not
        interpret this value but stores it for convenience.
    """

    prefs: Optional[Any] = None


__all__ = ["Smap"]
