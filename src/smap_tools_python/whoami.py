"""Network interface identifier helper."""
from pathlib import Path


def whoami():
    """Return a string containing MAC addresses for all network interfaces.

    The output mirrors the MATLAB ``smap.whoami`` helper by concatenating the
    hexadecimal hardware addresses of each interface, separated by leading
    periods. For example, a host with two interfaces might yield::

        '.001122334455.66778899aabb'

    Returns
    -------
    str
        Concatenated MAC addresses prefixed with a period. Interfaces lacking a
        hardware address are skipped.
    """
    sid = ""
    net_dir = Path("/sys/class/net")
    for addr_file in net_dir.glob("*/address"):
        try:
            addr = addr_file.read_text().strip()
        except OSError:
            continue
        if addr:
            sid += "." + addr.replace(":", "").lower()
    return sid
