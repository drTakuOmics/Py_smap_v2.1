from datetime import datetime, timedelta
from .zp import zp

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def fov_to_num(fov_ref):
    """Convert a field-of-view reference string to a numeric identifier."""
    if isinstance(fov_ref, (list, tuple)):
        fov_ref = fov_ref[0]
    date_part, letter_part, num_part = fov_ref.split("_")
    the_year = int("20" + date_part[4:6])
    the_month = int(date_part[0:2])
    the_date = int(date_part[2:4])
    baseline = datetime(2014, 1, 1, 12, 0, 0)
    target = datetime(the_year, the_month, the_date, 12, 0, 0)
    days = (target - baseline).days
    temp = list("0" * 9)
    temp[0:4] = list(zp(days, 4))
    idx = ALPHABET.index(letter_part.upper()) + 1
    temp[4:6] = list(zp(idx, 2))
    temp[6:9] = list(zp(int(num_part), 3))
    return int("".join(temp))


def num_to_fov(numref):
    """Convert a numeric identifier back to a field-of-view reference string."""
    num_str = str(int(numref)).zfill(9)
    days = int(num_str[0:4])
    baseline = datetime(2014, 1, 1, 12, 0, 0)
    target = baseline + timedelta(days=days)
    date_part = target.strftime("%m%d%y")
    letter_idx = int(num_str[4:6])
    letter = ALPHABET[letter_idx - 1]
    num_part = zp(int(num_str[6:9]), 4)
    return f"{date_part}_{letter}_{num_part}"
