"""Regenerate smap/_sf_table.py from the original MATLAB parameterizeSF.m.

usage: python tools/gen_sf_table.py path/to/@smap/parameterizeSF.m
"""
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
pairs = re.findall(r"case\s+'(\w+)'\s*\n\s*g\s*=\s*\[([^\]]+)\]", src)
out = ['"""Scattering-factor parameters (Peng et al. 5-Gaussian fit).', '',
       'Auto-generated from SMAP v2.1 @smap/parameterizeSF.m by tools/gen_sf_table.py.',
       'Keys are matched case-sensitively, exactly as in the MATLAB switch statement.',
       '"""', '', 'SF_TABLE = {']
for k, v in pairs:
    nums = [float(x) for x in v.split()]
    assert len(nums) == 10, (k, nums)
    out.append(f"    {k!r}: {nums!r},")
out.append('}')
Path(__file__).resolve().parents[1].joinpath('smap', '_sf_table.py').write_text('\n'.join(out) + '\n')
print(f"wrote {len(pairs)} elements")
