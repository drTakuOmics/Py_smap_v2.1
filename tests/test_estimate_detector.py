import numpy as np
from pathlib import Path

from smap_tools_python import estimate_detector
from smap_tools_python.mrc import write_mrc


def test_estimate_detector_runs(tmp_path: Path):
    # Create a few random micrographs
    files = []
    for i in range(3):
        arr = np.random.randn(32, 32).astype(np.float32)
        fn = tmp_path / f"frame_{i}.mrc"
        write_mrc(fn, arr[None, :, :], 1.0)
        files.append(str(fn))

    filt = estimate_detector(files)
    assert filt.ndim == 1
    assert np.isfinite(filt).all()
    assert len(filt) > 10
