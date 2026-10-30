#!/usr/bin/env python3
"""GPU smoke test (~1-2 min): the synthetic end-to-end case on CUDA vs CPU.

Runs search -> combine -> refinement -> optimisation on the same synthetic tile on both
devices and compares the outputs with smap.compare (default tolerances). Prints the GPU
name and the per-orientation search time. Exit 0 = GPU path agrees with CPU.

usage (from the Py_smap_v2.1 folder):  python tools/gpu_smoke.py [--device cuda:0]
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tests")]
from smap.compare import compare_outputs, format_report  # noqa: E402
from smap.search_global import main  # noqa: E402
from synthetic import make_tile, write_par  # noqa: E402


def run(d: Path, device: str, out: str) -> float:
    p = write_par(d, d / "img.mrc", n_cores=1, extra="optThr 5.0\ninc_degrees 2\n")
    t0 = time.time()
    rc = main([str(p), "--device", device, "--batch-size", "16", "--output-dir", str(d / out)])
    if rc != 0:
        raise SystemExit(f"run on {device} failed (exit {rc})")
    return time.time() - t0


def cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False: install a CUDA build of PyTorch "
              f"(torch {torch.__version__}, CUDA {torch.version.cuda})")
        return 1
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(a.device)}")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        make_tile(d, 1500, 50, 95, snr=14.0, phase_scale=0.002)
        t_cpu = run(d, "cpu", "cpu")
        t_gpu = run(d, a.device, "gpu")
        rep = compare_outputs(d / "cpu", d / "gpu")
        print(format_report(rep))
        print(f"wall time: CPU {t_cpu:.1f} s, GPU {t_gpu:.1f} s (tiny case; dominated by setup)")
        print("GPU SMOKE TEST PASSED" if rep["all_ok"] else "GPU SMOKE TEST: DIFFERENCES ABOVE TOLERANCE")
        return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(cli())
