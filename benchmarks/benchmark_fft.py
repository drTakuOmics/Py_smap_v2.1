"""Simple benchmarks for the FFT backends.

Run this module as a script to compare CPU and GPU implementations over a
range of transform sizes.  By default both a modest 256×256 and a large 4096×4096
roundtrip are performed as the crossover in efficiency between the two
backends depends strongly on the FFT size.
"""

from __future__ import annotations

import time
import numpy as np

from emClarity_FFT import has_gpu
from smap_tools_python import ftj, iftj
from smap_tools_python.gpu_fft import gpu_ftj, gpu_iftj


def benchmark(sizes: tuple[int, ...] = (256, 4096)) -> None:
    for size in sizes:
        print(f"\nBenchmarking {size}x{size} FFT")
        arr = np.random.rand(size, size)
        t0 = time.perf_counter()
        iftj(ftj(arr))
        cpu_time = time.perf_counter() - t0
        print(f"CPU roundtrip: {cpu_time:.4f}s")

        if has_gpu():  # pragma: no cover - optional dependency
            t0 = time.perf_counter()
            gpu_iftj(gpu_ftj(arr))
            gpu_time = time.perf_counter() - t0
            print(f"GPU roundtrip: {gpu_time:.4f}s")
        else:
            print("GPU backend not available")


if __name__ == "__main__":  # pragma: no cover - manual benchmark
    benchmark()
