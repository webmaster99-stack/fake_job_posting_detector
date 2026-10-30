"""Fake job posting detector."""

import os

__version__ = "0.1.0"

# Same floating-point code paths on every x86 CPU, so classical models train identically: numpy's
# AVX-512 kernels (X86_V4) and OpenBLAS's per-CPU kernels round differently from the AVX2 ones
# the committed models were trained with. Set here because every stage and the API import `fjd`
# before numpy loads; variables already set in the environment win.
os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "X86_V4")
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
