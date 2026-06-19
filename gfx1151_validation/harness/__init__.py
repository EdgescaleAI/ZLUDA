"""CUDA-Bridge differential-test harness (stdlib-only core).

The harness is backend-agnostic: a *reference* backend produces golden outputs,
a *candidate* backend produces the output under test, and the runner diffs them
within a per-dtype tolerance. Today both can be the pure-Python backend (runs on
stock macOS python3, no deps); later the NVIDIA reference and the Strix Halo /
CUDA-Bridge candidate drop in as `torch` backends without touching the rest.
"""
