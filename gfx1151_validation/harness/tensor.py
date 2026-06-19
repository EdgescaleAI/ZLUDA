"""Minimal pure-Python tensor ops (nested lists, row-major). Tiny shapes only —
T0 tests are intentionally small, so a naive implementation is fine and keeps the
reference math readable and dependency-free."""


def matmul(A, B):
    m, k, n = len(A), len(A[0]), len(B[0])
    if len(B) != k:
        raise ValueError(f"matmul shape mismatch: A is {m}x{k}, B is {len(B)}x{n}")
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        Ai, Oi = A[i], out[i]
        for t in range(k):
            a, Bt = Ai[t], B[t]
            for j in range(n):
                Oi[j] += a * Bt[j]
    return out


def transpose(A):
    return [list(col) for col in zip(*A)]
