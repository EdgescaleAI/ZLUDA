"""Tensor comparison. Flattens both trees and computes max abs / max rel error and
an element-wise allclose verdict: |a-b| <= atol + rtol*|a|, with `a` the reference."""


def _flatten(x, acc):
    if isinstance(x, list):
        for e in x:
            _flatten(e, acc)
    else:
        acc.append(float(x))
    return acc


def compare(ref, cand, rtol, atol):
    r = _flatten(ref, [])
    c = _flatten(cand, [])
    if len(r) != len(c):
        return {"n": max(len(r), len(c)), "n_fail": max(len(r), len(c)),
                "max_abs": float("inf"), "max_rel": float("inf"),
                "allclose": False, "shape_mismatch": True}
    max_abs = max_rel = 0.0
    n_fail = 0
    for a, b in zip(r, c):
        ae = abs(a - b)
        max_abs = max(max_abs, ae)
        denom = abs(a) if abs(a) > 0 else 1.0
        max_rel = max(max_rel, ae / denom)
        if ae > atol + rtol * abs(a):
            n_fail += 1
    return {"n": len(r), "n_fail": n_fail, "max_abs": max_abs,
            "max_rel": max_rel, "allclose": n_fail == 0, "shape_mismatch": False}
