"""Command-line entry. Subcommands:
  selftest  ref=purepy:fp32  cand=purepy:bf16  — proves the loop on this machine
  capture   --ref <spec>                       — write golden fixtures from a reference
  verify    --ref <spec> --cand <spec>          — diff a candidate against stored fixtures
  report                                        — coverage vs the op inventory
"""

import argparse
import sys

from harness.backends import get_backend
from harness.coverage import coverage_report
from harness.runner import run_case
from t0_ops.ops import REGISTRY


def _print_result(r):
    name, fam = r["op"], r["family"]
    if r["verdict"] == "PASS":
        print(f"  PASS  {name:26s} {fam:22s} max_rel={r['max_rel']:.2e}")
    elif r["verdict"] == "EVAL-ONLY":
        print(f"  EVAL  {name:26s} {fam:22s} (cross-precision; eval-validated)")
    elif r["verdict"] == "TODO":
        print(f"  TODO  {name:26s} {fam:22s} (no torch impl yet)")
    elif r["verdict"] == "NO-FIXTURE":
        print(f"  MISS  {name:26s} {fam:22s} (no fixture — run `capture` first)")
    else:
        extra = " SHAPE-MISMATCH" if r.get("shape_mismatch") else ""
        print(f"  FAIL  {name:26s} {fam:22s} max_rel={r['max_rel']:.2e} "
              f"n_fail={r['n_fail']}/{r['n']}{extra}")


def _run_all(ref, cand, capture):
    print(f"ref={ref.name}  cand={cand.name}")
    print("-" * 72)
    n_pass = n_fail = n_todo = n_other = 0
    for op in REGISTRY:
        r = run_case(op, ref, cand, capture=capture, allow_capture=capture)
        _print_result(r)
        v = r["verdict"]
        if v == "PASS":
            n_pass += 1
        elif v == "FAIL":
            n_fail += 1
        elif v == "TODO":
            n_todo += 1
        else:
            n_other += 1
    print("-" * 72)
    print(f"{n_pass} passed, {n_fail} failed, {n_todo} todo, {n_other} other  "
          f"of {len(REGISTRY)} ops")
    return 1 if n_fail else 0


def cmd_selftest(_):
    print("Differential self-test (stdlib-only; fp32 reference vs bf16 candidate)")
    return _run_all(get_backend("purepy:fp32"), get_backend("purepy:bf16"), capture=True)


def cmd_capture(a):
    ref = get_backend(a.ref)
    for op in REGISTRY:
        run_case(op, ref, ref, capture=True)
    print(f"Captured {len(REGISTRY)} fixtures with ref={ref.name}")
    return 0


def cmd_verify(a):
    return _run_all(get_backend(a.ref), get_backend(a.cand), capture=False)


def cmd_report(_):
    rows, n_cov, total, unknown = coverage_report(REGISTRY)
    print(f"Op-family coverage vs Cosmos Reason 2 inventory: {n_cov}/{total}")
    print("-" * 72)
    for fam, ops in rows:
        mark = "x" if ops else " "
        print(f"  [{mark}] {fam:26s} {', '.join(ops) if ops else '— no test yet'}")
    if unknown:
        print(f"\n  WARNING: ops use families not in the canonical list: {unknown}")
    return 0


def main():
    p = argparse.ArgumentParser(prog="cuda-bridge-tests")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)
    c = sub.add_parser("capture"); c.add_argument("--ref", default="purepy:fp32"); c.set_defaults(func=cmd_capture)
    v = sub.add_parser("verify"); v.add_argument("--ref", default="purepy:fp32"); v.add_argument("--cand", default="purepy:bf16"); v.set_defaults(func=cmd_verify)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = p.parse_args()
    try:
        return args.func(args)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
