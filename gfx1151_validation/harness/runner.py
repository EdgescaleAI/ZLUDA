"""The execution loop for one test case: get the reference (capture fresh or load a
fixture), run the candidate, diff within the dtype tolerance, return a verdict.
This is the unit the agent fleet fans out over — one op, one machine-checkable
verdict."""

from harness.backends import OpNotImplemented
from harness.diff import compare
from harness.fixtures import load_fixture, save_fixture
from harness.tolerances import tolerance_for


def run_case(op, ref_backend, cand_backend, capture=False, allow_capture=True):
    inputs = op.make_inputs()

    try:
        fix = None if capture else load_fixture(op.name, op.case_id)
        if fix is None:
            if not capture and not allow_capture:
                return {"op": op.name, "family": op.family, "verdict": "NO-FIXTURE"}
            out_ref = ref_backend.run(op, inputs)
            save_fixture(op.name, op.case_id, out_ref, {
                "op": op.name, "case": op.case_id, "family": op.family, "seed": op.seed,
                "ref_backend": ref_backend.name, "ref_precision": ref_backend.precision,
                "ref_dtype": op.ref_dtype,
            })
        else:
            out_ref = fix["output"]

        out_cand = cand_backend.run(op, inputs)
    except OpNotImplemented as e:
        return {"op": op.name, "family": op.family, "verdict": "TODO",
                "detail": f"no torch impl for op '{e}'"}

    rtol, atol, mode = tolerance_for(op.ref_dtype, op.cand_dtype)

    if mode == "eval":
        return {"op": op.name, "family": op.family, "mode": mode, "verdict": "EVAL-ONLY",
                "detail": "cross-precision/quantized — validate by eval, not bit-diff"}

    d = compare(out_ref, out_cand, rtol, atol)
    return {"op": op.name, "family": op.family, "mode": mode,
            "verdict": "PASS" if d["allclose"] else "FAIL",
            "max_abs": d["max_abs"], "max_rel": d["max_rel"],
            "n": d["n"], "n_fail": d["n_fail"], "shape_mismatch": d["shape_mismatch"]}
