"""Per-dtype tolerance policy. The key honesty rule (see ../../TEST-STRATEGY.md):
same-precision comparisons are tight numeric diffs; genuinely cross-precision /
quantized paths (fp4/fp8/int4 candidate) cannot bit-match a higher-precision
reference and must be validated by eval, not element-wise diff — those return the
'eval' mode and are reported EVAL-ONLY rather than PASS/FAIL."""

# (ref_dtype, cand_dtype) -> (rtol, atol, mode)
TOL = {
    ("fp64", "fp64"): (0.0, 0.0, "exact"),
    ("fp32", "fp32"): (1e-5, 1e-6, "close"),
    ("fp32", "bf16"): (3e-2, 3e-2, "close"),
    ("bf16", "bf16"): (1e-5, 1e-6, "close"),
    ("int", "int"): (0.0, 0.0, "exact"),
}

_QUANT_CANDIDATES = {"fp4", "fp8", "int4", "nvfp4", "mxfp4"}


def tolerance_for(ref_dtype, cand_dtype):
    key = (ref_dtype, cand_dtype)
    if key in TOL:
        return TOL[key]
    if cand_dtype in _QUANT_CANDIDATES:
        return (None, None, "eval")
    # default: treat as a same-class numeric comparison, generously
    return (3e-2, 3e-2, "close")
