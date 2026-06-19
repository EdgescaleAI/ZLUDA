"""Coverage tracking against the Cosmos Reason 2 op inventory
(tests/t1_coverage/op-inventory.md). The fan-out target: drive this toward 100%
via loop-until-dry. Families mirror the inventory's dedup'd T0 checklist."""

FAMILIES = [
    "gemm",
    "attn-causal-gqa",
    "attn-vision-varlen",
    "attn-eager-fallback",
    "attn-paged",
    "rmsnorm",
    "layernorm",
    "swiglu",
    "gelu",
    "rope-mrope",
    "rope-2d-vision",
    "repeat-kv",
    "residual-add",
    "embedding-gather",
    "posemb-interp",
    "spatial-merge",
    "token-scatter-deepstack",
    "softmax",
    "sampling-topk-topp",
    "kv-cache",
]


def coverage_report(registry):
    covered = {}
    for op in registry:
        covered.setdefault(op.family, []).append(op.name)
    rows = [(fam, covered.get(fam, [])) for fam in FAMILIES]
    n_cov = sum(1 for fam in FAMILIES if covered.get(fam))
    # families used by an op but not in the canonical list (typo guard)
    unknown = sorted(set(covered) - set(FAMILIES))
    return rows, n_cov, len(FAMILIES), unknown
