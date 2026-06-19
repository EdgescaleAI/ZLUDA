"""T0 primitive ops, derived from tests/t1_coverage/op-inventory.md.

Each Op: a framework-agnostic reference `fn(inputs)->nested-list`, a seeded
`make_inputs()`, and the inventory `family` it covers. Backends decide the dtype;
the fn is pure math. To add coverage, implement an op and append it to REGISTRY.
"""

import math
import random

from harness.tensor import matmul, transpose


class Op:
    def __init__(self, name, family, fn, mk, seed=1234,
                 ref_dtype="fp32", cand_dtype="bf16", case_id="c0"):
        self.name = name
        self.family = family
        self.fn = fn
        self._mk = mk
        self.seed = seed
        self.ref_dtype = ref_dtype
        self.cand_dtype = cand_dtype
        self.case_id = case_id

    def make_inputs(self):
        return self._mk(self.seed)


def _randmat(rng, r, c, lo=-1.0, hi=1.0):
    return [[rng.uniform(lo, hi) for _ in range(c)] for _ in range(r)]


# ---- GEMM (-> rocBLAS/hipBLASLt redirect target) -------------------------------
def _gemm_mk(seed):
    rng = random.Random(seed)
    return {"A": _randmat(rng, 8, 16), "B": _randmat(rng, 16, 8)}


def _gemm_fn(ins):
    return matmul(ins["A"], ins["B"])


# ---- RMSNorm (+ the per-head q/k-norm the inventory flagged) --------------------
def _rmsnorm_mk(seed):
    rng = random.Random(seed)
    return {"x": _randmat(rng, 4, 16), "w": [rng.uniform(0.5, 1.5) for _ in range(16)]}


def _rmsnorm_fn(ins):
    x, w, eps = ins["x"], ins["w"], 1e-6
    out = []
    for row in x:
        ms = sum(v * v for v in row) / len(row)
        inv = 1.0 / math.sqrt(ms + eps)
        out.append([v * inv * wi for v, wi in zip(row, w)])
    return out


# ---- Softmax (rowwise; the attention + sampling primitive) ---------------------
def _softmax_mk(seed):
    return {"x": _randmat(random.Random(seed), 4, 16)}


def _softmax_fn(ins):
    out = []
    for row in ins["x"]:
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        s = sum(exps)
        out.append([e / s for e in exps])
    return out


# ---- SwiGLU (SiLU(gate) * up) --------------------------------------------------
def _swiglu_mk(seed):
    rng = random.Random(seed)
    return {"gate": _randmat(rng, 4, 16), "up": _randmat(rng, 4, 16)}


def _swiglu_fn(ins):
    out = []
    for g, u in zip(ins["gate"], ins["up"]):
        out.append([(gv / (1.0 + math.exp(-gv))) * uv for gv, uv in zip(g, u)])
    return out


# ---- GELU (tanh approximation; vision/merger MLP activation) --------------------
def _gelu_mk(seed):
    return {"x": _randmat(random.Random(seed), 4, 16)}


def _gelu_fn(ins):
    k = 0.7978845608028654  # sqrt(2/pi)
    return [[0.5 * v * (1.0 + math.tanh(k * (v + 0.044715 * v ** 3))) for v in row]
            for row in ins["x"]]


# ---- RoPE (rotary position embedding) ------------------------------------------
def _rope_mk(seed):
    return {"x": _randmat(random.Random(seed), 4, 8)}  # seq=4, head_dim=8


def _rope_fn(ins):
    x = ins["x"]
    dim = len(x[0])
    half = dim // 2
    theta = 10000.0
    out = []
    for pos, row in enumerate(x):
        r = [0.0] * dim
        for i in range(half):
            ang = pos * (theta ** (-2.0 * i / dim))
            c, s = math.cos(ang), math.sin(ang)
            x1, x2 = row[i], row[i + half]
            r[i] = x1 * c - x2 * s
            r[i + half] = x1 * s + x2 * c
        out.append(r)
    return out


# ---- Attention (causal, eager bmm + softmax fallback path) ---------------------
def _attn_mk(seed):
    rng = random.Random(seed)
    return {"Q": _randmat(rng, 4, 8), "K": _randmat(rng, 4, 8), "V": _randmat(rng, 4, 8)}


def _attn_fn(ins):
    Q, K, V = ins["Q"], ins["K"], ins["V"]
    d = len(Q[0])
    scale = 1.0 / math.sqrt(d)
    scores = matmul(Q, transpose(K))  # seq x seq
    probs = []
    for i, row in enumerate(scores):
        masked = [row[j] * scale if j <= i else -1e30 for j in range(len(row))]
        m = max(masked)
        exps = [math.exp(v - m) for v in masked]
        s = sum(exps)
        probs.append([e / s for e in exps])
    return matmul(probs, V)  # seq x d


# ---- attention helpers (shared by the GQA / vision / paged variants) -----------
def _attend(Q, K, V, allow):
    """Generic attention. allow[i][j] gates which keys query i may see."""
    d = len(Q[0])
    scale = 1.0 / math.sqrt(d)
    scores = matmul(Q, transpose(K))
    probs = []
    for i, row in enumerate(scores):
        masked = [row[j] * scale if allow[i][j] else -1e30 for j in range(len(row))]
        m = max(masked)
        e = [math.exp(v - m) for v in masked]
        s = sum(e)
        probs.append([x / s for x in e])
    return matmul(probs, V)


def _split_heads(mat, n_heads, hd):
    return [[row[h * hd:(h + 1) * hd] for row in mat] for h in range(n_heads)]


def _concat_heads(heads):
    out = []
    for i in range(len(heads[0])):
        row = []
        for h in heads:
            row.extend(h[i])
        out.append(row)
    return out


# ---- attn-causal-gqa: grouped-query attention (Q heads > KV heads) --------------
def _gqa_mk(seed):
    rng = random.Random(seed)
    return {"Q": _randmat(rng, 3, 16), "K": _randmat(rng, 3, 8), "V": _randmat(rng, 3, 8)}


def _gqa_fn(ins):
    nq, nkv, hd = 4, 2, 4
    Qh = _split_heads(ins["Q"], nq, hd)
    Kh = _split_heads(ins["K"], nkv, hd)
    Vh = _split_heads(ins["V"], nkv, hd)
    seq = len(ins["Q"])
    causal = [[j <= i for j in range(seq)] for i in range(seq)]
    outs = [_attend(Qh[h], Kh[h // (nq // nkv)], Vh[h // (nq // nkv)], causal) for h in range(nq)]
    return _concat_heads(outs)


# ---- attn-vision-varlen: bidirectional, block-diagonal over two images ---------
def _visattn_mk(seed):
    rng = random.Random(seed)
    return {"Q": _randmat(rng, 4, 8), "K": _randmat(rng, 4, 8), "V": _randmat(rng, 4, 8)}


def _visattn_fn(ins):
    nh, hd = 2, 4
    seg = [0, 0, 1, 1]  # two images of length 2 (the cu_seqlens var-len pattern)
    allow = [[seg[i] == seg[j] for j in range(4)] for i in range(4)]
    Qh = _split_heads(ins["Q"], nh, hd)
    Kh = _split_heads(ins["K"], nh, hd)
    Vh = _split_heads(ins["V"], nh, hd)
    return _concat_heads([_attend(Qh[h], Kh[h], Vh[h], allow) for h in range(nh)])


# ---- attn-paged: KV stored in fixed-size pages, gathered then attended ---------
def _paged_mk(seed):
    rng = random.Random(seed)
    return {"Q": _randmat(rng, 4, 4), "K": _randmat(rng, 4, 4), "V": _randmat(rng, 4, 4)}


def _paged_fn(ins):
    page = 2
    K, V = ins["K"], ins["V"]
    Kg = [r for p in (K[i:i + page] for i in range(0, len(K), page)) for r in p]
    Vg = [r for p in (V[i:i + page] for i in range(0, len(V), page)) for r in p]
    seq = len(ins["Q"])
    causal = [[j <= i for j in range(seq)] for i in range(seq)]
    return _attend(ins["Q"], Kg, Vg, causal)


# ---- layernorm (vision / merger) -----------------------------------------------
def _ln_mk(seed):
    rng = random.Random(seed)
    return {"x": _randmat(rng, 4, 16),
            "g": [rng.uniform(0.5, 1.5) for _ in range(16)],
            "b": [rng.uniform(-0.5, 0.5) for _ in range(16)]}


def _ln_fn(ins):
    x, g, b, eps = ins["x"], ins["g"], ins["b"], 1e-6
    out = []
    for row in x:
        n = len(row)
        mu = sum(row) / n
        var = sum((v - mu) ** 2 for v in row) / n
        inv = 1.0 / math.sqrt(var + eps)
        out.append([(v - mu) * inv * gi + bi for v, gi, bi in zip(row, g, b)])
    return out


# ---- rope-2d-vision: first half encodes height, second half width --------------
def _rope2d_mk(seed):
    return {"x": _randmat(random.Random(seed), 4, 8)}  # 2x2 grid, dim 8


def _rope2d_fn(ins):
    x = ins["x"]
    dim = len(x[0])
    half = dim // 2
    theta = 10000.0
    out = []
    for t, row in enumerate(x):
        coords = (t // 2, t % 2)  # (h, w) on a 2x2 grid
        r = list(row)
        for axis, base in enumerate((0, half)):
            pos = coords[axis]
            hl = half // 2
            for i in range(hl):
                ang = pos * (theta ** (-2.0 * i / half))
                c, s = math.cos(ang), math.sin(ang)
                x1, x2 = row[base + i], row[base + hl + i]
                r[base + i] = x1 * c - x2 * s
                r[base + hl + i] = x1 * s + x2 * c
        out.append(r)
    return out


# ---- repeat-kv: broadcast KV heads up to Q-head count --------------------------
def _repkv_mk(seed):
    return {"kv": _randmat(random.Random(seed), 3, 8)}  # 2 kv heads x hd 4


def _repkv_fn(ins):
    nkv, hd, rep = 2, 4, 2
    heads = _split_heads(ins["kv"], nkv, hd)
    expanded = [h for h in heads for _ in range(rep)]
    return _concat_heads(expanded)


# ---- residual-add --------------------------------------------------------------
def _resid_mk(seed):
    rng = random.Random(seed)
    return {"x": _randmat(rng, 4, 16), "y": _randmat(rng, 4, 16)}


def _resid_fn(ins):
    return [[a + b for a, b in zip(rx, ry)] for rx, ry in zip(ins["x"], ins["y"])]


# ---- embedding-gather ----------------------------------------------------------
def _emb_mk(seed):
    rng = random.Random(seed)
    return {"table": _randmat(rng, 8, 4),
            "ids": [[float(rng.randrange(8)) for _ in range(5)]]}


def _emb_fn(ins):
    table = ins["table"]
    return [table[int(round(i))] for i in ins["ids"][0]]


# ---- posemb-interp: bilinear resize of a position-embedding grid ---------------
def _posemb_mk(seed):
    return {"grid": _randmat(random.Random(seed), 4, 4)}  # 2x2 grid, dim 4


def _posemb_fn(ins):
    g = ins["grid"]
    gh, gw, th, tw, d = 2, 2, 3, 3, len(ins["grid"][0])
    cell = lambda i, j: g[i * gw + j]
    out = []
    for ti in range(th):
        for tj in range(tw):
            si = ti * (gh - 1) / (th - 1)
            sj = tj * (gw - 1) / (tw - 1)
            i0, j0 = int(math.floor(si)), int(math.floor(sj))
            i1, j1 = min(i0 + 1, gh - 1), min(j0 + 1, gw - 1)
            di, dj = si - i0, sj - j0
            c00, c01, c10, c11 = cell(i0, j0), cell(i0, j1), cell(i1, j0), cell(i1, j1)
            row = []
            for k in range(d):
                top = c00[k] * (1 - dj) + c01[k] * dj
                bot = c10[k] * (1 - dj) + c11[k] * dj
                row.append(top * (1 - di) + bot * di)
            out.append(row)
    return out


# ---- spatial-merge: 2x2 patch group concatenated into one token ----------------
def _merge_mk(seed):
    return {"x": _randmat(random.Random(seed), 4, 4)}  # 2x2 grid, dim 4


def _merge_fn(ins):
    merged = []
    for row in ins["x"]:
        merged.extend(row)
    return [merged]


# ---- token-scatter-deepstack: place vision tokens + add deepstack taps ---------
def _scatter_mk(seed):
    rng = random.Random(seed)
    return {"llm": _randmat(rng, 4, 4), "vis": _randmat(rng, 2, 4), "tap": _randmat(rng, 2, 4)}


def _scatter_fn(ins):
    llm = [row[:] for row in ins["llm"]]
    for k, p in enumerate((1, 2)):  # scatter at the image-token positions
        llm[p] = [a + b for a, b in zip(ins["vis"][k], ins["tap"][k])]
    return llm


# ---- sampling-topk-topp: temperature + top-k filter + renormalized softmax -----
def _samp_mk(seed):
    rng = random.Random(seed)
    logits = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]  # well-separated => stable top-k
    rng.shuffle(logits)
    return {"logits": [logits]}


def _samp_fn(ins):
    logits = ins["logits"][0]
    temp, k = 0.8, 4
    scaled = [v / temp for v in logits]
    order = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)
    keep = set(order[:k])
    masked = [scaled[i] if i in keep else -1e30 for i in range(len(scaled))]
    m = max(masked)
    e = [math.exp(v - m) for v in masked]
    s = sum(e)
    return [[x / s for x in e]]


# ---- kv-cache: append new K/V to the cache and read the full cache back --------
def _kv_mk(seed):
    rng = random.Random(seed)
    return {"cache": _randmat(rng, 2, 4), "new": _randmat(rng, 1, 4)}


def _kv_fn(ins):
    return [row[:] for row in ins["cache"]] + [row[:] for row in ins["new"]]


REGISTRY = [
    # the original 7
    Op("gemm_8x16x8", "gemm", _gemm_fn, _gemm_mk),
    Op("rmsnorm_4x16", "rmsnorm", _rmsnorm_fn, _rmsnorm_mk),
    Op("softmax_4x16", "softmax", _softmax_fn, _softmax_mk),
    Op("swiglu_4x16", "swiglu", _swiglu_fn, _swiglu_mk),
    Op("gelu_4x16", "gelu", _gelu_fn, _gelu_mk),
    Op("rope_4x8", "rope-mrope", _rope_fn, _rope_mk),
    Op("attn_causal_4x8", "attn-eager-fallback", _attn_fn, _attn_mk),
    # wired in from the Cosmos op inventory (the remaining 13 families)
    Op("gqa_attn_3x16", "attn-causal-gqa", _gqa_fn, _gqa_mk),
    Op("vis_attn_varlen_4x8", "attn-vision-varlen", _visattn_fn, _visattn_mk),
    Op("paged_attn_4x4", "attn-paged", _paged_fn, _paged_mk),
    Op("layernorm_4x16", "layernorm", _ln_fn, _ln_mk),
    Op("rope2d_4x8", "rope-2d-vision", _rope2d_fn, _rope2d_mk),
    Op("repeat_kv_3x8", "repeat-kv", _repkv_fn, _repkv_mk),
    Op("residual_add_4x16", "residual-add", _resid_fn, _resid_mk),
    Op("embedding_gather_5", "embedding-gather", _emb_fn, _emb_mk),
    Op("posemb_interp_2to3", "posemb-interp", _posemb_fn, _posemb_mk),
    Op("spatial_merge_2x2", "spatial-merge", _merge_fn, _merge_mk),
    Op("token_scatter_4x4", "token-scatter-deepstack", _scatter_fn, _scatter_mk),
    Op("sampling_topk_8", "sampling-topk-topp", _samp_fn, _samp_mk),
    Op("kv_cache_2p1", "kv-cache", _kv_fn, _kv_mk),
]
