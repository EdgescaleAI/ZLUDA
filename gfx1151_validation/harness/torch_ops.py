"""Torch op implementations for TorchBackend, keyed by op.name.

Each impl takes the raw inputs dict (nested Python lists from op.make_inputs()),
a torch dtype, and a device; returns a torch.Tensor. The runner converts the
tensor back to nested lists for diffing/fixtures.

CONTRACT: each impl MUST match the corresponding pure-Python reference fn in
tests/t0_ops/ops.py exactly — same formula, same constants. Use the tanh-approx
GELU, eps=1e-6, theta=10000, etc., NOT a torch library default that differs (e.g.
F.gelu defaults to the erf form; rope/scaling conventions vary). The reference fn
is the spec.

STATUS: 6 worked examples below. The other 14 ops are the overnight task — see the
TODO list at the bottom. Implement one and `verify` flips it TODO -> PASS/FAIL.
NOTE: untested on the authoring machine (no torch there); validate in the venv.
"""

import math

import torch

TORCH_OPS = {}


def register(name):
    def deco(fn):
        TORCH_OPS[name] = fn
        return fn
    return deco


def _t(x, dtype, device):
    return torch.tensor(x, dtype=dtype, device=device)


# ---- worked examples (copy this pattern for the TODO ops) ----------------------

@register("gemm_8x16x8")
def _gemm(ins, dtype, device):
    return _t(ins["A"], dtype, device) @ _t(ins["B"], dtype, device)


@register("residual_add_4x16")
def _residual_add(ins, dtype, device):
    return _t(ins["x"], dtype, device) + _t(ins["y"], dtype, device)


@register("rmsnorm_4x16")
def _rmsnorm(ins, dtype, device):
    x = _t(ins["x"], dtype, device)
    w = _t(ins["w"], dtype, device)              # shape (16,), broadcasts over rows
    inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return x * inv * w


@register("softmax_4x16")
def _softmax(ins, dtype, device):
    return torch.softmax(_t(ins["x"], dtype, device), dim=-1)


@register("gelu_4x16")
def _gelu(ins, dtype, device):
    x = _t(ins["x"], dtype, device)
    k = 0.7978845608028654                       # sqrt(2/pi) — tanh approx, matches ref
    return 0.5 * x * (1.0 + torch.tanh(k * (x + 0.044715 * x.pow(3))))


@register("swiglu_4x16")
def _swiglu(ins, dtype, device):
    g = _t(ins["gate"], dtype, device)
    u = _t(ins["up"], dtype, device)
    return (g * torch.sigmoid(g)) * u            # SiLU(gate) * up


# ---- shared attention helper (mirrors _attend in t0_ops/ops.py) ----------------
def _attend(Q, K, V, allow):
    """Generic attention. `allow` is a bool tensor (sq, sk) gating which keys each
    query may see; masked entries get -1e30 before softmax — exactly as the
    reference _attend does (scale applied only to the kept scores, conceptually,
    but multiplying all scores by the scalar then masking is identical)."""
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = Q @ K.transpose(-1, -2)
    # Mask sentinel: the reference uses -1e30, which already overflows to -inf in
    # any fp16 context (exp(masked-m) underflows to 0 either way). CPU saturates
    # -1e30 -> -inf silently; MPS RAISES on the overflowing fill, so we use -inf
    # directly — numerically identical (fp32/bf16 outputs bit-unchanged), MPS-safe.
    neg = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
    masked = torch.where(allow, scores * scale, neg)
    probs = torch.softmax(masked, dim=-1)
    return probs @ V


def _heads(mat, n_heads, hd):
    """Column-split into heads, mirroring _split_heads: head h = cols [h*hd:(h+1)*hd]."""
    return [mat[:, h * hd:(h + 1) * hd] for h in range(n_heads)]


def _causal(seq, device):
    return torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device))


# ---- overnight ops: each mirrors the reference fn in t0_ops/ops.py EXACTLY ------
# NOTE on transcendental tables (rope, rope2d): the rotation angles/cos/sin are a
# data-independent table; they're computed in fp32 and cast to the working dtype
# before being applied (standard practice — real backends precompute RoPE tables
# in fp32). So Phase-C fp16/bf16 divergence here reflects the *data path*, not
# angle-table rounding. Documented choice; the rotation arithmetic itself runs in
# the working dtype.

@register("rope_4x8")
def _rope(ins, dtype, device):
    x = _t(ins["x"], dtype, device)
    seq, dim = x.shape
    half = dim // 2
    theta = 10000.0
    pos = torch.arange(seq, dtype=torch.float32, device=device).unsqueeze(1)   # (seq,1)
    i = torch.arange(half, dtype=torch.float32, device=device)                 # (half,)
    ang = pos * (theta ** (-2.0 * i / dim))                                    # (seq,half) fp32
    c, s = torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)
    x1, x2 = x[:, :half], x[:, half:]
    out = torch.empty_like(x)
    out[:, :half] = x1 * c - x2 * s
    out[:, half:] = x1 * s + x2 * c
    return out


@register("attn_causal_4x8")
def _attn_causal(ins, dtype, device):
    Q, K, V = (_t(ins[k], dtype, device) for k in ("Q", "K", "V"))
    return _attend(Q, K, V, _causal(Q.shape[0], device))


@register("gqa_attn_3x16")
def _gqa(ins, dtype, device):
    Q, K, V = (_t(ins[k], dtype, device) for k in ("Q", "K", "V"))
    nq, nkv, hd = 4, 2, 4                       # matches _gqa_fn
    Qh, Kh, Vh = _heads(Q, nq, hd), _heads(K, nkv, hd), _heads(V, nkv, hd)
    allow = _causal(Q.shape[0], device)
    outs = [_attend(Qh[h], Kh[h // (nq // nkv)], Vh[h // (nq // nkv)], allow)
            for h in range(nq)]
    return torch.cat(outs, dim=1)


@register("vis_attn_varlen_4x8")
def _vis_attn(ins, dtype, device):
    Q, K, V = (_t(ins[k], dtype, device) for k in ("Q", "K", "V"))
    nh, hd = 2, 4
    seg = [0, 0, 1, 1]                          # two images of length 2
    allow = torch.tensor([[seg[i] == seg[j] for j in range(4)] for i in range(4)],
                         dtype=torch.bool, device=device)
    Qh, Kh, Vh = _heads(Q, nh, hd), _heads(K, nh, hd), _heads(V, nh, hd)
    return torch.cat([_attend(Qh[h], Kh[h], Vh[h], allow) for h in range(nh)], dim=1)


@register("paged_attn_4x4")
def _paged(ins, dtype, device):
    # The reference gathers KV in fixed-size pages, but page collection is an
    # in-order identity for this fixture (contiguous pages reassembled in order),
    # so Kg == K, Vg == V. Single-head causal attention over the full sequence.
    Q, K, V = (_t(ins[k], dtype, device) for k in ("Q", "K", "V"))
    return _attend(Q, K, V, _causal(Q.shape[0], device))


@register("layernorm_4x16")
def _layernorm(ins, dtype, device):
    x = _t(ins["x"], dtype, device)
    g = _t(ins["g"], dtype, device)
    b = _t(ins["b"], dtype, device)
    mu = x.mean(dim=-1, keepdim=True)
    var = (x - mu).pow(2).mean(dim=-1, keepdim=True)   # population var, matches ref
    inv = torch.rsqrt(var + 1e-6)
    return (x - mu) * inv * g + b


@register("rope2d_4x8")
def _rope2d(ins, dtype, device):
    x = _t(ins["x"], dtype, device)
    seq, dim = x.shape
    half = dim // 2
    hl = half // 2
    theta = 10000.0
    out = x.clone()
    h = torch.tensor([t // 2 for t in range(seq)], dtype=torch.float32, device=device)
    w = torch.tensor([t % 2 for t in range(seq)], dtype=torch.float32, device=device)
    coords = (h, w)
    for axis, base in enumerate((0, half)):
        pos = coords[axis]
        for i in range(hl):
            ang = pos * (theta ** (-2.0 * i / half))   # (seq,) fp32
            c, s = torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)
            x1, x2 = x[:, base + i], x[:, base + hl + i]
            out[:, base + i] = x1 * c - x2 * s
            out[:, base + hl + i] = x1 * s + x2 * c
    return out


@register("repeat_kv_3x8")
def _repeat_kv(ins, dtype, device):
    kv = _t(ins["kv"], dtype, device)
    nkv, hd, rep = 2, 4, 2
    heads = _heads(kv, nkv, hd)
    expanded = [h for h in heads for _ in range(rep)]
    return torch.cat(expanded, dim=1)


@register("embedding_gather_5")
def _embedding_gather(ins, dtype, device):
    table = _t(ins["table"], dtype, device)
    ids = ins["ids"][0]                                 # raw floats; round in python (matches ref)
    idx = torch.tensor([int(round(i)) for i in ids], dtype=torch.long, device=device)
    return table.index_select(0, idx)


@register("posemb_interp_2to3")
def _posemb_interp(ins, dtype, device):
    g = _t(ins["grid"], dtype, device)
    gh, gw, th, tw = 2, 2, 3, 3
    cell = lambda i, j: g[i * gw + j]                   # row vector (dtype tensor)
    rows = []
    for ti in range(th):
        for tj in range(tw):
            si = ti * (gh - 1) / (th - 1)
            sj = tj * (gw - 1) / (tw - 1)
            i0, j0 = int(math.floor(si)), int(math.floor(sj))
            i1, j1 = min(i0 + 1, gh - 1), min(j0 + 1, gw - 1)
            di, dj = si - i0, sj - j0
            top = cell(i0, j0) * (1 - dj) + cell(i0, j1) * dj
            bot = cell(i1, j0) * (1 - dj) + cell(i1, j1) * dj
            rows.append(top * (1 - di) + bot * di)
    return torch.stack(rows)


@register("spatial_merge_2x2")
def _spatial_merge(ins, dtype, device):
    x = _t(ins["x"], dtype, device)                     # (4,4)
    return x.reshape(1, -1)                             # row-major concat -> (1,16)


@register("token_scatter_4x4")
def _token_scatter(ins, dtype, device):
    llm = _t(ins["llm"], dtype, device).clone()
    vis = _t(ins["vis"], dtype, device)
    tap = _t(ins["tap"], dtype, device)
    for k, p in enumerate((1, 2)):
        llm[p] = vis[k] + tap[k]
    return llm


@register("sampling_topk_8")
def _sampling_topk(ins, dtype, device):
    logits = _t(ins["logits"], dtype, device)           # (1,8)
    temp, k = 0.8, 4
    scaled = logits / temp
    topv, topi = torch.topk(scaled, k, dim=-1)          # well-separated => stable top-k
    # -inf, not -1e30: same MPS-fp16 overflow quirk as _attend (see note there).
    masked = torch.full_like(scaled, float("-inf"))
    masked.scatter_(-1, topi, topv)
    return torch.softmax(masked, dim=-1)


@register("kv_cache_2p1")
def _kv_cache(ins, dtype, device):
    cache = _t(ins["cache"], dtype, device)
    new = _t(ins["new"], dtype, device)
    return torch.cat([cache, new], dim=0)
