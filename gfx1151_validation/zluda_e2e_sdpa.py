#!/usr/bin/env python3
"""RUNG 8.5: full real-image Qwen3-VL-2B VLM forward through ZLUDA, now with the ATTENTION
matmuls (QK^T and softmax*V) ALSO routed through ZLUDA->rocBLAS — not just the Linear projections.

Rung 8 routed every nn.Linear GEMM through ZLUDA. This rung adds the *other* dominant FLOP class:
scaled-dot-product attention. We monkeypatch BOTH:
  - torch.nn.functional.linear  -> ZLUDA GEMM (reused from zluda_e2e, all vision+text+lm_head projections)
  - torch.nn.functional.scaled_dot_product_attention -> per (batch,head): S = scale*(Q@K^T) via ZLUDA GEMM,
    additive mask + softmax on host, O = softmax(S) @ V via ZLUDA GEMM.
HF still owns all host glue (preprocessing, patch embed, deepstack, rope/M-rope, scatter, norms, the softmax
nonlinearity). So now BOTH the projection GEMMs and the attention score/context matmuls of the literal target
workload (Cosmos-Reason2 == Qwen3-VL) execute on gfx1151 through ZLUDA, end-to-end, on a real image.

Oracle = the SAME model, SAME inputs, UNPATCHED on torch-CPU fp32 (eager attention to match math exactly).
Grade: top-k argmax order + max_logit_diff across the full 151k vocab.

Run: LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 zluda_e2e_sdpa.py
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_e2e import ZGemm, install_zluda_linear, uninstall_zluda_linear, build_inputs, MODEL
import zluda_e2e

_ZA = None  # ZGemm instance for attention

def _softmax_rows(S):
    m = np.max(S, axis=-1, keepdims=True)
    m[~np.isfinite(m)] = 0.0
    e = np.exp(S - m)
    e[~np.isfinite(e)] = 0.0
    denom = np.sum(e, axis=-1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return e / denom

def install_zluda_sdpa():
    """Patch F.scaled_dot_product_attention to run QK^T and P@V through ZLUDA->rocBLAS."""
    import torch
    import torch.nn.functional as F
    global _ZA
    _ZA = ZGemm()
    _orig = F.scaled_dot_product_attention
    counter = {"qk": 0, "pv": 0}

    def zluda_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, enable_gqa=False, **kw):
        q = query.detach().to(torch.float32)
        k = key.detach().to(torch.float32)
        v = value.detach().to(torch.float32)
        B, H, Sq, D = q.shape
        _, Hk, Sk, Dv = v.shape
        if Hk != H:                                   # GQA: expand kv heads to match q
            rep = H // Hk
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        sc = scale if scale is not None else 1.0 / math.sqrt(D)
        # assemble a [B,H,Sq,Sk] additive bias once (host) — cheap relative to the GEMMs
        bias = torch.zeros((B, H, Sq, Sk), dtype=torch.float32)
        if is_causal:
            cm = torch.triu(torch.ones(Sq, Sk, dtype=torch.bool), diagonal=1)
            bias = bias.masked_fill(cm, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                bias = bias.masked_fill(~attn_mask, float("-inf"))
            else:
                bias = bias + attn_mask.to(torch.float32)
        bias_np = bias.numpy()
        out = np.empty((B, H, Sq, D), dtype=np.float32)
        for b in range(B):
            for h in range(H):
                Qm = np.ascontiguousarray(q[b, h].numpy())          # [Sq, D]
                Km = np.ascontiguousarray(k[b, h].numpy())          # [Sk, D]
                Vm = np.ascontiguousarray(v[b, h].numpy())          # [Sk, D]
                S = _ZA.gemm_np(Qm, np.ascontiguousarray(Km.T), Sq, D, Sk)  # [Sq, Sk] = Q@K^T
                counter["qk"] += 1
                S = S * sc + bias_np[b, h]
                Pm = _softmax_rows(S).astype(np.float32)
                O = _ZA.gemm_np(Pm, Vm, Sq, Sk, D)                  # [Sq, D] = P@V
                counter["pv"] += 1
                out[b, h] = O
        res = torch.from_numpy(out).to(query.dtype)
        return res

    F.scaled_dot_product_attention = zluda_sdpa
    return _orig, counter

def uninstall_zluda_sdpa(orig):
    import torch.nn.functional as F
    F.scaled_dot_product_attention = orig

def main():
    import torch
    from transformers import AutoModelForImageTextToText
    torch.manual_seed(0)
    print("loading", MODEL, "CPU fp32 (attn_implementation=sdpa) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    inputs = build_inputs()
    seq = inputs["input_ids"].shape[1]
    npix = inputs.get("pixel_values", None)
    print("seq_len:", seq, "| pixel_values:", None if npix is None else tuple(npix.shape),
          "| grid_thw:", inputs.get("image_grid_thw", None), flush=True)

    with torch.no_grad():
        ref = model(**inputs).logits[0, -1].float().numpy().copy()
    print("CPU oracle forward done.", flush=True)

    orig_lin = install_zluda_linear()
    orig_sdpa, ctr = install_zluda_sdpa()
    try:
        with torch.no_grad():
            zl = model(**inputs).logits[0, -1].float().numpy().copy()
    finally:
        uninstall_zluda_sdpa(orig_sdpa)
        uninstall_zluda_linear(orig_lin)
    print(f"ZLUDA forward done. F.linear GEMMs: {zluda_e2e._Z.calls} | "
          f"attn QK^T GEMMs: {ctr['qk']} | attn P@V GEMMs: {ctr['pv']}", flush=True)

    z_top = np.argsort(-zl)[:10].tolist()
    r_top = np.argsort(-ref)[:10].tolist()
    print("ZLUDA top-10 ids:", z_top, flush=True)
    print("HF    top-10 ids:", r_top, flush=True)
    argmax_ok = z_top[0] == r_top[0]
    top5_ok = z_top[:5] == r_top[:5]
    set10_ok = set(z_top) == set(r_top)
    maxld = float(np.max(np.abs(zl[r_top] - ref[r_top])))
    maxld_all = float(np.max(np.abs(zl - ref)))
    print(f"argmax_match={argmax_ok} top5_match={top5_ok} top10_setmatch={set10_ok} "
          f"max_logit_diff_top10={maxld:.4f} max_logit_diff_all={maxld_all:.4f}", flush=True)
    ok = argmax_ok and top5_ok
    print("VERDICT:", "PASS" if ok else ("PARTIAL(argmax-only)" if argmax_ok else "FAIL"), flush=True)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
