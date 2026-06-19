#!/usr/bin/env python3
"""RUNG 7 (REAL WEIGHTS): real Qwen3-VL-2B-Instruct TEXT forward through ZLUDA on gfx1151.

The whole ladder used Qwen3 text models and synthetic VLM harnesses as stand-ins for the
gated Cosmos-Reason2 (= Qwen3-VL). `Qwen/Qwen3-VL-2B-Instruct` is OPEN/ungated and is the
LITERAL target architecture family, so this runs REAL target-model weights through ZLUDA.

Scope: the TEXT (language) tower forward. Qwen3-VL's LM is a 28-layer Qwen3 decoder
(hidden 2048, 16q/8kv GQA, head_dim 128, per-head QK-norm, SwiGLU, tied embeddings) that
uses M-RoPE. KEY FACT: for TEXT-ONLY input the 3 M-RoPE position axes all share the same
token index, so M-RoPE reduces EXACTLY to standard 1D RoPE with the model's own inv_freq
(theta 5e6). So the proven zluda_qwen forward (rope_f using the model inv_freq) applies.

Oracle = HF's own torch-CPU fp32 forward of the SAME model on the SAME ids (uses HF's correct
M-RoPE), so a top-k match is conclusive that the text tower runs correctly through ZLUDA.
Eval-style grade (top-k id agreement), per TEST-STRATEGY for whole-model tiers.

Run: LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 zluda_qwenvl.py
"""
import ctypes as C, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_diff import Cuda, compile_ptx, CUDA, CUBLAS, _ck
# reuse the exact device-memory + forward machinery proven on Qwen3 0.6B->32B
import zluda_qwen as ZQ
from zluda_qwen import QwenZLUDA, forward, dev_np, alloc, get_np, P

MODEL = "Qwen/Qwen3-VL-2B-Instruct"

def load_weights_vl():
    import torch
    from transformers import AutoModelForImageTextToText, AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    tc = getattr(cfg, "text_config", None) or cfg
    print("loading", MODEL, "on CPU fp32 ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    sd = model.state_dict()
    # locate the language-model weight prefix (Qwen3-VL nests the LM under model.language_model)
    cand = ["model.language_model.", "language_model.model.", "model.", "language_model."]
    prefix = None
    for pre in cand:
        if (pre + "layers.0.self_attn.q_proj.weight") in sd and (pre + "embed_tokens.weight") in sd:
            prefix = pre; break
    if prefix is None:
        raise RuntimeError("could not find LM weight prefix; keys sample: " +
                           ", ".join([k for k in list(sd.keys())[:8]]))
    print("LM weight prefix:", prefix, flush=True)
    # text-tower inv_freq: pick the rotary module whose inv_freq length == text head_dim/2
    want = tc.head_dim // 2
    inv_freq = None
    for mod in model.modules():
        f = getattr(mod, "inv_freq", None)
        if f is not None and f.numel() == want:
            inv_freq = f.detach().numpy().astype(np.float32).copy(); break
    if inv_freq is None:  # fall back: synthesize from rope_theta
        theta = float(getattr(tc, "rope_scaling", {}).get("rope_theta", getattr(tc, "rope_theta", 1e6)))
        inv_freq = (1.0 / (theta ** (np.arange(0, want).astype(np.float32) * 2.0 / tc.head_dim))).astype(np.float32)
        print("WARN: synthesized inv_freq from theta", theta, flush=True)
    def npT(t): return t.detach().numpy().astype(np.float32).T.copy()
    def npv(t): return t.detach().numpy().astype(np.float32).copy()
    L = tc.num_hidden_layers
    W = {"embed": npv(sd[prefix + "embed_tokens.weight"]),
         "final_norm": npv(sd[prefix + "norm.weight"]), "layers": []}
    if "lm_head.weight" in sd and not getattr(tc, "tie_word_embeddings", True):
        W["lm_head"] = npv(sd["lm_head.weight"])
    for i in range(L):
        p = f"{prefix}layers.{i}."
        W["layers"].append(dict(
            Wq=npT(sd[p+"self_attn.q_proj.weight"]), Wk=npT(sd[p+"self_attn.k_proj.weight"]),
            Wv=npT(sd[p+"self_attn.v_proj.weight"]), Wo=npT(sd[p+"self_attn.o_proj.weight"]),
            gQ=npv(sd[p+"self_attn.q_norm.weight"]), gK=npv(sd[p+"self_attn.k_norm.weight"]),
            gN1=npv(sd[p+"input_layernorm.weight"]), gN2=npv(sd[p+"post_attention_layernorm.weight"]),
            Wg=npT(sd[p+"mlp.gate_proj.weight"]), Wu=npT(sd[p+"mlp.up_proj.weight"]), Wd=npT(sd[p+"mlp.down_proj.weight"])))
    W["inv_freq"] = inv_freq
    Cd = {"H": tc.hidden_size, "L": L, "NQ": tc.num_attention_heads, "NKV": tc.num_key_value_heads,
          "HD": tc.head_dim, "I": tc.intermediate_size, "V": tc.vocab_size, "eps": float(tc.rms_norm_eps)}
    # oracle: HF's own text-only forward (correct M-RoPE) -> last-token logits
    ids = build_ids()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long))
        ref = out.logits[0, -1].float().numpy().copy()
    return W, Cd, ids, ref

def build_ids():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": "The capital of France is"}]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    except Exception:
        ids = tok("The capital of France is").input_ids
    return list(ids)

def main():
    W, cfg, ids, ref = load_weights_vl()
    print("config:", cfg, "| seq_len:", len(ids), flush=True)
    q = QwenZLUDA()
    logits = forward(q, W, cfg, ids)
    z_top = np.argsort(-logits)[:10].tolist()
    r_top = np.argsort(-ref)[:10].tolist()
    print("ZLUDA top-10 ids:", z_top, flush=True)
    print("HF    top-10 ids:", r_top, flush=True)
    argmax_ok = z_top[0] == r_top[0]
    top5_ok = z_top[:5] == r_top[:5]
    set10_ok = set(z_top) == set(r_top)
    # logit diff on the shared support
    maxld = float(np.max(np.abs(logits[r_top] - ref[r_top])))
    print(f"argmax_match={argmax_ok} top5_match={top5_ok} top10_setmatch={set10_ok} max_logit_diff={maxld:.3f}", flush=True)
    ok = argmax_ok and top5_ok
    print("VERDICT:", "PASS" if ok else ("PARTIAL" if argmax_ok else "FAIL"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
