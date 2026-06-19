#!/usr/bin/env python3
"""RUNG 6 (frontier): run a 32B-size-class OPEN model through ZLUDA on gfx1151, the
brief's named frontier (thesis: a 32B in BF16 ~64GB fits the ~96GB unified pool, beyond
typical discrete-GPU VRAM). Uses the OPEN Qwen3-32B as the size-class stand-in exactly as
the open Qwen3-8B stood in for the gated Cosmos-Reason2-8B (no HF token needed).

Memory-frugal so a 32B fits the cube's ~120GB host RAM:
  * load the model in fp16 (~64GB for 32B; fp32 would be 128GB > node RAM -> impossible);
  * run the transformers CPU forward as the INDEPENDENT oracle (its own kernels, not ours);
  * stream-convert weights to fp16 numpy, POPPING each tensor out of the state_dict as we go
    so host RAM stays ~one-model-size, not two;
  * reuse zluda_qwen.forward: dev_np() upcasts each fp16 weight to fp32 only at upload and the
    forward frees every layer's device buffers before the next -> device footprint is per-layer.

Grading flips here per the brief (TEST-STRATEGY): BF16/FP16 tiers are EVAL-validated, NOT bit-diff.
We grade behaviorally on top-k next-token agreement (argmax + top-5) between the ZLUDA forward
(fp32 GEMM on the same fp16 weights) and the transformers fp16 reference, and report the full
top-10 + logit diff honestly. Same source weights both paths; only compute precision differs.

Run: HSA_OVERRIDE_GFX_VERSION=11.5.1 LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib \
     python3 zluda_frontier.py Qwen/Qwen3-32B [fp16|bf16]
"""
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zluda_qwen as Q

def main():
    name  = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-32B"
    dtag  = sys.argv[2] if len(sys.argv) > 2 else "fp16"
    prompt_ids = [785, 6722, 315, 9625, 374]   # "The capital of France is"
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtag]
    print(f"loading {name} in {dtag} on CPU (memory-frugal) ...", flush=True)
    cfg = AutoConfig.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dt, low_cpu_mem_usage=True).eval()

    # --- INDEPENDENT oracle: transformers' own CPU forward in the loaded dtype ---
    with torch.no_grad():
        ref = model(torch.tensor([prompt_ids])).logits[0, -1].float()
    rtop = torch.topk(ref, 10)
    ref_ids = rtop.indices.tolist()
    ref_val = [round(v, 3) for v in rtop.values.tolist()]
    print("CPU(oracle) top ids:   ", ref_ids, flush=True)

    # exact inv_freq table from the model's rotary embedding (before we tear the model down)
    inv = None
    for m in model.modules():
        if getattr(m, "inv_freq", None) is not None:
            inv = m.inv_freq.detach().float().numpy().astype(np.float32).copy(); break

    # --- stream-convert weights to fp16 numpy, popping from the state_dict to bound RAM ---
    sd = model.state_dict()
    del model; gc.collect()    # params share storage with sd, so they persist via sd until popped
    L = cfg.num_hidden_layers
    h16 = lambda key: sd.pop(key).detach().float().numpy().astype(np.float16)   # value, fp16 store
    h16T = lambda key: np.ascontiguousarray(h16(key).T)                          # Linear [out,in]->[in,out]
    W = {"embed": h16("model.embed_tokens.weight"),
         "final_norm": h16("model.norm.weight"), "inv_freq": inv, "layers": []}
    if "lm_head.weight" in sd:
        W["lm_head"] = h16("lm_head.weight")    # untied LM head (Qwen3-8B/14B/32B)
    for i in range(L):
        p = f"model.layers.{i}."
        W["layers"].append(dict(
            Wq=h16T(p+"self_attn.q_proj.weight"), Wk=h16T(p+"self_attn.k_proj.weight"),
            Wv=h16T(p+"self_attn.v_proj.weight"), Wo=h16T(p+"self_attn.o_proj.weight"),
            gQ=h16(p+"self_attn.q_norm.weight"),  gK=h16(p+"self_attn.k_norm.weight"),
            gN1=h16(p+"input_layernorm.weight"),  gN2=h16(p+"post_attention_layernorm.weight"),
            Wg=h16T(p+"mlp.gate_proj.weight"), Wu=h16T(p+"mlp.up_proj.weight"), Wd=h16T(p+"mlp.down_proj.weight")))
        if i % 8 == 0: gc.collect()
    del sd; gc.collect()
    C_ = {"H": cfg.hidden_size, "L": L, "NQ": cfg.num_attention_heads,
          "NKV": cfg.num_key_value_heads, "HD": cfg.head_dim, "I": cfg.intermediate_size,
          "V": cfg.vocab_size, "eps": float(cfg.rms_norm_eps)}
    nparams = C_["V"]*C_["H"] + L*(  # rough, for the log
        3*C_["H"]*C_["I"] + 2*C_["H"]*(C_["NQ"]*C_["HD"]) + 2*C_["H"]*(C_["NKV"]*C_["HD"]))
    print(f"config: {C_}  (~{nparams/1e9:.1f}B params est., fp16 weights ~{nparams*2/1e9:.0f} GB)", flush=True)

    # --- ZLUDA forward (fp32 GEMM on the same fp16 weights; per-layer device frees) ---
    q = Q.QwenZLUDA()
    logits = Q.forward(q, W, C_, prompt_ids)
    zt = np.argsort(-logits)[:10].tolist()
    zv = [round(float(logits[i]), 3) for i in zt]
    print("ZLUDA top ids:    ", zt)
    print("CPU   top ids:    ", ref_ids)
    print("ZLUDA top logits: ", zv)
    print("CPU   top logits: ", ref_val)
    argmax_ok = zt[0] == ref_ids[0]
    top5_ok   = zt[:5] == ref_ids[:5]
    top10_set = set(zt) == set(ref_ids)
    maxd = max(abs(zv[i]-ref_val[i]) for i in range(10))
    print(f"argmax_match={argmax_ok} top5_match={top5_ok} top10_setmatch={top10_set} "
          f"max_logit_diff={maxd:.3f}", flush=True)
    # EVAL grade for the fp16/bf16 frontier tier: top-5 next-token agreement is the pass bar.
    print("VERDICT:", "PASS" if (argmax_ok and top5_ok) else "FAIL")
    return 0 if (argmax_ok and top5_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
