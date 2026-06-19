#!/usr/bin/env python3
"""Scale demo: run a LARGER Qwen3 through ZLUDA on gfx1151, self-graded vs a live
torch-CPU-fp32 reference (no fixture needed). Supports the unified-memory thesis
(gfx1151 ~96 GB unified -> multi-billion-param models fit on one APU).

Reuses the rung-3 forward (zluda_qwen.forward) — entirely ZLUDA cuBLAS->rocBLAS +
nvrtc->PTX kernels, no torch CUDA kernels. One forward; compares top-10 next-token.

Run: ... python3 zluda_scale.py Qwen/Qwen3-1.7B
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zluda_qwen as Q

def main():
    name = sys.argv[1] if len(sys.argv)>1 else "Qwen/Qwen3-1.7B"
    prompt_ids = [785, 6722, 315, 9625, 374]   # "The capital of France is"
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    print(f"loading {name} on CPU ...", flush=True)
    cfg=AutoConfig.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    # live CPU reference (oracle)
    with torch.no_grad():
        ref=model(torch.tensor([prompt_ids])).logits[0,-1].float()
    rtop=torch.topk(ref,10); ref_ids=rtop.indices.tolist(); ref_val=[round(v,3) for v in rtop.values.tolist()]
    # extract weights for ZLUDA forward (reuse zluda_qwen helpers via the loaded model)
    sd=model.state_dict()
    npT=lambda t:t.detach().numpy().astype(np.float32).T.copy(); npv=lambda t:t.detach().numpy().astype(np.float32).copy()
    inv=None
    for m in model.modules():
        if getattr(m,"inv_freq",None) is not None: inv=m.inv_freq.detach().numpy().astype(np.float32).copy(); break
    L=cfg.num_hidden_layers
    W={"embed":npv(sd["model.embed_tokens.weight"]),"final_norm":npv(sd["model.norm.weight"]),"inv_freq":inv,"layers":[]}
    if "lm_head.weight" in sd: W["lm_head"]=npv(sd["lm_head.weight"])   # untied LM head (e.g. Qwen3-8B)
    for i in range(L):
        p=f"model.layers.{i}."
        W["layers"].append(dict(Wq=npT(sd[p+"self_attn.q_proj.weight"]),Wk=npT(sd[p+"self_attn.k_proj.weight"]),
            Wv=npT(sd[p+"self_attn.v_proj.weight"]),Wo=npT(sd[p+"self_attn.o_proj.weight"]),
            gQ=npv(sd[p+"self_attn.q_norm.weight"]),gK=npv(sd[p+"self_attn.k_norm.weight"]),
            gN1=npv(sd[p+"input_layernorm.weight"]),gN2=npv(sd[p+"post_attention_layernorm.weight"]),
            Wg=npT(sd[p+"mlp.gate_proj.weight"]),Wu=npT(sd[p+"mlp.up_proj.weight"]),Wd=npT(sd[p+"mlp.down_proj.weight"])))
    C_={"H":cfg.hidden_size,"L":L,"NQ":cfg.num_attention_heads,"NKV":cfg.num_key_value_heads,
        "HD":cfg.head_dim,"I":cfg.intermediate_size,"V":cfg.vocab_size,"eps":float(cfg.rms_norm_eps)}
    nparams=sum(int(np.prod(t.shape)) for t in sd.values())
    print(f"config: {C_}  (~{nparams/1e9:.2f}B params, fp32 weights ~{nparams*4/1e9:.1f} GB)", flush=True)
    del model
    q=Q.QwenZLUDA()
    logits=Q.forward(q,W,C_,prompt_ids)
    zt=np.argsort(-logits)[:10].tolist(); zv=[round(float(logits[i]),3) for i in zt]
    print("ZLUDA top ids:  ",zt)
    print("CPU   top ids:  ",ref_ids)
    print("ZLUDA top logits:",zv)
    print("CPU   top logits:",ref_val)
    argmax_ok=zt[0]==ref_ids[0]; top5_ok=zt[:5]==ref_ids[:5]
    maxd=max(abs(zv[i]-ref_val[i]) for i in range(10))
    print(f"argmax_match={argmax_ok} top5_match={top5_ok} max_logit_diff={maxd:.3f}")
    print("VERDICT:", "PASS" if (argmax_ok and top5_ok) else "FAIL")
    return 0 if (argmax_ok and top5_ok) else 1

if __name__=="__main__": sys.exit(main())
