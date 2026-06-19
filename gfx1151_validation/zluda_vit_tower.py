#!/usr/bin/env python3
"""RUNG 7.75 (REAL WEIGHTS, depth): the FULL 24-block Qwen3-VL-2B vision-block stack through ZLUDA.

Rung 7.5 proved one real Qwen3VLVisionBlock matches HF. This chains ALL `depth` real vision
blocks (residual stream device-resident) through ZLUDA and grades the final hidden states vs
HF running the same real blocks sequentially on identical inputs — the real-vision analog of
rung 2.75 (deep decoder stack). It answers the depth question a single-block test can't: does
fp32 error ACCUMULATE across two-dozen composed real blocks, or stay bounded?

Both paths get identical initial hidden_states + rotary cos/sin (isolating ZLUDA execution from
HF grid logic). The vision-model's pos-embed interpolation / deepstack / merger are HF host glue
(no new ZLUDA-translatable op vs the block) and are out of scope; the block is the repeating unit.

Run: LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 zluda_vit_tower.py [seed]
"""
import ctypes as C, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_diff import Cuda, compile_ptx, CUDA, CUBLAS, _ck
from zluda_vit_real import KSRC, Z, P, MODEL, SEQ

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20260619
DEPTH_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0 = all blocks; else first N

def zblock(z, dH, W, dcos, dsin, SEQ, NH, HD, INT, EPS=1e-6):
    Wd = NH*HD
    dn1 = z.alloc(SEQ*Wd); z.k("layernorm",(SEQ,1,1),(1,1,1),[P(dH),P(W['dg1']),P(W['db1']),P(dn1),C.c_int(SEQ),C.c_int(Wd),C.c_float(EPS)])
    dqkv = z.gemm(dn1, W['dWqkv'], SEQ, Wd, 3*Wd)
    z.k("bias_add",((SEQ*3*Wd+255)//256,1,1),(256,1,1),[P(dqkv),P(W['dbqkv']),C.c_int(SEQ),C.c_int(3*Wd)])
    qkv = z.get(dqkv, SEQ*3*Wd).reshape(SEQ, 3, NH, HD)
    dQ = z.dev(qkv[:,0].reshape(SEQ, Wd)); dK = z.dev(qkv[:,1].reshape(SEQ, Wd)); dV = z.dev(qkv[:,2].reshape(SEQ, Wd))
    dQr = z.alloc(SEQ*Wd); z.k("rope_apply",(SEQ,1,1),(1,1,1),[P(dQ),P(dcos),P(dsin),P(dQr),C.c_int(SEQ),C.c_int(Wd),C.c_int(NH),C.c_int(HD)])
    dKr = z.alloc(SEQ*Wd); z.k("rope_apply",(SEQ,1,1),(1,1,1),[P(dK),P(dcos),P(dsin),P(dKr),C.c_int(SEQ),C.c_int(Wd),C.c_int(NH),C.c_int(HD)])
    dao = z.alloc(SEQ*Wd); z.k("mha",(1,1,1),(1,1,1),[P(dQr),P(dKr),P(dV),P(dao),C.c_int(SEQ),C.c_int(NH),C.c_int(HD)])
    do = z.gemm(dao, W['dWproj'], SEQ, Wd, Wd)
    z.k("bias_add",((SEQ*Wd+255)//256,1,1),(256,1,1),[P(do),P(W['dbproj']),C.c_int(SEQ),C.c_int(Wd)])
    dx1 = z.alloc(SEQ*Wd); z.k("addk",((SEQ*Wd+255)//256,1,1),(256,1,1),[P(dH),P(do),P(dx1),C.c_int(SEQ*Wd)])
    dn2 = z.alloc(SEQ*Wd); z.k("layernorm",(SEQ,1,1),(1,1,1),[P(dx1),P(W['dg2']),P(W['db2']),P(dn2),C.c_int(SEQ),C.c_int(Wd),C.c_float(EPS)])
    dh1 = z.gemm(dn2, W['dWfc1'], SEQ, Wd, INT); z.k("bias_add",((SEQ*INT+255)//256,1,1),(256,1,1),[P(dh1),P(W['dbfc1']),C.c_int(SEQ),C.c_int(INT)])
    dhg = z.alloc(SEQ*INT); z.k("gelu",((SEQ*INT+255)//256,1,1),(256,1,1),[P(dh1),P(dhg),C.c_int(SEQ*INT)])
    dmlp = z.gemm(dhg, W['dWfc2'], SEQ, INT, Wd); z.k("bias_add",((SEQ*Wd+255)//256,1,1),(256,1,1),[P(dmlp),P(W['dbfc2']),C.c_int(SEQ),C.c_int(Wd)])
    dout = z.alloc(SEQ*Wd); z.k("addk",((SEQ*Wd+255)//256,1,1),(256,1,1),[P(dx1),P(dmlp),P(dout),C.c_int(SEQ*Wd)])
    return dout

def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoConfig
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = AutoConfig.from_pretrained(MODEL); vc = cfg.vision_config
    HVS = vc.hidden_size; NH = vc.num_heads; HD = HVS // NH; INT = vc.intermediate_size
    print(f"loading {MODEL} full vision-block stack (depth={vc.depth}) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    blocks = [m for n, m in model.named_modules() if m.__class__.__name__ == "Qwen3VLVisionBlock"]
    if DEPTH_LIMIT: blocks = blocks[:DEPTH_LIMIT]
    H = torch.randn(SEQ, HVS, dtype=torch.float32) * 0.5
    ang = torch.randn(SEQ, HD // 2, dtype=torch.float32); emb = torch.cat([ang, ang], -1)
    cos = emb.cos(); sin = emb.sin(); cu = torch.tensor([0, SEQ], dtype=torch.int32)
    with torch.no_grad():
        h = H
        for blk in blocks:
            h = blk(h, cu_seqlens=cu, position_embeddings=(cos, sin))
        ref = h.float().numpy()
    # ZLUDA: stage all block weights, chain on device
    z = Z()
    def WT(l): return l.weight.detach().numpy().astype(np.float32).T.copy()
    def BV(l): return l.bias.detach().numpy().astype(np.float32).copy()
    devW = []
    for blk in blocks:
        devW.append(dict(
            dg1=z.dev(blk.norm1.weight.detach().numpy()), db1=z.dev(blk.norm1.bias.detach().numpy()),
            dg2=z.dev(blk.norm2.weight.detach().numpy()), db2=z.dev(blk.norm2.bias.detach().numpy()),
            dWqkv=z.dev(WT(blk.attn.qkv)), dbqkv=z.dev(BV(blk.attn.qkv)),
            dWproj=z.dev(WT(blk.attn.proj)), dbproj=z.dev(BV(blk.attn.proj)),
            dWfc1=z.dev(WT(blk.mlp.linear_fc1)), dbfc1=z.dev(BV(blk.mlp.linear_fc1)),
            dWfc2=z.dev(WT(blk.mlp.linear_fc2)), dbfc2=z.dev(BV(blk.mlp.linear_fc2))))
    dcos = z.dev(cos.numpy()); dsin = z.dev(sin.numpy())
    dH = z.dev(H.numpy())
    for W in devW:
        dH = zblock(z, dH, W, dcos, dsin, SEQ, NH, HD, INT)
    got = z.get(dH, SEQ*HVS).reshape(SEQ, HVS)
    diff = np.abs(got - ref); rel = diff / (np.abs(ref) + 1e-6)
    rtol, atol = 3e-3, 3e-3
    nf = int(np.sum(diff > atol + rtol*np.abs(ref)))
    print(f"REAL Qwen3-VL-2B {len(blocks)}-block vision stack through ZLUDA: "
          f"max_abs={diff.max():.3e} max_rel={rel.max():.3e} n_fail={nf}/{got.size}", flush=True)
    print("VERDICT:", "PASS" if nf == 0 else "FAIL")
    return 0 if nf == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
