#!/usr/bin/env python3
"""RUNG 3: real Qwen3-0.6B forward through ZLUDA on gfx1151, graded vs the fixture.

Loads Qwen3-0.6B on CPU via transformers (no GPU kernels — just weights/config), then
runs the forward pass entirely through ZLUDA primitives proven in rungs 1-2.75:
  embedding gather -> [28x: RMSNorm -> q/k/v proj (cuBLAS->rocBLAS) -> per-head QK-norm
  -> RoPE -> causal GQA attention -> o_proj (cuBLAS) -> residual -> RMSNorm -> SwiGLU MLP
  (cuBLAS x3) -> residual] -> final RMSNorm -> LM head (cuBLAS, tied embeddings).
Grades the last-token top-10 (ids + logits) and the 8-token greedy continuation against
tests/fixtures/qwen3_06b_mps_smoke.json (torch-cpu-fp32 oracle). No torch CUDA kernels are
used (the wheel's are SASS-only) — every kernel here is nvrtc->PTX->ZLUDA or cuBLAS->rocBLAS.

Run: LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib python3 zluda_qwen.py
"""
import ctypes as C, os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_diff import Cuda, compile_ptx, CUDA, CUBLAS, _ck

KSRC = open(os.path.join(os.path.dirname(__file__),"zluda_layer.py")).read().split("KSRC=r'''")[1].split("'''")[0]
# Exact RoPE using the model's own inv_freq table (avoids assuming rope_theta).
KSRC += r'''
extern "C" __global__ void rope_f(const float* x,const float* invf,float* o,int rows,int width,int nh,int hd){
  int r=blockIdx.x; if(r>=rows) return; int half=hd/2; for(int k=0;k<width;k++) o[r*width+k]=x[r*width+k];
  for(int h=0;h<nh;h++){int b=h*hd; for(int i=0;i<half;i++){float ang=r*invf[i];float c=cosf(ang),s=sinf(ang);
    float x1=x[r*width+b+i],x2=x[r*width+b+half+i]; o[r*width+b+i]=x1*c-x2*s; o[r*width+b+half+i]=x1*s+x2*c; } } }
'''

CUDA.cuMemFree_v2.argtypes=[C.c_ulonglong]          # 64-bit device ptr — required or frees silently no-op
CUDA.cuMemAlloc_v2.argtypes=[C.POINTER(C.c_ulonglong),C.c_size_t]
# sizes must be c_size_t (64-bit) — else >2GB transfers truncate to 32-bit -> INVALID_VALUE
CUDA.cuMemcpyHtoD_v2.argtypes=[C.c_ulonglong,C.c_void_p,C.c_size_t]
CUDA.cuMemcpyDtoH_v2.argtypes=[C.c_void_p,C.c_ulonglong,C.c_size_t]
_CHUNK=1<<30   # 1 GiB — a single HtoD/DtoH >2GB fails (32-bit size path); chunk large transfers
def dev_np(a):
    a=np.ascontiguousarray(a,dtype=np.float32)
    d=C.c_ulonglong(); _ck(CUDA.cuMemAlloc_v2(C.byref(d),a.nbytes),"alloc")
    base=a.ctypes.data_as(C.c_void_p).value; off=0; n=a.nbytes
    while off<n:
        c=min(_CHUNK,n-off)
        _ck(CUDA.cuMemcpyHtoD_v2(C.c_ulonglong(d.value+off),C.c_void_p(base+off),c),"h2d")
        off+=c
    return d
def alloc(n):
    d=C.c_ulonglong(); _ck(CUDA.cuMemAlloc_v2(C.byref(d),n*4),"alloc"); return d
def free(*ds):
    for d in ds:
        try: CUDA.cuMemFree_v2(d)
        except Exception: pass
def get_np(d,n):
    a=np.empty(n,dtype=np.float32); _ck(CUDA.cuMemcpyDtoH_v2(a.ctypes.data_as(C.c_void_p),d,n*4),"d2h"); return a
P=lambda d:C.c_void_p(d.value)

class QwenZLUDA:
    def __init__(self):
        self.cu=Cuda(); self.m=self.cu.module(compile_ptx(KSRC))
        self.h=C.c_void_p(); _ck(CUBLAS.cublasCreate_v2(C.byref(self.h)),"cublasCreate")
        CUBLAS.cublasSgemm_v2.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
            C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int]
    def gemm(self,dA,dB,M,K,N):
        dC=alloc(M*N); a=C.c_float(1.0); b=C.c_float(0.0)
        _ck(CUBLAS.cublasSgemm_v2(self.h,0,0,N,M,K,C.byref(a),P(dB),N,P(dA),K,C.byref(b),P(dC),N),"sgemm")
        _ck(CUDA.cuCtxSynchronize(),"sync"); return dC
    def k(self,name,grid,blk,ps):
        f=self.cu.func(self.m,name); self.cu.launch(f,grid,blk,ps)

def load_weights():
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    name="Qwen/Qwen3-0.6B"
    cfg=AutoConfig.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    sd=model.state_dict()
    # exact inv_freq table from the model's rotary embedding (len head_dim/2)
    inv_freq=None
    for mod in model.modules():
        if hasattr(mod,"inv_freq") and getattr(mod,"inv_freq") is not None:
            inv_freq=mod.inv_freq.detach().numpy().astype(np.float32).copy(); break
    def npT(t): return t.detach().numpy().astype(np.float32).T.copy()   # Linear [out,in] -> [in,out]
    def npv(t): return t.detach().numpy().astype(np.float32).copy()
    L=cfg.num_hidden_layers
    W={"embed":npv(sd["model.embed_tokens.weight"]),  # [V,H]
       "final_norm":npv(sd["model.norm.weight"]), "layers":[]}
    if "lm_head.weight" in sd: W["lm_head"]=npv(sd["lm_head.weight"])   # untied LM head
    for i in range(L):
        p=f"model.layers.{i}."
        W["layers"].append(dict(
            Wq=npT(sd[p+"self_attn.q_proj.weight"]), Wk=npT(sd[p+"self_attn.k_proj.weight"]),
            Wv=npT(sd[p+"self_attn.v_proj.weight"]), Wo=npT(sd[p+"self_attn.o_proj.weight"]),
            gQ=npv(sd[p+"self_attn.q_norm.weight"]), gK=npv(sd[p+"self_attn.k_norm.weight"]),
            gN1=npv(sd[p+"input_layernorm.weight"]), gN2=npv(sd[p+"post_attention_layernorm.weight"]),
            Wg=npT(sd[p+"mlp.gate_proj.weight"]), Wu=npT(sd[p+"mlp.up_proj.weight"]), Wd=npT(sd[p+"mlp.down_proj.weight"])))
    C_={"H":cfg.hidden_size,"L":L,"NQ":cfg.num_attention_heads,"NKV":cfg.num_key_value_heads,
        "HD":cfg.head_dim,"I":cfg.intermediate_size,"V":cfg.vocab_size,
        "eps":float(cfg.rms_norm_eps)}
    W["inv_freq"]=inv_freq
    return W,C_

def forward(q, W, cfg, ids):
    H,NQ,NKV,HD,I,V=cfg["H"],cfg["NQ"],cfg["NKV"],cfg["HD"],cfg["I"],cfg["V"]
    eps=cfg["eps"]; S=len(ids)
    QC,KC=NQ*HD,NKV*HD
    dInv=dev_np(W["inv_freq"])
    # embedding gather (host) -> device
    emb=W["embed"][np.array(ids)]   # (S,H)
    dx=dev_np(emb)
    # device weights cached per call (fine for a few forwards)
    for w in W["layers"]:
        dWq=dev_np(w["Wq"]);dWk=dev_np(w["Wk"]);dWv=dev_np(w["Wv"]);dWo=dev_np(w["Wo"])
        dWg=dev_np(w["Wg"]);dWu=dev_np(w["Wu"]);dWd=dev_np(w["Wd"])
        dgN1=dev_np(w["gN1"]);dgN2=dev_np(w["gN2"]);dgQ=dev_np(w["gQ"]);dgK=dev_np(w["gK"])
        dxn=alloc(S*H); q.k("rmsnorm",(S,1,1),(1,1,1),[P(dx),P(dgN1),P(dxn),C.c_int(S),C.c_int(H),C.c_float(eps)])
        dQ=q.gemm(dxn,dWq,S,H,QC); dK=q.gemm(dxn,dWk,S,H,KC); dV=q.gemm(dxn,dWv,S,H,KC)
        dQn=alloc(S*QC); q.k("qknorm",(S,1,1),(1,1,1),[P(dQ),P(dgQ),P(dQn),C.c_int(S),C.c_int(QC),C.c_int(NQ),C.c_int(HD),C.c_float(eps)])
        dKn=alloc(S*KC); q.k("qknorm",(S,1,1),(1,1,1),[P(dK),P(dgK),P(dKn),C.c_int(S),C.c_int(KC),C.c_int(NKV),C.c_int(HD),C.c_float(eps)])
        dQr=alloc(S*QC); q.k("rope_f",(S,1,1),(1,1,1),[P(dQn),P(dInv),P(dQr),C.c_int(S),C.c_int(QC),C.c_int(NQ),C.c_int(HD)])
        dKr=alloc(S*KC); q.k("rope_f",(S,1,1),(1,1,1),[P(dKn),P(dInv),P(dKr),C.c_int(S),C.c_int(KC),C.c_int(NKV),C.c_int(HD)])
        dao=alloc(S*QC); q.k("gqa",(1,1,1),(1,1,1),[P(dQr),P(dKr),P(dV),P(dao),C.c_int(S),C.c_int(NQ),C.c_int(NKV),C.c_int(HD)])
        do=q.gemm(dao,dWo,S,QC,H)
        dx1=alloc(S*H); q.k("addk",((S*H+63)//64,1,1),(64,1,1),[P(dx),P(do),P(dx1),C.c_int(S*H)])
        dxn2=alloc(S*H); q.k("rmsnorm",(S,1,1),(1,1,1),[P(dx1),P(dgN2),P(dxn2),C.c_int(S),C.c_int(H),C.c_float(eps)])
        dg=q.gemm(dxn2,dWg,S,H,I); du=q.gemm(dxn2,dWu,S,H,I)
        dsw=alloc(S*I); q.k("swiglu",((S*I+63)//64,1,1),(64,1,1),[P(dg),P(du),P(dsw),C.c_int(S*I)])
        dd=q.gemm(dsw,dWd,S,I,H)
        dxnew=alloc(S*H); q.k("addk",((S*H+63)//64,1,1),(64,1,1),[P(dx1),P(dd),P(dxnew),C.c_int(S*H)])
        # free this layer's device weights + intermediates (bounds memory for large models)
        free(dx,dWq,dWk,dWv,dWo,dWg,dWu,dWd,dgN1,dgN2,dgQ,dgK,dxn,dQ,dK,dV,dQn,dKn,dQr,dKr,dao,do,dx1,dxn2,dg,du,dsw,dd)
        dx=dxnew
    dxf=alloc(S*H); q.k("rmsnorm",(S,1,1),(1,1,1),[P(dx),P(dev_np(W["final_norm"])),P(dxf),C.c_int(S),C.c_int(H),C.c_float(eps)])
    # last-token logits only: (1,H) @ embed^T (H,V)
    last=get_np(dxf,S*H).reshape(S,H)[-1:].copy()   # (1,H)
    dlast=dev_np(last); dWlm=dev_np(W.get("lm_head", W["embed"]).T)    # [H,V]; untied models have separate lm_head
    dlog=q.gemm(dlast,dWlm,1,H,V)
    return get_np(dlog,V)

def main():
    fix=json.load(open(os.path.join(os.path.dirname(__file__),"fixtures","qwen3_06b_mps_smoke.json")))
    ids=fix["input_ids"]; exp_top=fix["cpu_fp32_top"]; exp_cont=fix["mps_greedy_continuation_ids"]
    print("loading Qwen3-0.6B weights on CPU ...", flush=True)
    W,cfg=load_weights(); print("config:",cfg, flush=True)
    q=QwenZLUDA()
    logits=forward(q,W,cfg,ids)
    top_idx=np.argsort(-logits)[:10].tolist()
    top_val=[round(float(logits[i]),4) for i in top_idx]
    print("ZLUDA top ids:   ",top_idx)
    print("fixture top ids: ",exp_top["ids"])
    print("ZLUDA top logits:",top_val)
    print("fixture logits:  ",exp_top["logits"])
    argmax_ok = top_idx[0]==exp_top["ids"][0]
    top5_ok = top_idx[:5]==exp_top["ids"][:5]
    maxld=max(abs(top_val[i]-exp_top["logits"][i]) for i in range(min(len(top_val),len(exp_top["logits"]))))
    print(f"argmax_match={argmax_ok} top5_match={top5_ok} max_logit_diff={maxld:.3f}")
    # greedy continuation
    cont=[]; cur=list(ids)
    for _ in range(len(exp_cont)):
        lg=forward(q,W,cfg,cur); nxt=int(np.argmax(lg)); cont.append(nxt); cur.append(nxt)
    print("ZLUDA continuation:",cont)
    print("fixture continuation:",exp_cont)
    cont_ok = cont==exp_cont
    print("VERDICT:", "PASS" if (argmax_ok and cont_ok) else ("PARTIAL" if argmax_ok else "FAIL"))
    return 0 if (argmax_ok and cont_ok) else 1

if __name__=="__main__": sys.exit(main())
