#!/usr/bin/env python3
"""Rung-2.75: a DEEP STACK of Qwen3-style decoder layers through ZLUDA on gfx1151.

Stacks N decoder layers (each: RMSNorm->QKV cuBLAS->QK-norm->RoPE->GQA attn->O-proj cuBLAS
->residual->RMSNorm->SwiGLU MLP 3xcuBLAS->residual) + a final RMSNorm + tied LM head (cuBLAS),
keeping the residual stream device-resident across all layers — i.e. the structure of a whole
transformer forward, with random weights graded against an independent pure-Python fp64 oracle.
Proves the forward pass composes AT DEPTH (the dimension a whole model adds over one layer).
Real Qwen3-0.6B weights/tokenizer are the remaining rung-3 piece (see ZLUDA-MORNING-SUMMARY.md).

Run: LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib python3 zluda_stack.py [nlayers] [seed]
"""
import ctypes as C, os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_layer import Z, S, H, NQ, NKV, HD, I, THETA, EPS, mm, rmsnorm, add, rope_heads, qknorm, gqa_attn, swiglu, flat

NLAYERS = int(sys.argv[1]) if len(sys.argv)>1 else 6
VOCAB = 48
random.seed(int(sys.argv[2]) if len(sys.argv)>2 else 314)

def rnd(n): return [random.uniform(-0.4,0.4) for _ in range(n)]
def mat(r,c): return [rnd(c) for _ in range(r)]

# per-layer weights
LW=[]
for _ in range(NLAYERS):
    LW.append(dict(Wq=mat(H,NQ*HD),Wk=mat(H,NKV*HD),Wv=mat(H,NKV*HD),Wo=mat(NQ*HD,H),
                   Wg=mat(H,I),Wu=mat(H,I),Wd=mat(I,H),gN1=rnd(H),gN2=rnd(H),gQ=rnd(HD),gK=rnd(HD)))
gFinal=rnd(H); Wlm=mat(H,VOCAB)
X0=mat(S,H)

# ---- reference (pure python fp64) ----
def ref_layer(x,w):
    xn=rmsnorm(x,w['gN1']); Q=mm(xn,w['Wq']); K=mm(xn,w['Wk']); V=mm(xn,w['Wv'])
    Q=qknorm(Q,NQ,w['gQ']); K=qknorm(K,NKV,w['gK']); Q=rope_heads(Q,NQ); K=rope_heads(K,NKV)
    o=mm(gqa_attn(Q,K,V),w['Wo']); x=add(x,o)
    xn2=rmsnorm(x,w['gN2']); sw=swiglu(mm(xn2,w['Wg']),mm(xn2,w['Wu'])); return add(x,mm(sw,w['Wd']))
def reference():
    x=X0
    for w in LW: x=ref_layer(x,w)
    x=rmsnorm(x,gFinal); return mm(x,Wlm)   # logits (S,VOCAB)

# ---- ZLUDA path ----
def zluda():
    z=Z()
    def dv(m): return z.dev(flat(m) if isinstance(m[0],list) else m)
    dx=dv(X0)
    DW=[{k:dv(v) for k,v in w.items()} for w in LW]
    dgF=dv(gFinal); dWlm=dv(Wlm)
    def K1(name,grid,blk,ps): z.launch(name,grid,blk,ps)
    P=lambda d:C.c_void_p(d.value)
    for w in DW:
        dxn=z.alloc(S*H); K1("rmsnorm",(S,1,1),(1,1,1),[P(dx),P(w['gN1']),P(dxn),C.c_int(S),C.c_int(H),C.c_float(EPS)])
        dQ=z.gemm(dxn,w['Wq'],S,H,NQ*HD); dK=z.gemm(dxn,w['Wk'],S,H,NKV*HD); dV=z.gemm(dxn,w['Wv'],S,H,NKV*HD)
        dQn=z.alloc(S*NQ*HD); K1("qknorm",(S,1,1),(1,1,1),[P(dQ),P(w['gQ']),P(dQn),C.c_int(S),C.c_int(NQ*HD),C.c_int(NQ),C.c_int(HD),C.c_float(EPS)])
        dKn=z.alloc(S*NKV*HD); K1("qknorm",(S,1,1),(1,1,1),[P(dK),P(w['gK']),P(dKn),C.c_int(S),C.c_int(NKV*HD),C.c_int(NKV),C.c_int(HD),C.c_float(EPS)])
        dQr=z.alloc(S*NQ*HD); K1("rope",(S,1,1),(1,1,1),[P(dQn),P(dQr),C.c_int(S),C.c_int(NQ*HD),C.c_int(NQ),C.c_int(HD),C.c_float(THETA)])
        dKr=z.alloc(S*NKV*HD); K1("rope",(S,1,1),(1,1,1),[P(dKn),P(dKr),C.c_int(S),C.c_int(NKV*HD),C.c_int(NKV),C.c_int(HD),C.c_float(THETA)])
        dao=z.alloc(S*NQ*HD); K1("gqa",(1,1,1),(1,1,1),[P(dQr),P(dKr),P(dV),P(dao),C.c_int(S),C.c_int(NQ),C.c_int(NKV),C.c_int(HD)])
        do=z.gemm(dao,w['Wo'],S,NQ*HD,H)
        dx1=z.alloc(S*H); K1("addk",((S*H+63)//64,1,1),(64,1,1),[P(dx),P(do),P(dx1),C.c_int(S*H)])
        dxn2=z.alloc(S*H); K1("rmsnorm",(S,1,1),(1,1,1),[P(dx1),P(w['gN2']),P(dxn2),C.c_int(S),C.c_int(H),C.c_float(EPS)])
        dg=z.gemm(dxn2,w['Wg'],S,H,I); du=z.gemm(dxn2,w['Wu'],S,H,I)
        dsw=z.alloc(S*I); K1("swiglu",((S*I+63)//64,1,1),(64,1,1),[P(dg),P(du),P(dsw),C.c_int(S*I)])
        dd=z.gemm(dsw,w['Wd'],S,I,H)
        dx=z.alloc(S*H); K1("addk",((S*H+63)//64,1,1),(64,1,1),[P(dx1),P(dd),P(dx),C.c_int(S*H)])
    dxf=z.alloc(S*H); K1("rmsnorm",(S,1,1),(1,1,1),[P(dx),P(dgF),P(dxf),C.c_int(S),C.c_int(H),C.c_float(EPS)])
    dlog=z.gemm(dxf,dWlm,S,H,VOCAB)
    fo=z.get(dlog,S*VOCAB); return [fo[i*VOCAB:(i+1)*VOCAB] for i in range(S)]

def main():
    ref=reference(); got=zluda()
    mxa=mxr=0.0; nf=0; rtol,atol=2e-4,2e-5
    for i in range(S):
        for j in range(VOCAB):
            a=abs(got[i][j]-ref[i][j]); r=a/(abs(ref[i][j])+1e-12); mxa=max(mxa,a); mxr=max(mxr,r)
            if a>atol+rtol*abs(ref[i][j]): nf+=1
    # also check argmax of last token matches (the decode-relevant quantity)
    am_ref=max(range(VOCAB),key=lambda j:ref[-1][j]); am_got=max(range(VOCAB),key=lambda j:got[-1][j])
    print(f"{NLAYERS}-layer stack through ZLUDA: max_abs={mxa:.3e} max_rel={mxr:.3e} n_fail={nf}/{S*VOCAB} argmax_ref={am_ref} argmax_zluda={am_got}")
    ok = (nf==0 and am_ref==am_got)
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__": sys.exit(main())
