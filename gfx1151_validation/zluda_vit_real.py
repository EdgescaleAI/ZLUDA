#!/usr/bin/env python3
"""RUNG 7.5 (REAL WEIGHTS): a real Qwen3-VL-2B VISION block through ZLUDA on gfx1151.

Rung 6.5 proved the vision op set COMPOSES (synthetic spec). Rung 7 ran the REAL text tower.
This closes the gap on the vision side: it runs an ACTUAL `Qwen/Qwen3-VL-2B-Instruct`
vision-transformer block (real Conv3d-patch-embed-trained weights) through ZLUDA, graded
against HF's own forward of that exact block (`Qwen3VLVisionBlock`).

The block (from HF modeling_qwen3_vl.py): h += attn(LayerNorm(h)); h += mlp(LayerNorm(h)),
with a FUSED qkv Linear(+bias), rotate_half 2D-rope, FULL (non-causal) MHA over the patch
sequence (single image -> one cu_seqlens window), proj(+bias), and an MLP fc1->GELU(tanh)->fc2
(+biases). HF's vision MLP act is `gelu_pytorch_tanh`, matching the tanh-GELU kernel here.

Method: feed IDENTICAL hidden_states and rotary cos/sin to both HF's block and the ZLUDA
reproduction (so this isolates ZLUDA's execution of the REAL block math, not HF grid logic),
and diff the block outputs at tight fp32 tolerance. Weights are the real trained tensors.

Run: LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 zluda_vit_real.py [seed]
"""
import ctypes as C, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_diff import Cuda, compile_ptx, CUDA, CUBLAS, _ck

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
SEQ = 16                         # patch tokens (single fake image window)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20260619
BLOCK_IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # which vision block to validate

KSRC = r'''
extern "C" __global__ void layernorm(const float* x,const float* g,const float* b,float* o,int rows,int cols,float eps){
  int r=blockIdx.x; if(r>=rows) return; float mean=0; for(int j=0;j<cols;j++) mean+=x[r*cols+j]; mean/=cols;
  float var=0; for(int j=0;j<cols;j++){float d=x[r*cols+j]-mean; var+=d*d;} var/=cols; float inv=rsqrtf(var+eps);
  for(int j=0;j<cols;j++) o[r*cols+j]=(x[r*cols+j]-mean)*inv*g[j]+b[j]; }
extern "C" __global__ void bias_add(float* x,const float* b,int rows,int cols){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<rows*cols) x[i]+=b[i%cols]; }
extern "C" __global__ void rope_apply(const float* x,const float* cosv,const float* sinv,float* o,int rows,int width,int nh,int hd){
  // rotate_half rope given per-token cos/sin of length hd; applied within each head block
  int r=blockIdx.x; if(r>=rows) return; int half=hd/2;
  for(int h=0;h<nh;h++){int base=h*hd;
    for(int i=0;i<half;i++){ float c=cosv[r*hd+i], s=sinv[r*hd+i];
      float x1=x[r*width+base+i], x2=x[r*width+base+half+i];
      o[r*width+base+i]=x1*c - x2*s; o[r*width+base+half+i]=x2*c + x1*s; } } }
extern "C" __global__ void mha(const float* Q,const float* K,const float* V,float* O,int S,int nh,int hd){
  float scale=rsqrtf((float)hd);
  for(int h=0;h<nh;h++){int b=h*hd; int c=nh*hd;
    for(int i=0;i<S;i++){ float sc[128]; float m=-1e30f;
      for(int j=0;j<S;j++){ float d=0; for(int t=0;t<hd;t++) d+=Q[i*c+b+t]*K[j*c+b+t]; sc[j]=d*scale; if(sc[j]>m)m=sc[j]; }
      float ss=0; for(int j=0;j<S;j++){float e=expf(sc[j]-m); sc[j]=e; ss+=e;}
      for(int t=0;t<hd;t++){float acc=0; for(int j=0;j<S;j++) acc+=(sc[j]/ss)*V[j*c+b+t]; O[i*c+b+t]=acc;} } } }
extern "C" __global__ void gelu(const float* x,float* o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){float v=x[i];
    float c=0.7978845608028654f; o[i]=0.5f*v*(1.f+tanhf(c*(v+0.044715f*v*v*v)));} }
extern "C" __global__ void addk(const float* a,const float* b,float* o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]+b[i]; }
'''

class Z:
    def __init__(self):
        self.cu = Cuda(); self.m = self.cu.module(compile_ptx(KSRC))
        self.h = C.c_void_p(); _ck(CUBLAS.cublasCreate_v2(C.byref(self.h)), "cublasCreate")
        CUBLAS.cublasSgemm_v2.argtypes = [C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
            C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int]
    def dev(self, a):
        a = np.ascontiguousarray(a, dtype=np.float32); d = C.c_ulonglong()
        _ck(CUDA.cuMemAlloc_v2(C.byref(d), a.nbytes), "alloc")
        _ck(CUDA.cuMemcpyHtoD_v2(d, a.ctypes.data_as(C.c_void_p), a.nbytes), "h2d"); return d
    def alloc(self, n):
        d = C.c_ulonglong(); _ck(CUDA.cuMemAlloc_v2(C.byref(d), n*4), "alloc"); return d
    def get(self, d, n):
        a = np.empty(n, dtype=np.float32); _ck(CUDA.cuMemcpyDtoH_v2(a.ctypes.data_as(C.c_void_p), d, n*4), "d2h"); return a
    def gemm(self, dA, dB, M, K, N):
        dC = self.alloc(M*N); a = C.c_float(1.0); b = C.c_float(0.0)
        _ck(CUBLAS.cublasSgemm_v2(self.h,0,0,N,M,K,C.byref(a),C.c_void_p(dB.value),N,C.c_void_p(dA.value),K,C.byref(b),C.c_void_p(dC.value),N),"sgemm")
        _ck(CUDA.cuCtxSynchronize(), "sync"); return dC
    def k(self, name, grid, blk, ps): self.cu.launch(self.cu.func(self.m, name), grid, blk, ps)

P = lambda d: C.c_void_p(d.value)

def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoConfig
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = AutoConfig.from_pretrained(MODEL); vc = cfg.vision_config
    HD = vc.hidden_size // vc.num_heads; NH = vc.num_heads; HVS = vc.hidden_size; INT = vc.intermediate_size
    print(f"loading {MODEL} (vision block 0) ... hidden={HVS} heads={NH} head_dim={HD} inter={INT}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    blocks = [mod for n, mod in model.named_modules() if mod.__class__.__name__ == "Qwen3VLVisionBlock"]
    blk = blocks[BLOCK_IDX]
    print(f"validating vision block {BLOCK_IDX}/{len(blocks)-1}", flush=True)
    # identical inputs to both paths
    H = torch.randn(SEQ, HVS, dtype=torch.float32) * 0.5
    ang = torch.randn(SEQ, HD // 2, dtype=torch.float32)
    emb = torch.cat([ang, ang], dim=-1)            # (SEQ, HD)
    cos = emb.cos(); sin = emb.sin()
    cu = torch.tensor([0, SEQ], dtype=torch.int32)
    with torch.no_grad():
        ref = blk(H, cu_seqlens=cu, position_embeddings=(cos, sin)).float().numpy()
    # pull real block weights (Linear weight is [out,in]; GEMM wants [in,out] => .T)
    def WT(lin): return lin.weight.detach().numpy().astype(np.float32).T.copy()
    def BV(lin): return lin.bias.detach().numpy().astype(np.float32).copy()
    def NV(ln):  return (ln.weight.detach().numpy().astype(np.float32).copy(),
                         ln.bias.detach().numpy().astype(np.float32).copy())
    g1, b1 = NV(blk.norm1); g2, b2 = NV(blk.norm2)
    Wqkv = WT(blk.attn.qkv); bqkv = BV(blk.attn.qkv)
    Wproj = WT(blk.attn.proj); bproj = BV(blk.attn.proj)
    Wfc1 = WT(blk.mlp.linear_fc1); bfc1 = BV(blk.mlp.linear_fc1)
    Wfc2 = WT(blk.mlp.linear_fc2); bfc2 = BV(blk.mlp.linear_fc2)

    z = Z(); EPS = 1e-6; W = HVS
    dH = z.dev(H.numpy())
    dcos = z.dev(cos.numpy()); dsin = z.dev(sin.numpy())
    dg1 = z.dev(g1); db1 = z.dev(b1); dg2 = z.dev(g2); db2 = z.dev(b2)
    dWqkv = z.dev(Wqkv); dbqkv = z.dev(bqkv); dWproj = z.dev(Wproj); dbproj = z.dev(bproj)
    dWfc1 = z.dev(Wfc1); dbfc1 = z.dev(bfc1); dWfc2 = z.dev(Wfc2); dbfc2 = z.dev(bfc2)
    # norm1
    dn1 = z.alloc(SEQ*W); z.k("layernorm",(SEQ,1,1),(1,1,1),[P(dH),P(dg1),P(db1),P(dn1),C.c_int(SEQ),C.c_int(W),C.c_float(EPS)])
    # fused qkv + bias
    dqkv = z.gemm(dn1, dWqkv, SEQ, W, 3*W)
    z.k("bias_add",((SEQ*3*W+255)//256,1,1),(256,1,1),[P(dqkv),P(dbqkv),C.c_int(SEQ),C.c_int(3*W)])
    qkv = z.get(dqkv, SEQ*3*W).reshape(SEQ, 3, NH, HD)
    dQ = z.dev(qkv[:,0].reshape(SEQ, NH*HD)); dK = z.dev(qkv[:,1].reshape(SEQ, NH*HD)); dV = z.dev(qkv[:,2].reshape(SEQ, NH*HD))
    # rope on q,k
    dQr = z.alloc(SEQ*NH*HD); z.k("rope_apply",(SEQ,1,1),(1,1,1),[P(dQ),P(dcos),P(dsin),P(dQr),C.c_int(SEQ),C.c_int(NH*HD),C.c_int(NH),C.c_int(HD)])
    dKr = z.alloc(SEQ*NH*HD); z.k("rope_apply",(SEQ,1,1),(1,1,1),[P(dK),P(dcos),P(dsin),P(dKr),C.c_int(SEQ),C.c_int(NH*HD),C.c_int(NH),C.c_int(HD)])
    # full attention
    dao = z.alloc(SEQ*NH*HD); z.k("mha",(1,1,1),(1,1,1),[P(dQr),P(dKr),P(dV),P(dao),C.c_int(SEQ),C.c_int(NH),C.c_int(HD)])
    # proj + bias + residual
    do = z.gemm(dao, dWproj, SEQ, W, W)
    z.k("bias_add",((SEQ*W+255)//256,1,1),(256,1,1),[P(do),P(dbproj),C.c_int(SEQ),C.c_int(W)])
    dx1 = z.alloc(SEQ*W); z.k("addk",((SEQ*W+255)//256,1,1),(256,1,1),[P(dH),P(do),P(dx1),C.c_int(SEQ*W)])
    # norm2 + mlp
    dn2 = z.alloc(SEQ*W); z.k("layernorm",(SEQ,1,1),(1,1,1),[P(dx1),P(dg2),P(db2),P(dn2),C.c_int(SEQ),C.c_int(W),C.c_float(EPS)])
    dh1 = z.gemm(dn2, dWfc1, SEQ, W, INT); z.k("bias_add",((SEQ*INT+255)//256,1,1),(256,1,1),[P(dh1),P(dbfc1),C.c_int(SEQ),C.c_int(INT)])
    dhg = z.alloc(SEQ*INT); z.k("gelu",((SEQ*INT+255)//256,1,1),(256,1,1),[P(dh1),P(dhg),C.c_int(SEQ*INT)])
    dmlp = z.gemm(dhg, dWfc2, SEQ, INT, W); z.k("bias_add",((SEQ*W+255)//256,1,1),(256,1,1),[P(dmlp),P(dbfc2),C.c_int(SEQ),C.c_int(W)])
    dout = z.alloc(SEQ*W); z.k("addk",((SEQ*W+255)//256,1,1),(256,1,1),[P(dx1),P(dmlp),P(dout),C.c_int(SEQ*W)])
    got = z.get(dout, SEQ*W).reshape(SEQ, W)

    diff = np.abs(got - ref); rel = diff / (np.abs(ref) + 1e-6)
    rtol, atol = 2e-3, 2e-3
    nf = int(np.sum(diff > atol + rtol*np.abs(ref)))
    print(f"REAL Qwen3-VL-2B vision block {BLOCK_IDX} through ZLUDA: max_abs={diff.max():.3e} max_rel={rel.max():.3e} "
          f"n_fail={nf}/{got.size}", flush=True)
    print("VERDICT:", "PASS" if nf == 0 else "FAIL")
    return 0 if nf == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
