#!/usr/bin/env python3
"""Rung-6.75: the composed cross-modal VLM FUSION seam through ZLUDA on gfx1151.

Rungs 0->6 proved the Qwen3-VL LANGUAGE tower (full real models 0.6B->32B) and rung 6.5
proved the VISION tower (composed ViT block + patch-merger). Both were validated SEPARATELY.
This rung composes the one VLM-unique path neither tower alone exercises: the FUSION seam.

End-to-end mini-VLM forward, device-resident through ZLUDA:
  vision encoder (LayerNorm -> proj GEMM -> GELU) producing N_IMG image tokens of dim H
    -> SCATTER those tokens into the text embedding stream at the <image> placeholder rows
    -> 2 stacked Qwen3 decoder layers (RMSNorm/QK-norm/RoPE/causal-GQA/SwiGLU) over the FUSED seq
    -> final RMSNorm -> LM-head GEMM -> logits -> last-token argmax.

The new path vs prior harnesses is the cross-modal SCATTER (image embeddings written into
specific text-sequence rows) and the decoder running causal attention over a MIXED image+text
sequence. cuBLAS->rocBLAS GEMMs chained with nvrtc->PTX->ZLUDA kernels; graded vs an independent
pure-Python fp64 oracle implementing the same spec at tight fp32 tolerance.

Run: LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 zluda_vlm.py [seed]
"""
import ctypes as C, os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_diff import Cuda, compile_ptx, CUDA, CUBLAS, _ck

# ---- dims (small, model-realistic ratios) ----
HV = 48                      # raw vision feature dim
H  = 64                      # LM hidden dim (== vision projector output, so tokens splice in)
NQ, NKV, HD = 8, 2, 8        # GQA 4:1 ; NQ*HD == H
I  = 128                     # SwiGLU intermediate
N_IMG = 4                    # image tokens produced by the vision encoder
S_TXT = 8                    # total fused sequence length (image tokens occupy IMG_POS within it)
IMG_POS = [2, 3, 4, 5]       # rows of the fused sequence replaced by image embeddings
V  = 32                      # LM-head vocab
NL = 2                       # stacked decoder layers
THETA, EPS = 10000.0, 1e-6
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 20260619)
assert len(IMG_POS) == N_IMG and NQ*HD == H

def rnd(n): return [random.uniform(-0.5, 0.5) for _ in range(n)]
def mat(r, c): return [rnd(c) for _ in range(r)]

# ---- vision encoder + projector weights ----
VIN = mat(N_IMG, HV)                 # raw vision features (N_IMG tokens)
vln_g = rnd(HV); vln_b = rnd(HV)     # vision LayerNorm
Wvp = mat(HV, H)                     # vision projector HV -> H (output matches LM hidden)
# ---- text embeddings (the non-image rows are real tokens; image rows get overwritten) ----
TXT = mat(S_TXT, H)
# ---- per-layer decoder weights ----
LAYERS = []
for _ in range(NL):
    LAYERS.append(dict(
        Wq=mat(H, NQ*HD), Wk=mat(H, NKV*HD), Wv=mat(H, NKV*HD), Wo=mat(NQ*HD, H),
        Wg=mat(H, I), Wu=mat(H, I), Wd=mat(I, H),
        gN1=rnd(H), gN2=rnd(H), gQ=rnd(HD), gK=rnd(HD)))
gNf = rnd(H)                          # final RMSNorm
Whead = mat(H, V)                     # LM head

# ================= pure-python fp64 reference (oracle) =================
def mm(A, B):
    m, k, n = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][t]*B[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
def layernorm(x, g, b):
    out = []
    for row in x:
        n = len(row); mean = sum(row)/n; var = sum((v-mean)**2 for v in row)/n
        inv = 1.0/math.sqrt(var+EPS)
        out.append([(row[j]-mean)*inv*g[j] + b[j] for j in range(n)])
    return out
def rmsnorm(x, g):
    out = []
    for row in x:
        ss = sum(v*v for v in row)/len(row); inv = 1.0/math.sqrt(ss+EPS)
        out.append([row[j]*inv*g[j] for j in range(len(row))])
    return out
def gelu_tanh(x):
    c = math.sqrt(2.0/math.pi)
    return [[0.5*v*(1.0+math.tanh(c*(v+0.044715*v**3))) for v in row] for row in x]
def add(a, b): return [[a[i][j]+b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
def qknorm(m_, nh, g):
    out = [row[:] for row in m_]
    for h in range(nh):
        base = h*HD
        for p in range(len(m_)):
            ss = sum(m_[p][base+t]**2 for t in range(HD))/HD; inv = 1.0/math.sqrt(ss+EPS)
            for t in range(HD): out[p][base+t] = m_[p][base+t]*inv*g[t]
    return out
def rope_heads(m_, nh):
    out = [row[:] for row in m_]; half = HD//2
    for h in range(nh):
        base = h*HD
        for p in range(len(m_)):
            for i in range(half):
                ang = p*(THETA**(-2.0*i/HD)); c = math.cos(ang); s = math.sin(ang)
                x1 = m_[p][base+i]; x2 = m_[p][base+half+i]
                out[p][base+i] = x1*c - x2*s; out[p][base+half+i] = x1*s + x2*c
    return out
def gqa_attn(Q, K, V_):
    scale = 1.0/math.sqrt(HD); S = len(Q); out = [[0.0]*(NQ*HD) for _ in range(S)]
    for h in range(NQ):
        kvh = h//(NQ//NKV); qb = h*HD; kb = kvh*HD
        for i in range(S):
            sc = []
            for j in range(S):
                sc.append(sum(Q[i][qb+t]*K[j][kb+t] for t in range(HD))*scale if j <= i else -1e30)
            m = max(sc); e = [math.exp(v-m) if v > -1e29 else 0.0 for v in sc]; ssum = sum(e)
            for t in range(HD):
                out[i][qb+t] = sum((e[j]/ssum)*V_[j][kb+t] for j in range(S))
    return out
def swiglu(g, u):
    return [[(g[i][j]/(1.0+math.exp(-g[i][j])))*u[i][j] for j in range(len(g[0]))] for i in range(len(g))]

def decoder_layer(X, W):
    xn = rmsnorm(X, W['gN1'])
    Q = mm(xn, W['Wq']); K = mm(xn, W['Wk']); Vv = mm(xn, W['Wv'])
    Q = qknorm(Q, NQ, W['gQ']); K = qknorm(K, NKV, W['gK'])
    Q = rope_heads(Q, NQ); K = rope_heads(K, NKV)
    ao = gqa_attn(Q, K, Vv); o = mm(ao, W['Wo']); x1 = add(X, o)
    xn2 = rmsnorm(x1, W['gN2'])
    g = mm(xn2, W['Wg']); u = mm(xn2, W['Wu']); sw = swiglu(g, u); d = mm(sw, W['Wd'])
    return add(x1, d)

def reference():
    # vision encoder + projector -> N_IMG image tokens of dim H
    vn = layernorm(VIN, vln_g, vln_b)
    img = gelu_tanh(mm(vn, Wvp))                 # (N_IMG, H)
    # fuse: scatter image tokens into the text embedding stream
    fused = [row[:] for row in TXT]
    for k, pos in enumerate(IMG_POS):
        fused[pos] = img[k][:]
    # decoder stack over the fused sequence
    h = fused
    for W in LAYERS:
        h = decoder_layer(h, W)
    h = rmsnorm(h, gNf)
    return mm(h, Whead)                          # (S_TXT, V) logits

# ================= ZLUDA path =================
KSRC = r'''
extern "C" __global__ void layernorm(const float* x,const float* g,const float* b,float* o,int rows,int cols,float eps){
  int r=blockIdx.x; if(r>=rows) return; float mean=0; for(int j=0;j<cols;j++) mean+=x[r*cols+j]; mean/=cols;
  float var=0; for(int j=0;j<cols;j++){float d=x[r*cols+j]-mean; var+=d*d;} var/=cols; float inv=rsqrtf(var+eps);
  for(int j=0;j<cols;j++) o[r*cols+j]=(x[r*cols+j]-mean)*inv*g[j]+b[j]; }
extern "C" __global__ void gelu(const float* x,float* o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){float v=x[i];
    float c=0.7978845608028654f; o[i]=0.5f*v*(1.f+tanhf(c*(v+0.044715f*v*v*v)));} }
extern "C" __global__ void rmsnorm(const float* x,const float* g,float* o,int rows,int cols,float eps){
  int r=blockIdx.x; if(r>=rows) return; float ss=0; for(int j=0;j<cols;j++){float v=x[r*cols+j]; ss+=v*v;}
  float inv=rsqrtf(ss/cols+eps); for(int j=0;j<cols;j++) o[r*cols+j]=x[r*cols+j]*inv*g[j]; }
extern "C" __global__ void qknorm(const float* x,const float* g,float* o,int rows,int width,int nh,int hd,float eps){
  int r=blockIdx.x; if(r>=rows) return;
  for(int h=0;h<nh;h++){int b=h*hd; float ss=0; for(int t=0;t<hd;t++){float v=x[r*width+b+t]; ss+=v*v;}
    float inv=rsqrtf(ss/hd+eps); for(int t=0;t<hd;t++) o[r*width+b+t]=x[r*width+b+t]*inv*g[t]; } }
extern "C" __global__ void rope(const float* x,float* o,int rows,int width,int nh,int hd,float theta){
  int r=blockIdx.x; if(r>=rows) return; int half=hd/2; for(int k=0;k<width;k++) o[r*width+k]=x[r*width+k];
  for(int h=0;h<nh;h++){int b=h*hd; for(int i=0;i<half;i++){float ang=r*powf(theta,-2.f*i/hd);float c=cosf(ang),s=sinf(ang);
    float x1=x[r*width+b+i],x2=x[r*width+b+half+i]; o[r*width+b+i]=x1*c-x2*s; o[r*width+b+half+i]=x1*s+x2*c; } } }
extern "C" __global__ void gqa(const float* Q,const float* K,const float* V,float* O,int S,int nq,int nkv,int hd){
  float scale=rsqrtf((float)hd);
  for(int h=0;h<nq;h++){int kvh=h/(nq/nkv); int qb=h*hd; int kb=kvh*hd; int qc=nq*hd; int kc=nkv*hd;
    for(int i=0;i<S;i++){ float sc[64]; float m=-1e30f;
      for(int j=0;j<S;j++){ if(j<=i){float d=0;for(int t=0;t<hd;t++)d+=Q[i*qc+qb+t]*K[j*kc+kb+t]; sc[j]=d*scale; if(sc[j]>m)m=sc[j];} else sc[j]=-1e30f; }
      float ss=0; for(int j=0;j<S;j++){float e=(sc[j]<=-1e29f)?0.f:expf(sc[j]-m); sc[j]=e; ss+=e;}
      for(int t=0;t<hd;t++){float acc=0; for(int j=0;j<S;j++) acc+=(sc[j]/ss)*V[j*kc+kb+t]; O[i*qc+qb+t]=acc;} } } }
extern "C" __global__ void swiglu(const float* g,const float* u,float* o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){float x=g[i]; o[i]=(x/(1.f+expf(-x)))*u[i];} }
extern "C" __global__ void addk(const float* a,const float* b,float* o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]+b[i]; }
extern "C" __global__ void scatter_rows(const float* src,float* dst,const int* pos,int n_rows,int width){
  // write src row k into dst at row pos[k]; one block per src row
  int k=blockIdx.x; if(k>=n_rows) return; int d=pos[k];
  for(int t=0;t<width;t++) dst[d*width+t]=src[k*width+t]; }
'''

def flat(m): return [v for row in m for v in row]

class Z:
    def __init__(self):
        self.cu = Cuda(); self.m = self.cu.module(compile_ptx(KSRC))
        self.h = C.c_void_p(); _ck(CUBLAS.cublasCreate_v2(C.byref(self.h)), "cublasCreate")
        CUBLAS.cublasSgemm_v2.argtypes = [C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
            C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int]
    def fn(self, name): return self.cu.func(self.m, name)
    def dev(self, floats): return self.cu.to_dev(floats)
    def dev_int(self, ints):
        arr = (C.c_int * len(ints))(*ints); dptr = self.cu.alloc(len(ints))  # 4 bytes/int == 4 bytes/float
        _ck(CUDA.cuMemcpyHtoD_v2(dptr, arr, 4*len(ints)), "htod_int"); return dptr
    def alloc(self, n): return self.cu.alloc(n)
    def gemm(self, dA, dB, M, K, N):
        dC = self.alloc(M*N); a = C.c_float(1.0); b = C.c_float(0.0)
        pB = C.c_void_p(dB.value); pA = C.c_void_p(dA.value); pC = C.c_void_p(dC.value)
        _ck(CUBLAS.cublasSgemm_v2(self.h,0,0,N,M,K,C.byref(a),pB,N,pA,K,C.byref(b),pC,N), "sgemm")
        _ck(CUDA.cuCtxSynchronize(), "sync"); return dC
    def launch(self, name, grid, block, params): self.cu.launch(self.fn(name), grid, block, params)
    def get(self, dptr, n): return self.cu.from_dev(dptr, n)

def vp(p): return C.c_void_p(p.value)

def run_zluda():
    z = Z()
    S = S_TXT
    # vision encoder + projector
    dVIN = z.dev(flat(VIN)); dvg = z.dev(vln_g); dvb = z.dev(vln_b); dWvp = z.dev(flat(Wvp))
    dvn = z.alloc(N_IMG*HV); z.launch("layernorm",(N_IMG,1,1),(1,1,1),[vp(dVIN),vp(dvg),vp(dvb),vp(dvn),C.c_int(N_IMG),C.c_int(HV),C.c_float(EPS)])
    dvp = z.gemm(dvn, dWvp, N_IMG, HV, H)
    dimg = z.alloc(N_IMG*H); z.launch("gelu",((N_IMG*H+63)//64,1,1),(64,1,1),[vp(dvp),vp(dimg),C.c_int(N_IMG*H)])
    # fuse: copy text embeddings, then scatter image tokens into IMG_POS rows
    dh = z.dev(flat(TXT))
    dpos = z.dev_int(IMG_POS)
    z.launch("scatter_rows",(N_IMG,1,1),(1,1,1),[vp(dimg),vp(dh),vp(dpos),C.c_int(N_IMG),C.c_int(H)])
    _ck(CUDA.cuCtxSynchronize(), "sync")
    # decoder stack over the fused sequence
    for W in LAYERS:
        dWq=z.dev(flat(W['Wq'])); dWk=z.dev(flat(W['Wk'])); dWv=z.dev(flat(W['Wv'])); dWo=z.dev(flat(W['Wo']))
        dWg=z.dev(flat(W['Wg'])); dWu=z.dev(flat(W['Wu'])); dWd=z.dev(flat(W['Wd']))
        dgN1=z.dev(W['gN1']); dgN2=z.dev(W['gN2']); dgQ=z.dev(W['gQ']); dgK=z.dev(W['gK'])
        dxn=z.alloc(S*H); z.launch("rmsnorm",(S,1,1),(1,1,1),[vp(dh),vp(dgN1),vp(dxn),C.c_int(S),C.c_int(H),C.c_float(EPS)])
        dQ=z.gemm(dxn,dWq,S,H,NQ*HD); dK=z.gemm(dxn,dWk,S,H,NKV*HD); dV=z.gemm(dxn,dWv,S,H,NKV*HD)
        dQn=z.alloc(S*NQ*HD); z.launch("qknorm",(S,1,1),(1,1,1),[vp(dQ),vp(dgQ),vp(dQn),C.c_int(S),C.c_int(NQ*HD),C.c_int(NQ),C.c_int(HD),C.c_float(EPS)])
        dKn=z.alloc(S*NKV*HD); z.launch("qknorm",(S,1,1),(1,1,1),[vp(dK),vp(dgK),vp(dKn),C.c_int(S),C.c_int(NKV*HD),C.c_int(NKV),C.c_int(HD),C.c_float(EPS)])
        dQr=z.alloc(S*NQ*HD); z.launch("rope",(S,1,1),(1,1,1),[vp(dQn),vp(dQr),C.c_int(S),C.c_int(NQ*HD),C.c_int(NQ),C.c_int(HD),C.c_float(THETA)])
        dKr=z.alloc(S*NKV*HD); z.launch("rope",(S,1,1),(1,1,1),[vp(dKn),vp(dKr),C.c_int(S),C.c_int(NKV*HD),C.c_int(NKV),C.c_int(HD),C.c_float(THETA)])
        dao=z.alloc(S*NQ*HD); z.launch("gqa",(1,1,1),(1,1,1),[vp(dQr),vp(dKr),vp(dV),vp(dao),C.c_int(S),C.c_int(NQ),C.c_int(NKV),C.c_int(HD)])
        do=z.gemm(dao,dWo,S,NQ*HD,H)
        dx1=z.alloc(S*H); z.launch("addk",((S*H+63)//64,1,1),(64,1,1),[vp(dh),vp(do),vp(dx1),C.c_int(S*H)])
        dxn2=z.alloc(S*H); z.launch("rmsnorm",(S,1,1),(1,1,1),[vp(dx1),vp(dgN2),vp(dxn2),C.c_int(S),C.c_int(H),C.c_float(EPS)])
        dg=z.gemm(dxn2,dWg,S,H,I); du=z.gemm(dxn2,dWu,S,H,I)
        dsw=z.alloc(S*I); z.launch("swiglu",((S*I+63)//64,1,1),(64,1,1),[vp(dg),vp(du),vp(dsw),C.c_int(S*I)])
        dd=z.gemm(dsw,dWd,S,I,H)
        dout=z.alloc(S*H); z.launch("addk",((S*H+63)//64,1,1),(64,1,1),[vp(dx1),vp(dd),vp(dout),C.c_int(S*H)])
        dh = dout
    # final RMSNorm + LM head
    dgNf=z.dev(gNf); dWhead=z.dev(flat(Whead))
    dhn=z.alloc(S*H); z.launch("rmsnorm",(S,1,1),(1,1,1),[vp(dh),vp(dgNf),vp(dhn),C.c_int(S),C.c_int(H),C.c_float(EPS)])
    dlog=z.gemm(dhn,dWhead,S,H,V)
    fo=z.get(dlog, S*V)
    return [fo[i*V:(i+1)*V] for i in range(S)]

def main():
    ref = reference(); got = run_zluda()
    S = S_TXT; mxa = mxr = 0.0; nf = 0; rtol, atol = 1e-4, 1e-5; tot = S*V
    for i in range(S):
        for j in range(V):
            a = abs(got[i][j]-ref[i][j]); r = a/(abs(ref[i][j])+1e-12)
            mxa = max(mxa, a); mxr = max(mxr, r)
            if a > atol + rtol*abs(ref[i][j]): nf += 1
    # last-token argmax (the decode-relevant quantity) must match
    am_ref = max(range(V), key=lambda j: ref[S-1][j])
    am_got = max(range(V), key=lambda j: got[S-1][j])
    print(f"Qwen3-VL fusion seam (vision->scatter->{NL}x decoder->logits) through ZLUDA: "
          f"max_abs={mxa:.3e} max_rel={mxr:.3e} n_fail={nf}/{tot} argmax_ref={am_ref} argmax_got={am_got}")
    ok = (nf == 0 and am_ref == am_got)
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
