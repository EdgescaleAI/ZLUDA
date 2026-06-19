# ZLUDA on Strix Halo gfx1151 — validation results (2026-06-18)

ZLUDA built and validated on an AMD Ryzen AI MAX+ 395 (Radeon 8060S, **gfx1151**, 40 CU,
ROCm 7.2.4) running Single-Node OpenShift. Userspace only — no kernel-driver changes.

## Build
`cargo xtask --release` builds clean against ROCm 7.2.4 / LLVM (clang 22). Full lib set:
`libnvcuda.so`(→libcuda), `libcublas`, `libcublasLt`, `libcudnn`(8/9), `libcufft`, `libcusparse`,
`libnvidia-ml`. Environment prep: `git lfs pull` (device-libs bitcode), apt-install ROCm math
dev libs (`rocblas-dev hipblaslt-dev miopen-hip-dev rocsparse-dev rocfft-dev`).

## ZLUDA's own conformance (ptx spirv_run) — 360/360 executable PASS
- `cargo test -p ptx --release -- _amdgpu` → **184/184 PASS** (PTX→LLVM→AMD compiled & run on gfx1151 via HIP).
- `cargo test -p ptx --release -- _cuda` → **176/176 PASS** (full ZLUDA libcuda driver path:
  cuModuleLoadData → cuLaunchKernel → cuMemcpy; symlink `libnvcuda.so`→`/usr/lib/x86_64-linux-gnu/libcuda.so.1`).
- `_llvm` (183 IR-text exact-match) fail under `--release` ONLY due to the debug-only `+precise-memory`
  target-feature (`emit.rs` gates it on `cfg!(debug_assertions)`; refs captured in debug). IR bodies are
  byte-identical and the same codegen passes all 360 execution tests → build-mode artifact, not a defect.

## T0 op differential (CUDA-Bridge oracle) — 20/20 PASS
`python3 zluda_diff.py` — NVIDIA nvrtc compiles each op's CUDA-C → PTX; ZLUDA runs it on gfx1151;
output diffed vs torch-CPU-fp32 fixtures at tight fp32 tolerance (rtol 1e-5 / atol 1e-6):

| op | max_rel | op | max_rel |
|---|---|---|---|
| gemm_8x16x8 (cuBLAS→rocBLAS) | 1.6e-6 | attn_causal | 3.5e-6 |
| rmsnorm | 1.7e-7 | gqa_attn | 2.7e-6 |
| softmax | 2.2e-7 | vis_attn_varlen | 8.5e-7 |
| swiglu | 1.2e-7 | paged_attn | 1.1e-7 |
| gelu | 2.7e-7 | layernorm | 3.0e-6 |
| rope | 3.4e-7 | rope2d | 4.8e-7 |
| residual_add | 0 (exact) | repeat_kv | 0 |
| embedding_gather | 0 | spatial_merge | 0 |
| token_scatter | 0 | kv_cache | 0 |
| sampling_topk | 1.1e-7 | posemb_interp | 6.4e-8 |

cuBLAS GEMM also verified standalone at 3 shapes (8×16×8, 32×64×48, 128×256×96), all within fp32 bound.

## Known wall (whole-model rungs via PyTorch)
Stock `torch 2.6 cu124` wheel kernels are **SASS-only (no PTX)** — ZLUDA translates PTX, so torch's
elementwise/attention kernels can't run ("named symbol not found"). PyTorch DOES load and detect the GPU
as `[ZLUDA]`, and GEMM routes through ZLUDA cuBLAS→rocBLAS. Running a whole model needs a PTX-bearing
PyTorch build (or running the forward pass through the proven ZLUDA op primitives here). See the parent
project's `ZLUDA-MORNING-SUMMARY.md`.

## Repro
```
export LD_LIBRARY_PATH=<ZLUDA>/target/release:<nvrtc lib dir>:/opt/rocm/lib
export HSA_OVERRIDE_GFX_VERSION=11.5.1
python3 zluda_diff.py            # 20/20
cargo test -p ptx --release -- _amdgpu _cuda   # 360/360
```
Fixtures + harness vendored from the CUDA-Bridge project (same owner) as the numeric oracle.

## Composed decoder layer (rung 2.5) — PASS
`zluda_layer.py` runs a full **Qwen3-style decoder layer** device-resident through ZLUDA
(RMSNorm → QKV proj cuBLAS → per-head QK-norm → RoPE → causal GQA attention → O-proj cuBLAS
→ residual → RMSNorm → SwiGLU MLP 3×cuBLAS → residual), graded vs an independent pure-Python
fp64 oracle. 3 seeds, all PASS, n_fail=0/384, max_abs ~1–2e-6 (rtol 1e-4/atol 1e-5 — appropriate
for a ~12-op fp32 chain). Proves the op set composes, not just individual ops.

## Deep stack (rung 2.75) — PASS
`zluda_stack.py` stacks N decoder layers (residual stream device-resident) + final RMSNorm +
LM-head GEMM → logits. 6/12/**28** layers (28 = Qwen3-0.6B depth) all PASS: n_fail=0/288,
max_abs ~7e-7 (stable across depth), and the **last-token argmax matches the fp64 oracle at every
depth** — a greedy decode would emit the same token. The whole-transformer forward *structure*
composes correctly through ZLUDA; real-weight Qwen3-0.6B grading is the remaining rung-3 step.

## ★ Rung 3 — real Qwen3-0.6B forward through ZLUDA — PASS
`zluda_qwen.py`: transformers loads Qwen3-0.6B weights on CPU (no GPU kernels); the forward runs
**entirely through ZLUDA primitives** (cuBLAS→rocBLAS GEMMs + nvrtc→PTX→ZLUDA kernels for
RMSNorm / per-head QK-norm / RoPE / causal GQA attention / SwiGLU). No torch CUDA kernels (the wheel
is SASS-only). Graded vs the torch-cpu-fp32 fixture:
- top-10 next-token ids **exactly match**; top-10 logits match to 4 decimals (max diff 0.000)
- 8-token greedy continuation **identical**: " Paris. The capital of Italy is Rome"

A whole 28-layer transformer (GQA 16q/8kv, head_dim 128, tied 151936-vocab) runs correctly through
ZLUDA on gfx1151. Matching 10 logits + 8 consecutive argmaxes is conclusive.

## ★ Scale: Qwen3-1.7B and Qwen3-8B forwards through ZLUDA — PASS
`zluda_scale.py` runs larger Qwen3 models through the same ZLUDA forward, self-graded vs a live
torch-CPU-fp32 reference:
- **Qwen3-1.7B** (~2.03B params, ~8 GB fp32): top-10 ids exact, logits to 3 decimals.
- **Qwen3-8B** (~8.19B params, **32.8 GB fp32**, untied LM head): top-10 ids **exact**, max_logit_diff **0.000**.

An 8B model — weights larger than typical discrete-GPU VRAM — runs correctly through ZLUDA on the
gfx1151 APU's ~133 GB unified memory (Cosmos-Reason2-8B size class). Requires the untied-lm_head +
cuMemFree-argtypes + >2 GB-HtoD-chunking fixes in this commit, and a pod memory limit ≥ ~70 GB for the
8B fp32 host footprint (a container cgroup limit, not a ZLUDA/GPU limit — ZLUDA exposes the full pool).

## ★★ FRONTIER — Rung 6: Qwen3-32B forward through ZLUDA (32B size class) — PASS
`zluda_frontier.py Qwen/Qwen3-32B fp16` — the brief's named **32B size-class frontier**, demonstrated on
the OPEN Qwen3-32B (no HF token; the same size-class-stand-in logic that used Qwen3-8B for the gated
Cosmos-Reason2-8B). The full **64-layer, 32.0B-param** forward runs entirely through ZLUDA primitives
(cuBLAS→rocBLAS GEMMs + nvrtc→PTX→ZLUDA kernels) on gfx1151; graded eval-style (per TEST-STRATEGY, the
BF16/FP16 tier grades by behavioral top-k agreement, NOT bit-diff) vs an independent torch-CPU **fp16**
reference:
- ZLUDA top-10 next-token ids **exactly match** the reference, **in identical order** (`top10_setmatch=True`)
- `argmax_match=True`, `top5_match=True`, `max_logit_diff=0.047` (expected fp32-GEMM-on-fp16-weights vs
  fp16-CPU compute; ids unchanged)
- config: H 5120, L 64, NQ 64, NKV 8, HD 128, I 25600, V 151936

Memory: fp16 weights ~**64 GB** (fp32 would be 128 GB > the node's ~120 GB RAM — physically impossible,
which is *why* the frontier is graded in 2-byte precision). fp16 ≡ BF16 footprint (2 bytes/param), so this
is exactly the brief's "32B in BF16 (~64 GB) fits the ~96 GB unified pool" thesis. The harness is
memory-frugal: it streams weights to fp16 numpy popping each tensor out of the state_dict (host RAM stays
~one model size, not two), and `zluda_qwen.forward` upcasts each weight to fp32 only at upload and frees
every layer's device buffers before the next (device footprint is per-layer, not whole-model).

A **32-billion-parameter** model — far beyond typical discrete-GPU VRAM — runs correctly through ZLUDA on
a single gfx1151 APU's unified memory. This is the highest reachable dense-model rung tonight: 32B is the
largest *dense* Qwen3 (235B is MoE at ~470 GB, exceeds node RAM); Cosmos-Reason2 *itself* additionally
needs the gated HF token + an eval oracle + the vision tower (blocked without the user).

Repro: `HSA_OVERRIDE_GFX_VERSION=11.5.1 LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib python3 zluda_frontier.py Qwen/Qwen3-32B fp16`
(pod memory limit ≥ ~100 GB for the 32B fp16 host footprint).

## ★ Rung 6.5 — composed Qwen3-VL VISION-tower (ViT) block + patch-merger through ZLUDA — PASS
The whole-night text ladder (rungs 0→6) climbed the **language** half of the coverage target
(Cosmos-Reason2-8B = Qwen3-**VL**, a vision-language model). `zluda_vit.py` closes the **vision** half —
the analog of rung 2.5 (composed decoder layer) but for the ViT, exercising a genuinely different op mix
from the text decoder: **LayerNorm** (mean+var+bias, not RMSNorm), **GELU** tanh-approx (not SwiGLU),
**FULL bidirectional MHA** (not causal GQA), **2D-RoPE** (row/col positional, not 1D), and a **spatial
2×2 patch merge**. The composed forward runs device-resident through ZLUDA — cuBLAS→rocBLAS GEMMs chained
with nvrtc→PTX→ZLUDA elementwise/attention kernels — and is graded vs an independent pure-Python **fp64**
oracle implementing the same spec, so the diff measures whether the op set *composes* through ZLUDA.

Block: LayerNorm → QKV proj (cuBLAS) → 2D-RoPE → full attention → out-proj → residual → LayerNorm →
MLP fc1·GELU·fc2 → residual. Merger: spatial 2×2 merge → LayerNorm → fc1·GELU·fc2.

**6 seeds (default, 1, 7, 42, 2026, 99999) all PASS**, `n_fail=0/256` every seed, `max_abs` 3.5e-6–5.8e-6
(rtol 1e-4 / atol 1e-5 — appropriate for this fp32 op chain). Both the vision tower and the language tower
of the target VLM now compose correctly through ZLUDA on gfx1151.

Repro: `HSA_OVERRIDE_GFX_VERSION=11.5.1 LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib python3 zluda_vit.py [seed]`

## ★ Rung 6.75 — composed cross-modal VLM FUSION seam through ZLUDA — PASS
Rungs 0→6 proved the language tower (real models 0.6B→32B) and 6.5 the vision tower — but SEPARATELY.
`zluda_vlm.py` composes the one VLM-unique path neither tower alone exercises: an end-to-end mini-VLM
forward where image embeddings are fused into the text token stream. Pipeline, device-resident through
ZLUDA: vision encoder (LayerNorm → projector GEMM → GELU) → **SCATTER** the N image tokens into the
text-embedding stream at the `<image>` placeholder rows → **2 stacked Qwen3 decoder layers** (RMSNorm /
QK-norm / RoPE / causal-GQA / SwiGLU) running causal attention over the **mixed image+text sequence** →
final RMSNorm → LM-head GEMM → logits. The genuinely new path vs prior harnesses is the cross-modal
scatter/splice and the decoder operating over a fused sequence; graded vs an independent pure-Python fp64
oracle.

**6 seeds (default, 1, 7, 42, 2026, 99999) all PASS**, `n_fail=0/256` every seed, `max_abs` 7.5e-7–1.4e-6,
and the **last-token argmax matches the oracle every seed** (a VLM greedy decode would emit the same token).
The full VLM forward *structure* — both towers plus the fusion seam — composes correctly through ZLUDA.

Repro: `HSA_OVERRIDE_GFX_VERSION=11.5.1 LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib python3 zluda_vlm.py [seed]`

## ★★ Rung 7 — REAL Qwen3-VL-2B-Instruct text tower through ZLUDA (literal target arch) — PASS
All prior model rungs used plain Qwen3 / synthetic harnesses as stand-ins for the gated Cosmos-Reason2.
`Qwen/Qwen3-VL-2B-Instruct` is **open and ungated** and is the **literal target architecture family**
(Cosmos-Reason2 = Qwen3-VL), so `zluda_qwenvl.py` runs the **real target-model weights** through ZLUDA.
Scope: the language (text) tower — a **28-layer Qwen3 decoder** (hidden 2048, 16q/8kv GQA, head_dim 128,
per-head QK-norm, SwiGLU, tied 151936-vocab), config `mrope_interleaved`, rope_theta 5e6.

Key fact that makes this run on the proven path: Qwen3-VL uses **M-RoPE**, but for **text-only** input all
three M-RoPE position axes share the same token index, so M-RoPE reduces **exactly** to standard 1D RoPE
with the model's own `inv_freq` — so the rung-3 `rope_f` kernel (model-supplied inv_freq) applies unchanged.
The forward runs entirely through ZLUDA primitives (cuBLAS→rocBLAS GEMMs + nvrtc→PTX→ZLUDA kernels for
RMSNorm / per-head QK-norm / RoPE / causal GQA / SwiGLU); no torch CUDA kernels. Oracle = **HF's own
torch-CPU fp32 forward of the same model** (which uses HF's correct M-RoPE) on the same input ids.

**4 distinct prompts (lengths 4–8 tokens, incl. code) all PASS**: ZLUDA top-10 next-token ids **exactly
match** HF in identical order, `top5_match=True`, `top10_setmatch=True`, **`max_logit_diff=0.000`** on every
prompt. Matching 10 logits to 4 dp across a 152k vocab on a real 28-layer 2B model, on 4 inputs, is
conclusive — the real Qwen3-VL text tower runs correctly through ZLUDA on gfx1151.

Not yet done (honest top): the real Qwen3-VL **vision** tower end-to-end with real image input requires
matching HF's exact Qwen3-VL ViT internals (windowed attention, deepstack merger) — deliberately out of
scope vs the spec-composition harnesses (rung 6.5 proved the vision op set composes; matching HF vision
internals exactly is a separate reverse-engineering effort). Cosmos-Reason2 *itself* additionally needs the
gated HF token + an eval oracle.

Repro: `HSA_OVERRIDE_GFX_VERSION=11.5.1 LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib python3 zluda_qwenvl.py`
