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
