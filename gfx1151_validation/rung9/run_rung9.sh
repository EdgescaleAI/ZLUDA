#!/bin/bash
# Rung 9a runner — compile real CUDA binaries with nvcc (PTX-bearing) and run them through ZLUDA on gfx1151.
# Run inside the pod after ZLUDA is built and the CUDA toolkit (nvcc + cudart + cublas headers) is installed.
set -x
echo "RUNG9_START $(date -u)"

ZL=/root/ZLUDA/target/release
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
echo "nvcc = $NVCC"; "$NVCC" --version

# ZLUDA provides the driver shim as libnvcuda.so; the loader looks for libcuda.so.1.
ln -sf "$ZL/libnvcuda.so" "$ZL/libcuda.so.1"
ln -sf "$ZL/libnvcuda.so" "$ZL/libcuda.so"
ls -la "$ZL"/libcuda* "$ZL"/libcublas* 2>&1

cd /root/r9
# PTX-bearing fatbin: code=compute_XX embeds PTX (ZLUDA translates PTX, NOT SASS). compute_70 is a safe baseline.
GENCODE="-gencode arch=compute_70,code=compute_70"

echo "=== compile vecadd ==="
"$NVCC" $GENCODE vecadd.cu -o vecadd 2>&1 || { echo "NVCC_VECADD_FAIL"; }
echo "=== compile gemm_rt ==="
"$NVCC" $GENCODE gemm_rt.cu -lcublas -o gemm_rt 2>&1 || { echo "NVCC_GEMM_FAIL"; }

# Run with ZLUDA's driver/cublas ahead of everything; cudart/cublas runtime libs come from the CUDA toolkit.
export LD_LIBRARY_PATH="$ZL:/opt/rocm/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION=11.5.1

echo "=== RUN vecadd (custom kernel via PTX frontend) ==="
./vecadd 2>&1
echo "VECADD_RC=$?"

echo "=== RUN gemm_rt (cudart -> cublas -> ZLUDA -> rocBLAS) ==="
./gemm_rt 2>&1
echo "GEMM_RC=$?"
echo "RUNG9_DONE $(date -u)"
