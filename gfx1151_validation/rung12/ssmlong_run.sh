#!/bin/bash
echo "SSMLONG_BUILD $(date -u)"
export PATH=/usr/local/cuda-12.4/bin:$PATH
ZL=/root/ZLUDA/target/release
STUB=$(ls /usr/local/cuda*/lib64/stubs/libcuda.so 2>/dev/null | head -1)
mkdir -p /root/linkstubs; ln -sf "$STUB" /root/linkstubs/libcuda.so.1; ln -sf "$STUB" /root/linkstubs/libcuda.so
nvcc -gencode arch=compute_70,code=compute_70 /root/ssmlong.cu -o /root/ssmlong \
  -L/root/linkstubs -lcuda -Wno-deprecated-gpu-targets 2>&1 | tail -8
echo "SSMLONG_NVCC_RC=${PIPESTATUS[0]}"
export LD_LIBRARY_PATH="$ZL:/usr/local/cuda-12.4/lib64:/opt/rocm/lib"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
echo "=== run through ZLUDA ==="
ZLUDA_PTX_DEBUG=1 /root/ssmlong 2>&1 | grep -vE "^$" | tail -40
echo "SSMLONG_DONE $(date -u)"
