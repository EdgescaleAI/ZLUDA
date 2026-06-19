#!/bin/bash
echo "MODISO_BUILD $(date -u)"
export PATH=/usr/local/cuda-12.4/bin:$PATH
ZL=/root/ZLUDA/target/release
# compile PTX-bearing (compute_70) so ZLUDA translates it; link against cudart, driver stub
STUB=$(ls /usr/local/cuda*/lib64/stubs/libcuda.so 2>/dev/null | head -1)
mkdir -p /root/linkstubs; ln -sf "$STUB" /root/linkstubs/libcuda.so.1; ln -sf "$STUB" /root/linkstubs/libcuda.so
nvcc -gencode arch=compute_70,code=compute_70 /root/mod_iso.cu -o /root/mod_iso \
  -L/root/linkstubs -lcuda -Wno-deprecated-gpu-targets 2>&1 | tail -8
echo "MODISO_NVCC_RC=${PIPESTATUS[0]}"
export LD_LIBRARY_PATH="$ZL:/usr/local/cuda-12.4/lib64:/opt/rocm/lib"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
echo "=== run through ZLUDA ==="
ZLUDA_PTX_DEBUG=1 /root/mod_iso 2>&1 | grep -vE "^$" | tail -30
echo "MODISO_DONE $(date -u)"
