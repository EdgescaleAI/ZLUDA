#!/bin/bash
set -x
echo "TRSM_RUN_START $(date -u)"
export PATH=/usr/local/cuda-12.4/bin:$PATH
ZL=/root/ZLUDA/target/release
CUDALIB=/usr/local/cuda-12.4/lib64
cd /root/r11
# compile against real cuda headers/libs; ZLUDA wins at runtime via LD_LIBRARY_PATH
nvcc -o trsm_test trsm_test.cu -I/usr/local/cuda-12.4/include -lcublas -lcudart -L"$CUDALIB" 2>&1 | tail -25
echo "NVCC_RC=${PIPESTATUS[0]}"
# ensure ZLUDA provides the cublas/cuda sonames the binary needs
cd "$ZL"
[ -e libnvcuda.so ] && { for s in libcuda.so libcuda.so.1; do ln -sf libnvcuda.so $s; done; }
ls libcublas.so 2>/dev/null && for v in 11 12 13; do [ -e libcublas.so.$v ] || ln -sf libcublas.so libcublas.so.$v; done
ls -la libcublas* libcuda* 2>&1 | head
cd /root/r11
export LD_LIBRARY_PATH="$ZL:$CUDALIB:/opt/rocm/lib"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
./trsm_test
echo "TRSM_RUN_RC=$?"
echo "TRSM_RUN_DONE $(date -u)"
