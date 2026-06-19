#!/bin/bash
set -x
echo "TBO_RUN_START $(date -u)"
ZL=/root/ZLUDA/target/release
CUDALIB=$(ls -d /usr/local/cuda*/lib64 2>/dev/null | head -1)
export LD_LIBRARY_PATH="$ZL:$CUDALIB:/opt/rocm/lib"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
cd /root/llama.cpp
TBO=$(find /root/llama.cpp/build -name 'test-backend-ops' -type f | head -1)
echo "TBO_BIN=$TBO"
# list backends ZLUDA exposes
timeout 120 "$TBO" --help < /dev/null 2>&1 | head -20
echo "--- RUN test (CUDA backend = ZLUDA) ---"
# 'test' mode runs correctness for all ops; filter output for the CUDA backend
timeout 3600 "$TBO" test < /dev/null 2>&1
echo "TBO_RUN_RC=$?"
echo "TBO_RUN_DONE $(date -u)"
