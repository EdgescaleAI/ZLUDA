#!/bin/bash
# Focused SSM_CONV diagnostic through ZLUDA with PTX debug
echo "SSM_RUN_START $(date -u)"
ZL=/root/ZLUDA/target/release
export LD_LIBRARY_PATH="$ZL:/usr/local/cuda-12.4/lib64:/opt/rocm/lib"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
TBO=/root/llama.cpp/build/bin/test-backend-ops
strip() { sed -r 's/\x1b\[[0-9;]*m//g'; }
echo "=== SSM_CONV with ZLUDA_PTX_DEBUG=1 (translate errors, if any) ==="
ZLUDA_PTX_DEBUG=1 timeout 300 "$TBO" test -o SSM_CONV -b CUDA0 < /dev/null 2>&1 | strip | grep -iE "ZLUDA|PTX|Unsupported|Unrecognized|Translate|error|SSM_CONV.*(OK|FAIL|ERR)" | head -60
echo "=== SSM_CONV plain verdicts ==="
timeout 300 "$TBO" test -o SSM_CONV -b CUDA0 < /dev/null 2>&1 | strip | grep -E "SSM_CONV|OK|FAIL|backend" | tail -30
echo "SSM_RUN_DONE $(date -u)"
