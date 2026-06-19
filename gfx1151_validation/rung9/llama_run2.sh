#!/bin/bash
set -x
echo "LLAMA_RUN2_START $(date -u)"
ZL=/root/ZLUDA/target/release
export LD_LIBRARY_PATH="$ZL:/usr/local/cuda/lib64:/opt/rocm/lib"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
cd /root/llama.cpp
timeout 600 ./build/bin/llama-cli -m /root/tiny.gguf -ngl 99 \
  -p "The capital of France is" -n 24 --temp 0 -fa off --seed 1 < /dev/null 2>&1
echo "LLAMA_RUN2_RC=$?"
echo "LLAMA_RUN2_DONE $(date -u)"
