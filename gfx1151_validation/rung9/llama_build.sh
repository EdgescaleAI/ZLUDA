#!/bin/bash
set -x
echo "LLAMA_BUILD_START $(date -u)"
cd /root
which cmake || apt-get install -y cmake
which git || apt-get install -y git
if [ ! -d /root/llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp 2>&1 | tail -5
fi
cd /root/llama.cpp
echo "LLAMA_COMMIT $(git rev-parse HEAD)"
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
# 70-virtual => fatbin carries PTX only (compute_70), which ZLUDA translates (not SASS).
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70-virtual \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF 2>&1 | tail -25
echo "LLAMA_CONFIGURE_RC=${PIPESTATUS[0]}"
cmake --build build --config Release -j6 --target llama-cli 2>&1 | tail -40
echo "LLAMA_BUILD_RC=${PIPESTATUS[0]}"
ls -la build/bin/llama-cli 2>&1
echo "LLAMA_BUILD_DONE $(date -u)"
