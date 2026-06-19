#!/bin/bash
set -x
echo "LLAMA_RELINK2_START $(date -u)"
mkdir -p /root/linkstubs
ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /root/linkstubs/libcuda.so.1
ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /root/linkstubs/libcuda.so
ls -la /root/linkstubs
cd /root/llama.cpp
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
# Link against REAL cuda libs (versioned cublas + driver stub for libcuda.so.1).
# rpath-link is link-time-only (not recorded) so runtime LD_LIBRARY_PATH (ZLUDA) wins.
LF="-L/usr/local/cuda/lib64 -L/root/linkstubs -Wl,-rpath-link,/usr/local/cuda/lib64:/root/linkstubs"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70-virtual \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="$LF" -DCMAKE_SHARED_LINKER_FLAGS="$LF" 2>&1 | tail -3
echo "RECONF_RC=${PIPESTATUS[0]}"
cmake --build build --config Release -j6 --target llama-cli 2>&1 | tail -18
echo "LLAMA_BUILD_RC=${PIPESTATUS[0]}"
ls -la build/bin/llama-cli 2>&1
echo "LLAMA_RELINK2_DONE $(date -u)"
