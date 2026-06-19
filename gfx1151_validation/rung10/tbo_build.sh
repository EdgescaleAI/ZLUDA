#!/bin/bash
set -x
echo "TBO_BUILD_START $(date -u)"
cd /root
export PATH=/usr/local/cuda/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDACXX=$(which nvcc)
echo "NVCC=$CUDACXX"; nvcc --version | tail -2
if [ ! -d /root/llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp 2>&1 | tail -3
fi
cd /root/llama.cpp
echo "LLAMA_COMMIT $(git rev-parse HEAD)"
# driver stub so versioned libcuda.so.1 resolves at link; ZLUDA wins at runtime via LD_LIBRARY_PATH
mkdir -p /root/linkstubs
STUB=$(ls /usr/local/cuda*/lib64/stubs/libcuda.so 2>/dev/null | head -1)
ln -sf "$STUB" /root/linkstubs/libcuda.so.1
ln -sf "$STUB" /root/linkstubs/libcuda.so
CUDALIB=$(dirname "$STUB")/..
LF="-L${CUDALIB} -L/root/linkstubs -Wl,-rpath-link,${CUDALIB}:/root/linkstubs"
# 70-virtual => PTX-only fatbins (ZLUDA translates PTX, not SASS)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70-virtual \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_EXE_LINKER_FLAGS="$LF" -DCMAKE_SHARED_LINKER_FLAGS="$LF" 2>&1 | tail -6
echo "TBO_CONFIGURE_RC=${PIPESTATUS[0]}"
cmake --build build --config Release -j6 --target test-backend-ops 2>&1 | tail -40
echo "TBO_BUILD_RC=${PIPESTATUS[0]}"
find /root/llama.cpp/build -name 'test-backend-ops' -type f 2>&1
echo "TBO_BUILD_DONE $(date -u)"
