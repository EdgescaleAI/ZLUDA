#!/bin/bash
set -x
echo "ZBUILD_START $(date -u)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y 2>&1 | tail -2
apt-get install -y curl git git-lfs cmake build-essential pkg-config 2>&1 | tail -3
git lfs install 2>&1 | tail -2
# Rust toolchain
if [ ! -d /root/.rustup ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1 | tail -3
fi
source /root/.cargo/env
rustc --version
# 4 ROCm -dev math libs (image ships runtime not -dev)
apt-get install -y rocblas-dev hipblaslt-dev miopen-hip-dev rocsparse-dev 2>&1 | tail -3
# Clone ZLUDA fork branch
if [ ! -d /root/ZLUDA ]; then
  git clone --branch overnight/gfx1151-extend --depth 1 https://github.com/EdgescaleAI/ZLUDA /root/ZLUDA 2>&1 | tail -5
fi
cd /root/ZLUDA
echo "ZLUDA_HEAD $(git rev-parse HEAD)"
git lfs pull 2>&1 | tail -3
git submodule update --init --depth 1 2>&1 | tail -5
# verify lfs bitcode real (not pointer stub)
ls -la $(find . -name 'ockl.bc' -o -name 'ocml.bc' 2>/dev/null) 2>&1
echo "ZBUILD_CARGO_START $(date -u)"
cargo xtask --release 2>&1 | tail -40
echo "ZBUILD_CARGO_RC=${PIPESTATUS[0]}"
ls -la /root/ZLUDA/target/release/libnvcuda.so 2>&1
# symlinks so loader finds libcuda / libcublas under ZLUDA names
cd /root/ZLUDA/target/release
for s in libcuda.so libcuda.so.1; do ln -sf libnvcuda.so $s; done
ls -la libnvcuda.so libcuda.so* libcublas* 2>&1
echo "ZBUILD_DONE $(date -u)"
