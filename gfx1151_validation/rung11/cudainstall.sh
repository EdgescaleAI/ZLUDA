#!/bin/bash
set -x
echo "CUDAINSTALL_START $(date -u)"
export DEBIAN_FRONTEND=noninteractive
apt-get install -y wget gnupg ca-certificates 2>&1 | tail -2
cd /root
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb 2>&1 | tail -2
apt-get update -y 2>&1 | tail -2
# nvcc + cudart-dev (cuda_runtime.h) + cublas-dev (cublas_v2.h) + driver-dev (cuda.h stub)
apt-get install -y cuda-nvcc-12-4 cuda-cudart-dev-12-4 libcublas-dev-12-4 cuda-driver-dev-12-4 2>&1 | tail -4
echo "CUDAINSTALL_RC=${PIPESTATUS[0]}"
ls -la /usr/local/cuda-12.4/bin/nvcc 2>&1
echo "CUDAINSTALL_DONE $(date -u)"
