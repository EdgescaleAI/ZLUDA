#!/bin/bash
ZL=/root/ZLUDA/target/release
export LD_LIBRARY_PATH="$ZL:/usr/local/cuda/lib64:/opt/rocm/lib"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
cd /root/llama.cpp
gen() {
  echo "##### PROMPT: $1"
  ( timeout 45 ./build/bin/llama-cli -m /root/tiny.gguf -ngl 99 -p "$1" -n "$2" --temp 0 -fa off --seed 1 < /dev/null 2>/dev/null ) \
    | sed -n '/^> /,$p' | grep -v '^> *$' | head -6
}
gen "The capital of Japan is" 12
gen "Two plus two equals" 8
gen "Roses are red, violets are" 8
echo "MULTI_DONE $(date -u)"
