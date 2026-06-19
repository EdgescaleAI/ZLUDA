#!/bin/bash
# Run each ggml op through test-backend-ops separately so one hard CUDA error
# can't abort the rest. Emits a clean per-op verdict line.
echo "PEROP_START $(date -u)"
ZL=/root/ZLUDA/target/release
export LD_LIBRARY_PATH="$ZL:/usr/local/cuda-12.4/lib64:/opt/rocm/lib"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
TBO=/root/llama.cpp/build/bin/test-backend-ops
cd /root/llama.cpp
OPS=$("$TBO" --list-ops 2>/dev/null | sed -n 's/^GGML operations://p; ' )
OPS=$("$TBO" --list-ops 2>/dev/null | grep -oE '\b[A-Z][A-Z0-9_]+\b' | grep -vE '^GGML$|^Total$' | sort -u)
strip() { sed -r 's/\x1b\[[0-9;]*m//g'; }
for op in $OPS; do
  out=$(timeout 300 "$TBO" test -o "$op" -b CUDA0 < /dev/null 2>&1 | strip)
  rc=$?
  ok=$(echo "$out" | grep -cE '\): OK')
  notsup=$(echo "$out" | grep -cE 'not supported \[CUDA0\]')
  fail=$(echo "$out" | grep -cE '\): FAIL')
  cudaerr=$(echo "$out" | grep -c 'CUDA error')
  verdict="PASS"
  [ "$ok" -eq 0 ] && [ "$notsup" -gt 0 ] && [ "$fail" -eq 0 ] && [ "$cudaerr" -eq 0 ] && verdict="NOTSUP_GGML"
  [ "$fail" -gt 0 ] && verdict="FAIL"
  [ "$cudaerr" -gt 0 ] && verdict="CUDA_ERROR(rc=$rc)"
  errline=$(echo "$out" | grep -E 'CUDA error|): FAIL' | head -1)
  echo "PEROP $op ok=$ok notsup=$notsup fail=$fail cudaerr=$cudaerr -> $verdict ${errline}"
done
echo "PEROP_DONE $(date -u)"
