#!/usr/bin/env bash
# Zero-skipping: measured saving vs the cost model's nnz-proportional
# assumption, at two shapes. Usage: idea2_veriflex/rtl/scripts/run_sparse_sweep.sh
set -u
RTL="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
printf '16 16 32\n4 64 8\n' | while read -r m k p; do
  iverilog -g2012 ${IVFLAGS:-} -o "$TMP/s" -DCPE_WEIGHT_DIR="\"$RTL/weights\"" \
    -DSP_M=$m -DSP_K=$k -DSP_P=$p \
    "$RTL/common/versaaccel_pkg.sv" "$RTL"/versaaccel_rtl/*.sv \
    "$RTL/testbenches/tb_gemm_sparse.sv" 2>/dev/null || { echo "build failed $m $k $p"; continue; }
  "$TMP/s" 2>&1 | sed -n '/^====/,/^====$/p;/PASS\]\|FAIL\]/p' | grep -v '^$'
  echo
done
