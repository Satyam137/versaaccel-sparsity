#!/usr/bin/env bash
# Sweep GEMM shapes through the tiling controller, including partial tiles.
# Each shape is run twice inside the bench -- narrow drain (1 element/cycle)
# and wide drain (a whole PE column/cycle) -- and both must agree with the
# reference GEMM. The cycle columns show what the wide bus plus the output
# double buffer buy.
# Usage: idea2_veriflex/rtl/scripts/run_gemm_shapes.sh
set -u
RTL="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
printf '%-30s %-9s %-8s %-8s %-7s %s\n' "  shape" "tiles" "narrow" "wide" "gain" "result"
printf '32 6 32\n17 5 20\n16 1 16\n8 12 40\n33 3 17\n5 7 9\n' | while read -r m k p; do
  iverilog -g2012 ${IVFLAGS:-} -o "$TMP/g" -DCPE_WEIGHT_DIR="\"$RTL/weights\"" \
    -DGEMM_M=$m -DGEMM_K=$k -DGEMM_P=$p \
    "$RTL/common/versaaccel_pkg.sv" "$RTL"/versaaccel_rtl/*.sv \
    "$RTL/testbenches/tb_gemm_golden.sv" 2>/dev/null || { echo "  build failed $m $k $p"; continue; }
  out=$("$TMP/g" 2>&1)
  n=$(echo "$out" | sed -n 's/.*narrow drain, 1 elem\/cy: *\([0-9]*\) cycles.*/\1/p')
  w=$(echo "$out" | sed -n 's/.*wide drain, [0-9]* elem\/cy: *\([0-9]*\) cycles.*/\1/p')
  g=$(echo "$out" | sed -n 's/.*speedup from the wide bus: *//p')
  if echo "$out" | grep -q '\[FAIL\]'; then res="FAIL"; else res="PASS both widths"; fi
  printf "  M=%-3s K=%-3s P=%-3s          %-9s %-8s %-8s %-7s %s\n" \
    "$m" "$k" "$p" "$(echo "$out" | sed -n 's/.*tiles: //p')" "$n" "$w" "$g" "$res"
done
