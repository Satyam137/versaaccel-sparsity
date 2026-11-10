#!/usr/bin/env bash
# Run the whole RTL testbench suite and summarise. Exit 1 if anything fails.
# Usage: idea2_veriflex/rtl/scripts/run_all_tb.sh
set -u
RTL="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PKG="$RTL/common/versaaccel_pkg.sv"
NET="$RTL/versaaccel_rtl/benes_switch.sv $RTL/versaaccel_rtl/distribution_network.sv $RTL/versaaccel_rtl/reduction_network.sv"
FULL="$RTL/versaaccel_rtl"
fail=0

run() {  # run <name> <sources...>
  local name="$1"; shift
  if ! iverilog -g2012 ${IVFLAGS:-} -o "$TMP/$name" -DCPE_WEIGHT_DIR="\"$RTL/weights\"" \
       $PKG "$@" 2>"$TMP/$name.log"; then
    printf "  %-22s BUILD FAILED\n" "$name"; fail=1; return
  fi
  local out; out=$("$TMP/$name" 2>&1)
  local line; line=$(echo "$out" | grep -E '^\s*\[(PASS|FAIL)\]' | tail -1 | sed 's/^ *//')
  if echo "$out" | grep -q '\[FAIL\]'; then fail=1; fi
  printf "  %-22s %s\n" "$name" "${line:-NO RESULT LINE}"
}

echo "=== VersaAccel RTL testbench suite (NR x NC from versaaccel_pkg) ==="
run tb_systolic        "$RTL/systolic_baseline/systolic_top.sv" "$RTL/testbenches/tb_systolic.sv"
run tb_systolic_golden "$RTL/systolic_baseline/systolic_top.sv" "$RTL/testbenches/tb_systolic_golden.sv"
run tb_sigma           $NET "$RTL/sigma_baseline/sigma_top.sv" "$RTL/testbenches/tb_sigma.sv"
run tb_sigma_golden    $NET "$RTL/sigma_baseline/sigma_top.sv" "$RTL/testbenches/tb_sigma_golden.sv"
run tb_datapath_golden "$FULL"/*.sv "$RTL/testbenches/tb_datapath_golden.sv"
run tb_cpe_golden      "$FULL"/*.sv "$RTL/testbenches/tb_cpe_golden.sv"
run tb_versaaccel      "$FULL"/*.sv "$RTL/testbenches/tb_versaaccel.sv"
run tb_versaaccel_golden "$FULL"/*.sv "$RTL/testbenches/tb_versaaccel_golden.sv"
run tb_gemm_golden     "$FULL"/*.sv "$RTL/testbenches/tb_gemm_golden.sv"
run tb_gemm_sparse     "$FULL"/*.sv "$RTL/testbenches/tb_gemm_sparse.sv"
run tb_layer_config    "$FULL"/*.sv "$RTL/testbenches/tb_layer_config.sv"
echo "==================================================================="
if [ "$fail" -eq 0 ]; then echo "  ALL TESTBENCHES PASS"; else echo "  FAILURES PRESENT"; fi
exit $fail
