#!/usr/bin/env bash
# run_college.sh -- Genus batch 5 (Rev 21): the cost of reshaping, re-measured.
#
#   csh ; source ~/cshrc_new          (puts genus on PATH)
#   cd <this folder: project/syn/genus>
#   nohup bash run_college.sh >& OUTPUT_r22.log &
#
# Batch 3 put the three-shape reshapeable array (B15, -DVERSAACCEL_RESHAPE) at
# +57% area and 0.28 ns over a 2.5 ns clock -- an area figure inflated by the
# missed clock. Two runs settle what reshaping costs:
#   1  reshape_tall at 2.5 ns  -- square + TALL only (Rev 21), which keeps
#                                 almost all of the sparsity gain (section 2.9)
#   2  reshape at 2.8 ns       -- the three-shape array at a clock it can meet
#                                 (its path was 2.78 ns), so its area is not
#                                 inflated by timing pressure
# The default build's 2.5 ns area (1.864 mm2, batches 3 and 4) is the reference.
# Results: OUTPUT_r22/, one line per run in OUTPUT_r22/SUMMARY.txt.
set -u
cd "$(dirname "$0")"
command -v genus >/dev/null || { echo "genus not on PATH -- source ~/cshrc_new first"; exit 1; }
: "${OUTDIR:=OUTPUT_r22}"
: "${CORNER:=slow_vdd1v0}"

miss=0
for f in ../../idea2_veriflex/rtl/common/versaaccel_pkg.sv \
         ../../idea2_veriflex/rtl/versaaccel_rtl/versaaccel_top.sv \
         ../../idea2_veriflex/rtl/versaaccel_rtl/cpe_weights_rom.svh \
         syn_college.tcl; do
  [ -f "$f" ] || { echo "MISSING: $(pwd)/$f"; miss=1; }
done
grep -q VERSAACCEL_RESHAPE_TALL ../../idea2_veriflex/rtl/common/versaaccel_pkg.sv \
  || { echo "OLD RTL (no VERSAACCEL_RESHAPE_TALL)"; miss=1; }
grep -q reshape_tall syn_college.tcl || { echo "OLD syn_college.tcl (no reshape_tall)"; miss=1; }
[ "$miss" = 0 ] || { echo "Preflight failed -- nothing started."; exit 1; }
echo "Preflight OK: RTL and scripts present."
mkdir -p "$OUTDIR/logs"

synth() {   # variant period
  local tag="versaaccel_top_$1_${CORNER}_$2ns"
  if [ -f "$OUTDIR/$tag/DONE" ]; then echo "skip $tag (done)"; return; fi
  echo "=== $tag  ($(date +%H:%M)) ==="
  DESIGN=versaaccel_top VARIANT=$1 CLK_PERIOD=$2 POWER=0 CORNER=$CORNER OUTDIR=$OUTDIR \
    genus -batch -files syn_college.tcl -log "$OUTDIR/logs/$tag" > "$OUTDIR/logs/$tag.out" 2>&1
  grep SUMMARY "$OUTDIR/logs/$tag.out" | tail -1 || echo "  no SUMMARY -- see $OUTDIR/logs/$tag.out"
}

synth reshape_tall 2.5
synth reshape      2.8

echo; echo "=== $OUTDIR/SUMMARY.txt ==="; cat "$OUTDIR/SUMMARY.txt" 2>/dev/null
echo "Zip $OUTDIR/ (without *.db and *_netlist.v) and send it back."
