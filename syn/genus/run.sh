#!/usr/bin/env bash
# =============================================================================
# run.sh -- synthesise all three designs under identical constraints, and
# optionally sweep the clock to find each one's Fmax.
#
#   ./run.sh                 one run of each design at the setup.tcl period
#   ./run.sh --sweep         sweep the period to find Fmax per design
#   ./run.sh --design X      just one design
#
# Everything lands in results/<design>_<period>ns/.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DESIGNS=(versaaccel_top systolic_top sigma_top)
SWEEP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --sweep)  SWEEP=1; shift ;;
    --design) DESIGNS=("$2"); shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

if ! command -v genus >/dev/null 2>&1; then
  echo "genus not on PATH. Source your Cadence setup first, e.g."
  echo "    source /opt/cadence/GENUS/setup.sh"
  exit 1
fi
if grep -q '/path/to/nangate45' "$HERE/setup.tcl"; then
  echo "Edit LIB_PATH in setup.tcl first -- it still points at the placeholder."
  exit 1
fi

mkdir -p "$HERE/results" "$HERE/logs"

# Binary search would be fewer runs, but a linear sweep gives the whole
# area-vs-frequency curve, which is the more useful thing to put in a paper.
if [ "$SWEEP" -eq 1 ]; then
  PERIODS=(0.6 0.8 1.0 1.2 1.5 2.0 3.0)
else
  PERIODS=("")
fi

for d in "${DESIGNS[@]}"; do
  for p in "${PERIODS[@]}"; do
    if [ -n "$p" ]; then
      echo "=== $d @ ${p}ns ==="
      DESIGN="$d" CLK_PERIOD_NS="$p" \
        genus -batch -f "$HERE/syn.tcl" -log "$HERE/logs/${d}_${p}ns" \
        > "$HERE/logs/${d}_${p}ns.out" 2>&1
    else
      echo "=== $d ==="
      DESIGN="$d" \
        genus -batch -f "$HERE/syn.tcl" -log "$HERE/logs/${d}" \
        > "$HERE/logs/${d}.out" 2>&1
    fi
    tail -3 "$HERE/logs/${d}${p:+_${p}ns}.out" | grep SUMMARY || \
      echo "  no SUMMARY line -- check the log"
  done
done

echo
echo "=== comparison (identical constraints, see constraints.sdc) ==="
printf "  %-18s %-10s %-14s %-12s\n" design period area slack
grep -h SUMMARY "$HERE"/logs/*.out 2>/dev/null | while read -r _ d per ar sl; do
  printf "  %-18s %-10s %-14s %-12s\n" "$d" "${per#period=}" "${ar#area=}" "${sl#slack=}"
done
echo
echo "Fmax for a design = 1 / (smallest period whose slack is still >= 0)."
echo "Feed the measured reconfiguration latency into SWITCH_COST in"
echo "experiments/switch_cost_sweep.py and re-run section 4."
