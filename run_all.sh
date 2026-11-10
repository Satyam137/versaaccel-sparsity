#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- regenerate every number in PROJECT_REPORT.md from a checkout (R4)
#
#   ./run_all.sh            everything except re-profiling     (~4.5 h on an M-series laptop)
#   ./run_all.sh --quick    skip the multi-hour runs            (~50 min)
#   ./run_all.sh --profile  ALSO re-profile the traces first    (+ ~2 h, needs network once)
#
# Needs: Python 3.11 + requirements.txt, and Icarus Verilog 13 for the RTL steps.
# Synthesis (syn/genus) is not run here -- it needs Genus; see syn/genus/README.md.
# Ends with experiments/audit_report_numbers.py, which fails if any recomputed
# headline number no longer appears verbatim in the report.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
QUICK=0; PROFILE=0
for a in "$@"; do
  case $a in --quick) QUICK=1 ;; --profile) PROFILE=1 ;; *) echo "unknown option $a"; exit 2 ;; esac
done

step() { echo; echo "=== [$(date +%H:%M:%S)] $1"; shift; "$@"; }

if [ $PROFILE = 1 ]; then
  step "profile ResNet-50 (Imagenette)"   bash -c "cd idea1_adaptaccel && $PY -m profiler.profile_resnet --download"
  step "profile BERT (SST-2)"             bash -c "cd idea1_adaptaccel && $PY -m profiler.profile_bert --dataset sst2"
  step "profile N:M models"               $PY idea2_veriflex/profile_nm_models.py --download --dataset sst2
  step "profile MobileNetV2 (B9)"         bash -c "cd idea1_adaptaccel && $PY -m profiler.profile_resnet --download --arch mobilenet_v2 --out-dir profiling_data/b9"
  step "profile OPT-125m (B9)"            bash -c "cd idea1_adaptaccel && $PY -m profiler.profile_opt --dataset sst2"
  step "refresh trace manifest (R6)"      $PY experiments/verify_traces.py --update
fi

step "R6 traces match MANIFEST.sha256"    $PY experiments/verify_traces.py

# --- cost model, scheduling, prediction (sections 2-6, 8) --------------------
step "cost model vs paper Table II"       $PY -m base_simulator.validate
step "3.1, 4.1 scheduler + window ablation" $PY idea1_adaptaccel/scheduler/dp_scheduler.py
step "3.2-3.3 speedup over 200 passes"    $PY experiments/scheduler_variance.py
step "3.0 cost models A / B / D"         $PY experiments/cost_model_variants.py
step "3.4 prior-art form (B4)"            $PY experiments/prior_art_comparison.py
step "3.5 operating-point grid (B8)"      $PY experiments/operating_point_grid.py
step "4.2 switch-cost sweep"              $PY experiments/switch_cost_sweep.py
step "5.1 array-size sweep"               $PY experiments/array_size_sweep.py
step "5.2 predictor baselines"            $PY experiments/predictor_baselines.py
step "2.4 nnz-row estimator"              $PY experiments/nnz_rows_sensitivity.py
step "2.6 structure in cycles"            $PY experiments/structure_value.py
step "5.3 structure predictor, 10 seeds"  $PY experiments/structure_predictor.py
step "5.3b structural cost model (B6)"    $PY experiments/structure_decisions.py
step "B9 two more model families"         $PY experiments/generalisation_b9.py
step "6.1 N:M as a fifth config"          $PY -m idea2_veriflex.integrated_scheduler
step "8.1 speculation sensitivity"        $PY idea4_speculative/speculative_engine.py --sensitivity
step "7.12 cost model vs RTL (B3)"        $PY experiments/costmodel_vs_rtl.py

# --- tile geometry (sections 2.7-2.9) ----------------------------------------
step "2.7-2.8 tile height + regrouping"   $PY experiments/tile_geometry.py --images 16 --texts 24
step "2.9 tile shape"                     $PY experiments/tile_shape.py
step "2.9 bandwidth budget (B16)"         $PY experiments/tile_shape_bandwidth.py

# --- RTL (section 7, B13, B15) -----------------------------------------------
step "RTL suite (13,215 golden checks)"   bash idea2_veriflex/rtl/scripts/run_all_tb.sh
step "RTL GEMM shapes"                    bash idea2_veriflex/rtl/scripts/run_gemm_shapes.sh
step "RTL zero-skip sweep"                bash idea2_veriflex/rtl/scripts/run_sparse_sweep.sh
step "2.8 regrouping on the RTL (B13)"    $PY experiments/regroup_rtl.py
step "2.9 shapes on the RTL"              $PY experiments/tile_shape_rtl.py
step "B15 reshapeable array on the RTL"   $PY experiments/reshape_rtl.py
step "B10 RTL suite, fill amortised"      env IVFLAGS=-DVERSAACCEL_AMORTISE bash idea2_veriflex/rtl/scripts/run_all_tb.sh
if [ $QUICK = 1 ]; then   # the full run below includes these parts
  step "B10/B12 amortised + streamed RTL"   $PY experiments/amortise_rtl.py --quick
fi
step "B1c activity VCDs for Genus"        $PY experiments/make_activity.py

if [ $QUICK = 0 ]; then
  step "5.2 predictor over 10 seeds (A8), ~15 min"  $PY experiments/predictor_seed_robustness.py
  step "7.12 headline with/without fill, ~20 min"   $PY experiments/costmodel_vs_rtl.py --headline
  step "B10/B12 amortised + streamed RTL, ~75 min" $PY experiments/amortise_rtl.py
  step "2.8 at 200 inputs, sample 0 (B14), ~45 min" $PY experiments/tile_geometry.py --images 200 --texts 200 --chunk 10 --seed 0
  step "2.8 at 200 inputs, sample 1 (B14), ~45 min" $PY experiments/tile_geometry.py --images 200 --texts 200 --chunk 10 --seed 1
  # the 16-input run writes the committed CSVs; re-run it last so they stay current
  step "2.7-2.8 tile height + regrouping (final)"   $PY experiments/tile_geometry.py --images 16 --texts 24
fi

step "A2 audit: report numbers vs results/"  $PY experiments/audit_report_numbers.py
echo; echo "=== all steps completed ==="
