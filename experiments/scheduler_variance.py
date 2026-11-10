"""
scheduler_variance.py -- speedups with error bars, over all inference passes.

FIX_PLAN.md M6. Every result in this project was previously computed from
`df[df.input_idx == 0]` -- a single inference pass out of the 200 profiled. The
traces hold 149,200 rows and roughly 500 were used. Single-sample numbers have
no error bars, so there was no way to tell a real 1.44x from noise.

This re-runs the Static / Greedy / DP / Oracle comparison independently for
every pass and reports mean, standard deviation, and a 95% CI across passes.

Activation sparsity genuinely varies pass to pass (different inputs produce
different ReLU/attention sparsity), so the spread here is the real quantity of
interest: it says whether a workload's config schedule is stable across inputs
or whether it flips around, which is exactly what a runtime adaptive scheduler
is supposed to handle.

Usage:
    python experiments/scheduler_variance.py [--passes N]
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "idea1_adaptaccel"))

from base_simulator.arch_config import ARRAY_ROWS, ARRAY_COLS, describe
from base_simulator.cost_model import CostModel
from base_simulator.layer_cost import get_layer_costs
from base_simulator.trace_validation import validate_traces
import scheduler.dp_scheduler as sched

DATA_DIR = os.path.join(REPO, "idea1_adaptaccel", "profiling_data")
OUT_DIR = os.path.join(REPO, "results")
# B2: under VERSAACCEL_SWITCH_COST=rtl the results go to *_rtlswitch.csv, beside the
# estimate-based ones every published number comes from.
SFX = "_rtlswitch" if os.environ.get("VERSAACCEL_SWITCH_COST") == "rtl" else ""
LOOKAHEAD = 5


def evaluate_pass(group, cost_model):
    """Static / Greedy / DP / Oracle for one inference pass."""
    costs = []
    for _, row in group.iterrows():
        attn = float(row["attention_sparsity"]) if "attention_sparsity" in row.index else 0.0
        act_sp = max(float(row["input_sparsity"]), attn)
        costs.append(get_layer_costs(cost_model, row, act_sp))

    oracle = sched.schedule_oracle(costs)
    oracle_cost = sched.rescore_schedule(oracle["schedule"], costs)["total_cost"]

    # static pays the same initial reconfiguration as every scheduler (B4)
    static = min(
        sum(c[cfg] for c in costs)
        + sched.SWITCH_COST[sched.CONFIG_IDX["DMR"]][sched.CONFIG_IDX[cfg]]
        for cfg in sched.CONFIG_NAMES
    )

    greedy = sched.schedule_greedy(costs)
    greedy_cost = sched.rescore_schedule(greedy["schedule"], costs)["total_cost"]
    dp = sched.schedule_dp(costs, lookahead=LOOKAHEAD)
    dp_cost = sched.rescore_schedule(dp["schedule"], costs)["total_cost"]

    nsw = lambda s: sum(1 for i in range(1, len(s)) if s[i] != s[i - 1])
    return {
        "speedup": static / oracle_cost,
        "greedy_gap": (greedy_cost - oracle_cost) / oracle_cost * 100,
        "dp_gap": (dp_cost - oracle_cost) / oracle_cost * 100,
        "oracle_switches": nsw(oracle["schedule"]),
        "schedule": "".join(str(sched.CONFIG_IDX[c]) for c in oracle["schedule"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=None,
                    help="limit number of inference passes (default: all)")
    args = ap.parse_args()

    print("=" * 94)
    print("  M6: Scheduler results across ALL inference passes (with error bars)")
    print("=" * 94)
    print(f"\n  {describe()}\n")

    cost_model = CostModel(Nr=ARRAY_ROWS, Nc=ARRAY_COLS)

    print(f"  {'model':<22} {'passes':>7} {'speedup (mean +/- sd)':>24} {'95% CI':>18} "
          f"{'uniq':>5}")
    print("  " + "-" * 88)

    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_traces.csv"))):
        name = os.path.basename(path).replace("_traces.csv", "")
        df = pd.read_csv(path)
        validate_traces(df, name)

        pass_ids = sorted(df["input_idx"].unique())
        if args.passes:
            pass_ids = pass_ids[: args.passes]

        per_pass = []
        for pid in pass_ids:
            g = df[df["input_idx"] == pid].reset_index(drop=True)
            if len(g) < 2:
                continue
            per_pass.append(evaluate_pass(g, cost_model))
        if not per_pass:
            continue

        sp = np.array([r["speedup"] for r in per_pass])
        gg = np.array([r["greedy_gap"] for r in per_pass])
        dg = np.array([r["dp_gap"] for r in per_pass])
        sw = np.array([r["oracle_switches"] for r in per_pass])
        uniq = len({r["schedule"] for r in per_pass})

        n = len(sp)
        ci = 1.96 * sp.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        rows.append({
            "model": name, "n_passes": n,
            "speedup_mean": sp.mean(), "speedup_std": sp.std(ddof=1) if n > 1 else 0.0,
            "speedup_ci95": ci, "speedup_min": sp.min(), "speedup_max": sp.max(),
            "greedy_gap_mean": gg.mean(), "dp_gap_mean": dg.mean(),
            "switches_mean": sw.mean(), "switches_std": sw.std(ddof=1) if n > 1 else 0.0,
            "unique_schedules": uniq,
        })
        print(f"  {name:<22} {n:>7} {sp.mean():>14.3f}x +/- {sp.std(ddof=1):<6.3f} "
              f"[{sp.mean()-ci:>6.3f}, {sp.mean()+ci:<6.3f}] {uniq:>5}")

    df_out = pd.DataFrame(rows)

    print("\n" + "=" * 94)
    print("  AGGREGATE")
    print("=" * 94)

    m = df_out.speedup_mean
    overall_ci = 1.96 * m.std(ddof=1) / np.sqrt(len(m))
    print(f"\n  Mean speedup across {len(df_out)} workloads: "
          f"{m.mean():.3f}x +/- {overall_ci:.3f} (95% CI)")
    print(f"  Range: {df_out.speedup_min.min():.3f}x to {df_out.speedup_max.max():.3f}x")

    # the split that actually matters
    hetero = df_out[df_out.speedup_mean > 1.10]
    homo = df_out[df_out.speedup_mean <= 1.10]
    if len(hetero):
        print(f"\n  Heterogeneous workloads (n={len(hetero)}): "
              f"{hetero.speedup_mean.mean():.3f}x mean, "
              f"up to {hetero.speedup_max.max():.3f}x")
    if len(homo):
        print(f"  Homogeneous workloads   (n={len(homo)}): "
              f"{homo.speedup_mean.mean():.3f}x mean "
              f"-- one config is optimal for the whole network")
    print(f"\n  Quote the split, never the blended average alone.")

    # schedule stability across inputs
    unstable = df_out[df_out.unique_schedules > 1]
    print(f"\n  Workloads whose optimal schedule VARIES across inputs: "
          f"{len(unstable)}/{len(df_out)}")
    if len(unstable):
        for _, r in unstable.iterrows():
            print(f"    {r['model']:<22} {int(r['unique_schedules'])} distinct schedules "
                  f"over {int(r['n_passes'])} passes")
        print(f"\n  These are the workloads where a RUNTIME scheduler is actually")
        print(f"  required; the rest could be scheduled once at compile time.")
    else:
        print(f"  None. Every workload has one optimal schedule across all inputs,")
        print(f"  so config selection could be done once at compile time.")

    os.makedirs(OUT_DIR, exist_ok=True)
    df_out.to_csv(os.path.join(OUT_DIR, "scheduler_variance" + SFX + ".csv"), index=False)
    print(f"\n  Saved to {OUT_DIR}/scheduler_variance.csv")


if __name__ == "__main__":
    main()
