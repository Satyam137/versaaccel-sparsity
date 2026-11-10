"""
switch_cost_sweep.py -- when does lookahead scheduling actually pay?

FIX_PLAN.md E2. This is the experiment that turns the repo's negative result
into a design rule.

Background
----------
The window-size ablation in dp_scheduler.py reports W=1 within 0.0033% of
Oracle on every workload. That is not evidence that the DP is well-tuned; it is
evidence that the scheduling problem is trivial *at the assumed switch costs*.
Reconfiguration is modelled at 100-250 cycles while layers cost 10^4-10^6, so a
config that wins by more than a fraction of a percent on even one layer always
justifies switching immediately. Lookahead has nothing to defer.

The interesting question is therefore not "does DP beat greedy here" (it does
not) but "how expensive would reconfiguration have to be before it did?" That
threshold is a property of the interconnect, and it tells a hardware designer
how much reconfiguration latency their design can afford before they need a
scheduler smarter than greedy.

Method
------
Scale the entire SWITCH_COST matrix by a sweep of factors. At each point,
schedule every workload with Static / Greedy / DP(W=1,2,3,5) / Oracle, score
all of them on one cost function, and record the gap to Oracle. Report the
switch cost both absolutely and as a fraction of median layer latency, which is
the dimensionless quantity that generalises across array sizes.

Usage:
    python experiments/switch_cost_sweep.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "idea1_adaptaccel"))

from base_simulator.arch_config import ARRAY_ROWS, ARRAY_COLS, NUM_PES, describe
from base_simulator.cost_model import CostModel
from base_simulator.layer_cost import get_layer_costs
from base_simulator.trace_validation import validate_traces
import scheduler.dp_scheduler as sched

DATA_DIR = os.path.join(REPO, "idea1_adaptaccel", "profiling_data")
OUT_DIR = os.path.join(REPO, "results")
# B2: under VERSAACCEL_SWITCH_COST=rtl the results go to *_rtlswitch.csv, beside the
# estimate-based ones every published number comes from.
SFX = "_rtlswitch" if os.environ.get("VERSAACCEL_SWITCH_COST") == "rtl" else ""

# Multipliers on the baseline SWITCH_COST matrix (whose max entry is 250 cycles)
SCALES = [0.1, 0.4, 1, 4, 10, 40, 100, 400, 1000, 4000, 10000]
# Under the RTL-measured matrix (B2) only the measured point and zero are run:
# the crossover itself is the estimate-based sweep's, a function of switch cost
# in cycles, and the measured point is far below it.
if SFX:
    SCALES = [0, 1]
WINDOWS = [1, 2, 3, 5]

# A window "matters" once it beats W=1 by more than this, in percent of Oracle.
MATTERS_PCT = 1.0


def load_workloads():
    """One representative inference pass per trace file, with costs precomputed."""
    workloads = []
    cost_model = CostModel(Nr=ARRAY_ROWS, Nc=ARRAY_COLS)
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_traces.csv"))):
        name = os.path.basename(path).replace("_traces.csv", "")
        df = pd.read_csv(path)
        validate_traces(df, name)
        single = df[df["input_idx"] == 0].reset_index(drop=True)
        if len(single) < 2:
            continue
        costs = []
        for _, row in single.iterrows():
            attn = float(row["attention_sparsity"]) if "attention_sparsity" in row.index else 0.0
            act_sp = max(float(row["input_sparsity"]), attn)
            costs.append(get_layer_costs(cost_model, row, act_sp))
        workloads.append({
            "name": name,
            "costs": costs,
            "n_layers": len(costs),
            "median_layer": float(np.median([min(c.values()) for c in costs])),
        })
    return workloads


def evaluate_at_scale(workloads, scale, base_matrix):
    """Schedule every workload with SWITCH_COST scaled by `scale`."""
    sched.SWITCH_COST = base_matrix * scale
    rows = []
    for w in workloads:
        costs = w["costs"]

        oracle = sched.schedule_oracle(costs)
        oracle_cost = sched.rescore_schedule(oracle["schedule"], costs)["total_cost"]

        # Static baseline pays the same initial reconfiguration (FIX_PLAN.md B4)
        static = min(
            sum(c[cfg] for c in costs)
            + sched.SWITCH_COST[sched.CONFIG_IDX["DMR"]][sched.CONFIG_IDX[cfg]]
            for cfg in sched.CONFIG_NAMES
        )

        greedy = sched.schedule_greedy(costs)
        greedy_cost = sched.rescore_schedule(greedy["schedule"], costs)["total_cost"]

        row = {
            "scale": scale,
            "model": w["name"],
            "max_switch_cycles": float(sched.SWITCH_COST.max()),
            "median_layer_cycles": w["median_layer"],
            "switch_frac_of_layer": sched.SWITCH_COST.max() / w["median_layer"],
            "oracle_cost": oracle_cost,
            "static_gap": (static - oracle_cost) / oracle_cost * 100,
            "greedy_gap": (greedy_cost - oracle_cost) / oracle_cost * 100,
        }
        for win in WINDOWS:
            res = sched.schedule_dp(costs, lookahead=win)
            c = sched.rescore_schedule(res["schedule"], costs)["total_cost"]
            row[f"w{win}_gap"] = (c - oracle_cost) / oracle_cost * 100
            row[f"w{win}_sw"] = sum(
                1 for i in range(1, len(res["schedule"]))
                if res["schedule"][i] != res["schedule"][i - 1]
            )
        rows.append(row)
    return rows


def main():
    print("=" * 96)
    print("  E2: Switch-Cost Sensitivity Sweep -- where does lookahead start to pay?")
    print("=" * 96)

    workloads = load_workloads()
    base_matrix = sched.SWITCH_COST.copy()
    print(f"\n  {len(workloads)} workloads | baseline max switch cost = "
          f"{base_matrix.max():.0f} cycles | windows tested: {WINDOWS}\n")

    all_rows = []
    for scale in SCALES:
        all_rows.extend(evaluate_at_scale(workloads, scale, base_matrix))
    sched.SWITCH_COST = base_matrix  # restore

    df = pd.DataFrame(all_rows)

    # ---- aggregate view ----
    print(f"  {'switch':>9} {'as % of':>9} | {'mean gap vs Oracle (%)':^42} | {'mean switches':^19}")
    print(f"  {'cycles':>9} {'layer':>9} | {'Static':>8} {'Greedy':>8} {'W=1':>7} {'W=2':>7} "
          f"{'W=5':>7} | {'W=1':>8} {'W=5':>8}")
    print("  " + "-" * 92)

    summary = []
    for scale in SCALES:
        s = df[df.scale == scale]
        rec = {
            "scale": scale,
            "max_switch_cycles": s.max_switch_cycles.iloc[0],
            "switch_frac": s.switch_frac_of_layer.mean(),
            "static_gap": s.static_gap.mean(),
            "greedy_gap": s.greedy_gap.mean(),
            "w1_gap": s.w1_gap.mean(),
            "w2_gap": s.w2_gap.mean(),
            "w3_gap": s.w3_gap.mean(),
            "w5_gap": s.w5_gap.mean(),
            "w1_sw": s.w1_sw.mean(),
            "w5_sw": s.w5_sw.mean(),
        }
        summary.append(rec)
        marker = "   <-- as modelled" if scale == 1 else ""
        print(f"  {rec['max_switch_cycles']:>9,.0f} {rec['switch_frac']:>8.1%} | "
              f"{rec['static_gap']:>8.1f} {rec['greedy_gap']:>8.3f} {rec['w1_gap']:>7.3f} "
              f"{rec['w2_gap']:>7.3f} {rec['w5_gap']:>7.3f} | "
              f"{rec['w1_sw']:>8.1f} {rec['w5_sw']:>8.1f}{marker}")

    sdf = pd.DataFrame(summary)

    # ---- locate the crossover ----
    print("\n" + "=" * 96)
    print("  CROSSOVER ANALYSIS")
    print("=" * 96)

    sdf["w1_minus_w5"] = sdf.w1_gap - sdf.w5_gap
    crossed = sdf[sdf.w1_minus_w5 > MATTERS_PCT]

    baseline_frac = sdf[sdf.scale == 1].switch_frac.iloc[0]
    print(f"\n  As modelled, reconfiguration is {sdf[sdf.scale==1].max_switch_cycles.iloc[0]:.0f} "
          f"cycles = {baseline_frac:.2%} of a median layer.")
    print(f"  At that point W=1 and W=5 differ by "
          f"{sdf[sdf.scale==1].w1_minus_w5.iloc[0]:.4f}% -- indistinguishable.\n")

    if len(crossed) == 0:
        print(f"  Lookahead never gains more than {MATTERS_PCT}% anywhere in the swept range")
        print(f"  (up to {sdf.max_switch_cycles.max():,.0f} cycles = "
              f"{sdf.switch_frac.max():.0%} of a layer).")
        print(f"  => A switch-cost-aware greedy rule is sufficient for any plausible")
        print(f"     reconfigurable interconnect. Lookahead scheduling is not justified.")
    else:
        first = crossed.iloc[0]
        print(f"  Lookahead first gains >{MATTERS_PCT}% at {first.max_switch_cycles:,.0f} cycles")
        print(f"  = {first.switch_frac:.1%} of a median layer "
              f"({first.max_switch_cycles / sdf[sdf.scale==1].max_switch_cycles.iloc[0]:.0f}x "
              f"the modelled cost).")
        print(f"\n  => DESIGN RULE: lookahead scheduling is justified only when")
        print(f"     reconfiguration exceeds ~{first.switch_frac:.0%} of layer latency.")
        print(f"     Below that, greedy is within {MATTERS_PCT}% of optimal.")

    # where does adapting at all stop being worth it?
    stopped = sdf[sdf.static_gap < 1.0]
    if len(stopped):
        f = stopped.iloc[0]
        print(f"\n  Adaptation itself stops paying at {f.max_switch_cycles:,.0f} cycles "
              f"({f.switch_frac:.0%} of a layer):")
        print(f"  beyond this the best static config is within 1% of Oracle, so the")
        print(f"  accelerator should simply not reconfigure.")

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "switch_cost_sweep_per_model" + SFX + ".csv"), index=False)
    sdf.to_csv(os.path.join(OUT_DIR, "switch_cost_sweep_summary" + SFX + ".csv"), index=False)
    print(f"\n  Saved to {OUT_DIR}/switch_cost_sweep_{{summary,per_model}}.csv")


if __name__ == "__main__":
    main()
