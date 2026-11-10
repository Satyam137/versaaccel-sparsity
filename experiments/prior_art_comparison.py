"""B4: per-layer dataflow selection against fixed-dataflow accelerators, in the
form prior art reports it.

Flexagon (Munoz-Martinez et al., ASPLOS 2023) reports per-layer dataflow
selection against accelerators fixed to one dataflow -- 4.59x over SIGMA-like
(inner product), 1.71x over SpArch-like (outer product), 1.35x over GAMMA-like
(Gustavson), averaged over its DNN layers. Section 3 reports something
different: selection against the best fixed configuration *chosen per
workload with hindsight*, which is the hardest baseline there is.

This puts the project's numbers in Flexagon's form, on the same cost model and
real traces as section 3:

  vs each fixed configuration   one dataflow for every layer of every workload
  vs best-per-workload static   section 3's baseline

Both use the oracle per-layer schedule with switch costs (section 4 shows
greedy is within 0.003% of it). The comparison is of *experiment design*, not
of hardware: Flexagon is SpMSpM on a MAERI-style 64-multiplier fabric with
different dataflows and workloads, so equal numbers would not mean equal
accelerators. What it shows is how much of a reported "x over fixed
dataflow" depends on which fixed dataflow is the baseline.

Usage: python experiments/prior_art_comparison.py [--passes 20]
Writes results/prior_art_comparison.csv.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "idea1_adaptaccel"))

from base_simulator.arch_config import ARRAY_ROWS, ARRAY_COLS   # noqa: E402
from base_simulator.cost_model import CostModel                 # noqa: E402
from base_simulator.layer_cost import get_layer_costs           # noqa: E402
import scheduler.dp_scheduler as sched                           # noqa: E402

DATA_DIR = os.path.join(REPO, "idea1_adaptaccel", "profiling_data")
REAL = ("resnet50_", "bert_")
FLEXAGON = {"SIGMA-like (inner product)": 4.59, "SpArch-like (outer product)": 1.71,
            "GAMMA-like (Gustavson)": 1.35}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=20)
    args = ap.parse_args()
    cm = CostModel(Nr=ARRAY_ROWS, Nc=ARRAY_COLS)
    init = sched.CONFIG_IDX["DMR"]
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_traces.csv"))):
        name = os.path.basename(path).replace("_traces.csv", "")
        df = pd.read_csv(path)
        for pid in sorted(df["input_idx"].unique())[: args.passes]:
            g = df[df["input_idx"] == pid].reset_index(drop=True)
            costs = []
            for _, row in g.iterrows():
                attn = float(row["attention_sparsity"]) if "attention_sparsity" in row.index else 0.0
                costs.append(get_layer_costs(cm, row, max(float(row["input_sparsity"]), attn)))
            oracle = sched.rescore_schedule(sched.schedule_oracle(costs)["schedule"], costs)["total_cost"]
            rec = dict(model=name, real=name.startswith(REAL), input_idx=pid, oracle=oracle)
            for cfg in sched.CONFIG_NAMES:
                rec[cfg] = sum(c[cfg] for c in costs) + sched.SWITCH_COST[init][sched.CONFIG_IDX[cfg]]
            rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(REPO, "results", "prior_art_comparison.csv"), index=False)

    w = d.groupby(["model", "real"]).mean(numeric_only=True).reset_index()
    for cfg in sched.CONFIG_NAMES:
        w[f"x_{cfg}"] = w[cfg] / w.oracle
    w["x_best_static"] = w[list(sched.CONFIG_NAMES)].min(axis=1) / w.oracle

    def gmean(x):
        return float(np.exp(np.log(x).mean()))

    print("=" * 92)
    print("  B4: per-layer selection vs fixed-dataflow baselines (Flexagon's form), 16x16")
    print("=" * 92)
    print(f"  {'workload':22s}" + "".join(f"{'vs ' + c:>11s}" for c in sched.CONFIG_NAMES)
          + f"{'vs best':>11s}")
    for _, r in w.iterrows():
        print(f"  {r.model:22s}" + "".join(f"{r['x_' + c]:>10.2f}x" for c in sched.CONFIG_NAMES)
              + f"{r.x_best_static:>10.2f}x")
    for label, sub in (("10 real workloads", w[w.real]), ("all 15 workloads", w)):
        print(f"\n  {label}: mean (geomean) speedup of per-layer selection over")
        for c in sched.CONFIG_NAMES:
            print(f"    fixed {c:6s} for every workload  {sub['x_' + c].mean():6.2f}x  "
                  f"({gmean(sub['x_' + c]):.2f}x)")
        print(f"    best fixed per workload (sec. 3)  {sub.x_best_static.mean():6.2f}x  "
              f"({gmean(sub.x_best_static):.2f}x)")
        best_global = min(sched.CONFIG_NAMES, key=lambda c: sub["x_" + c].mean())
        worst_global = max(sched.CONFIG_NAMES, key=lambda c: sub["x_" + c].mean())
        print(f"    -> range over the choice of fixed baseline: {sub['x_' + best_global].mean():.2f}x"
              f" ({best_global}) to {sub['x_' + worst_global].mean():.2f}x ({worst_global})")
    print("\n  Flexagon (ASPLOS 2023), same experiment design, its own workloads and hardware:")
    for k, v in FLEXAGON.items():
        print(f"    vs {k:30s} {v:.2f}x")


if __name__ == "__main__":
    main()
