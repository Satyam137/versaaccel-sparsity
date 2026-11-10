"""Section 3.0: the section 3 headline under three cost models (Rev 19).

  A  "nnz"  the model as published (VersaAccel Table II) -- Rev 18 and earlier
  B  "mac"  A with every configuration held to the MAC lower bound of the work
            it claims to do. ADOPTED for sections 3-6.
  D  "rtl"  the RTL-measured terms: DMR's tile fill (B11), sparse configurations
            charged dense compute (MAC-bounded) credited only realisable k-slice
            skipping (B6) -- i.e. the sparse configurations costed as if run on
            this RTL's shared-operand array

The headline is section 3.2's: per-layer selection (oracle schedule with switch
costs) against the best static configuration, mean over inference passes, per
workload. Also reports which configuration each model picks, layer by layer,
and how many workloads have more than one optimal schedule across inputs.

Usage: python experiments/cost_model_variants.py [--passes N]   (~30 min at 200)
Writes results/cost_model_variants.csv.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import scheduler_variance as sv                                   # noqa: E402

MODELS = {"A": "nnz", "B": "mac", "D": "rtl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=None)
    args = ap.parse_args()
    rows = []
    paths = sorted(glob.glob(os.path.join(sv.DATA_DIR, "*_traces.csv")))
    for tag, model in MODELS.items():
        cm = sv.CostModel(Nr=sv.ARRAY_ROWS, Nc=sv.ARRAY_COLS, model=model)
        for p in paths:
            name = os.path.basename(p).replace("_traces.csv", "")
            d = pd.read_csv(p)
            per, picks = [], {}
            for pid in sorted(d.input_idx.unique())[: args.passes]:
                g = d[d.input_idx == pid].reset_index(drop=True)
                if len(g) < 2:
                    continue
                r = sv.evaluate_pass(g, cm)
                per.append(r)
                for ch in r["schedule"]:
                    picks[ch] = picks.get(ch, 0) + 1
            sp = np.array([r["speedup"] for r in per])
            n = sum(picks.values())
            rows.append(dict(model=tag, cost_model=model, workload=name, passes=len(per),
                             speedup_mean=sp.mean(), speedup_max=sp.max(),
                             unique_schedules=len({r["schedule"] for r in per}),
                             **{f"pick_{c}": picks.get(str(i), 0) / n
                                for i, c in enumerate(sv.sched.CONFIG_NAMES)}))
            print(f"  {tag} {name:22s} {sp.mean():.3f}x  schedules {rows[-1]['unique_schedules']}",
                  flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(REPO, "results", "cost_model_variants.csv"), index=False)
    real = t.workload.str.startswith(("resnet50_", "bert_"))
    het = ~t.workload.str.startswith("bert_")
    print("\n  model  all 15   heterogeneous   10 real   pruned ResNet   input-dependent   picks DMR/D-MR/DM-R/D-M-R")
    for tag, x in t.groupby("model"):
        pr = x.workload.str.startswith("resnet50_pruned")
        pk = x[[f"pick_{c}" for c in sv.sched.CONFIG_NAMES]].mean()
        print(f"  {tag}      {x.speedup_mean.mean():.3f}x   {x[het[x.index]].speedup_mean.mean():.3f}x"
              f"          {x[real[x.index]].speedup_mean.mean():.3f}x    "
              f"{x[pr].speedup_mean.min():.2f}-{x[pr].speedup_mean.max():.2f}x       "
              f"{int((x.unique_schedules > 1).sum())}/{len(x)}            "
              + " / ".join(f"{v:.0%}" for v in pk))


if __name__ == "__main__":
    main()
