"""B8: the headline speedup over one operating-point grid.

Section 3's 1.184x is measured at one point: 16x16, the default memory
bandwidth, the modelled switch costs. Sections 4.2 and 5.1 each vary one of
those axes alone. This runs the headline -- per-layer selection (oracle
schedule with switch costs; section 4 shows greedy matches it) against the
best static configuration per workload -- over all three together:

  array          8x8, 16x16, 32x32
  bandwidth      2, 4, 8, 32 elements/cycle
  switch cost    x1, x10, x100 the modelled matrix (max 250 cycles)

Usage: python experiments/operating_point_grid.py [--passes 5]
Writes results/operating_point_grid.csv.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "idea1_adaptaccel"))

from base_simulator.cost_model import CostModel        # noqa: E402
from base_simulator.memory_model import MemoryModel    # noqa: E402
from base_simulator.layer_cost import get_layer_costs  # noqa: E402
import scheduler.dp_scheduler as sched                  # noqa: E402

DATA_DIR = os.path.join(REPO, "idea1_adaptaccel", "profiling_data")
ARRAYS = [8, 16, 32]
BWS = [2, 4, 8, 32]
SCALES = [1, 10, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=5)
    args = ap.parse_args()
    traces = {os.path.basename(p).replace("_traces.csv", ""): pd.read_csv(p)
              for p in sorted(glob.glob(os.path.join(DATA_DIR, "*_traces.csv")))}
    base = sched.SWITCH_COST.copy()
    init = sched.CONFIG_IDX["DMR"]
    rows = []
    for n in ARRAYS:
        for bw in BWS:
            cm = CostModel(Nr=n, Nc=n, memory=MemoryModel(hbm_bw_gbps=bw * 4, freq_ghz=1.0))
            costs = {}
            for name, df in traces.items():
                for pid in sorted(df["input_idx"].unique())[: args.passes]:
                    g = df[df["input_idx"] == pid]
                    costs[(name, pid)] = [
                        get_layer_costs(cm, r, max(float(r["input_sparsity"]),
                                                   float(r.get("attention_sparsity", 0.0) or 0.0)))
                        for _, r in g.iterrows()]
            for sc in SCALES:
                sched.SWITCH_COST = base * sc
                for (name, pid), c in costs.items():
                    orc = sched.rescore_schedule(sched.schedule_oracle(c)["schedule"], c)["total_cost"]
                    static = min(sum(x[k] for x in c) + sched.SWITCH_COST[init][sched.CONFIG_IDX[k]]
                                 for k in sched.CONFIG_NAMES)
                    rows.append(dict(array=n, bw=bw, switch_scale=sc, model=name, input_idx=pid,
                                     real=name.startswith(("resnet50_", "bert_")),
                                     speedup=static / orc))
            print(f"  {n}x{n} BW={bw:<2} done", flush=True)
    sched.SWITCH_COST = base
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(REPO, "results", "operating_point_grid.csv"), index=False)
    w = d.groupby(["array", "bw", "switch_scale", "model", "real"]).speedup.mean().reset_index()
    for label, sub in (("all 15 workloads", w), ("10 real workloads", w[w.real])):
        t = sub.groupby(["switch_scale", "array", "bw"]).speedup.mean().unstack("bw")
        t.columns = [f"BW={c}" for c in t.columns]
        print(f"\n=== Headline speedup (selection vs best static per workload), {label} ===")
        print(t.to_string(float_format=lambda v: f"{v:.3f}x"))


if __name__ == "__main__":
    main()
