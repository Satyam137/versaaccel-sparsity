"""B9: do the findings hold on two more model families?

Sections 2-6 rest on two families, ResNet-50 and BERT. This re-runs the
measurements that carry the paper's claims on MobileNetV2 (depthwise-separable
CNN, ReLU6, inverted residuals; profiled on the same Imagenette images) and
OPT-125m (decoder-only Transformer with ReLU feed-forward and causal attention;
the same SST-2 sentences), each dense and 50/70/90% magnitude-pruned:

  structure (2.5)    measured all-zero activation k-slices vs the i.i.d.
                     prediction mean(s^16) at the same density
  value (2.6)        realisable skip saving f*K/(K+NR+NC), work-weighted --
                     also without grouped (depthwise) layers, which run on a GEMM
                     array as a block-diagonal GEMM doing C times their real work,
                     almost all of it structurally zero; with them included the
                     saving mostly measures that recovered waste, not activation
                     structure
  scheduling (3.2)   per-layer selection vs best static configuration over all
                     200 inputs, and how many distinct optimal schedules occur

The new traces live in profiling_data/b9/ so that sections 3-6, computed on the
original 15 sets, are untouched; the ResNet-50 / BERT rows here are recomputed
the same way for side-by-side comparison (their scheduling numbers equal
results/scheduler_variance.csv).

Usage: python experiments/generalisation_b9.py [--passes N]   (~15 min)
Writes results/generalisation_b9.csv.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import scheduler_variance as sv                                  # noqa: E402

NR = NC = 16
DATA = os.path.join(REPO, "idea1_adaptaccel", "profiling_data")
FAMILY = {"resnet50": "ResNet-50 (CNN)", "mobilenet_v2": "MobileNetV2 (CNN, new)",
          "bert": "BERT-base (encoder)", "opt125m": "OPT-125m (decoder, new)"}


def family(name):
    return next(k for k in FAMILY if name.startswith(k + "_"))


def grouped_layers(path):
    """Layers that are grouped convolutions: in the DENSE model their (block-
    diagonal) GEMM weight is mostly zero, which no ungrouped layer's is. Only
    MobileNetV2 has any."""
    if not os.path.basename(path).startswith("mobilenet_v2_"):
        return set()
    dense = os.path.join(os.path.dirname(path),
                         os.path.basename(path).split("_pruned")[0].replace("_traces.csv", "")
                         + ("_dense_traces.csv" if "_pruned" in os.path.basename(path) else "_traces.csv"))
    d = pd.read_csv(dense, usecols=["layer_name", "weight_sparsity"])
    m = d.groupby("layer_name").weight_sparsity.mean()
    return set(m[m > 0.5].index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=None)
    args = ap.parse_args()
    paths = [p for p in sorted(glob.glob(os.path.join(DATA, "*_traces.csv")) +
                               glob.glob(os.path.join(DATA, "b9", "*_traces.csv")))
             if os.path.basename(p).startswith(tuple(k + "_" for k in FAMILY))]
    cm = sv.CostModel(Nr=sv.ARRAY_ROWS, Nc=sv.ARRAY_COLS)
    rows = []
    for p in paths:
        name = os.path.basename(p).replace("_traces.csv", "")
        d = pd.read_csv(p)
        work = (d.M * d.K * d.P).astype(float)
        w = work / work.sum()
        s = d.input_sparsity.clip(0, 1)
        iid = float((s ** NR).mean())
        ug = ~d.layer_name.isin(grouped_layers(p))
        wu = work[ug] / work[ug].sum()
        meas = float(d.a_kslice_zero.mean())
        per = []
        for pid in sorted(d.input_idx.unique())[: args.passes]:
            g = d[d.input_idx == pid].reset_index(drop=True)
            if len(g) >= 2:
                per.append(sv.evaluate_pass(g, cm))
        sp = np.array([r["speedup"] for r in per])
        rows.append(dict(
            workload=name, family=FAMILY[family(name)], new=family(name) in ("mobilenet_v2", "opt125m"),
            layers=d.layer_name.nunique(), passes=len(per),
            act_zero=float(s.mean()), act_kslice_zero=meas, iid_kslice_zero=iid,
            # no exact activation zeros (GELU) -> no ratio to speak of
            structure_vs_iid=meas / iid if s.mean() > 0.05 else float("nan"),
            weight_sparsity=float(d.weight_sparsity.mean()), w_kslice_zero=float(d.w_kslice_zero.mean()),
            skippable_frac=float((d.skippable_frac * w).sum()),
            realisable_saving=float((d.skippable_frac * d.K / (d.K + NR + NC) * w).sum()),
            grouped_layers=int(d[~ug].layer_name.nunique()),
            saving_ungrouped=float((d.skippable_frac * d.K / (d.K + NR + NC))[ug].mul(wu).sum()),
            speedup_mean=sp.mean(), speedup_ci95=1.96 * sp.std(ddof=1) / np.sqrt(len(sp)),
            unique_schedules=len({r["schedule"] for r in per})))
        r = rows[-1]
        print(f"  {name:24s} act zeros {r['act_zero']:.3f}  k-slice {r['act_kslice_zero']:.3f} "
              f"(iid {r['iid_kslice_zero']:.4f}, {r['structure_vs_iid']:6.1f}x)  "
              f"saving {r['realisable_saving']:6.1%} ({r['saving_ungrouped']:5.1%} w/o grouped)  speedup {r['speedup_mean']:.3f}x "
              f"+/- {r['speedup_ci95']:.3f}  schedules {r['unique_schedules']}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(REPO, "results", "generalisation_b9.csv"), index=False)
    print("\n=== By family ===")
    g = t.groupby("family", sort=False).agg(
        structure_vs_iid=("structure_vs_iid", lambda x: f"{x.min():.1f}-{x.max():.1f}x"),
        saving=("realisable_saving", lambda x: f"{100 * x.min():.1f}-{100 * x.max():.1f}%"),
        saving_ungrouped=("saving_ungrouped", lambda x: f"{100 * x.min():.1f}-{100 * x.max():.1f}%"),
        speedup=("speedup_mean", lambda x: f"{x.mean():.3f}x (max {x.max():.3f})"),
        input_dependent=("unique_schedules", lambda x: f"{int((x > 1).sum())}/{len(x)}"))
    print(g.to_string())


if __name__ == "__main__":
    main()
